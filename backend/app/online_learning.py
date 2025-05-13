# app/online_learning.py
import torch
import random
import time
import uuid
import pickle
from collections import deque
from pathlib import Path
from app.pipeline import InferenceDataPipeline
from app.highlighting import get_explanation
from app.model import integrated_gradients

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
STOP_WORDS = BASE_DIR / "artifacts/stopwords_set.pkl"
TFIDF = BASE_DIR / "artifacts/hate_tfidf_scores.pkl"


class OnlineHateSpeechDetector:
    def __init__(
        self,
        model,
        word_to_index,
        threshold,
        lr=0.0001,
        buffer_size=1000,
        update_frequency=10,
    ):
        """
        Initialize online learning wrapper for hate speech model.

        Args:
            model: Your HateSpeechDetectorFNN model
            word_to_index: Vocabulary mapping
            threshold: Classification threshold dictionary
            lr: Learning rate for online updates
            buffer_size: Maximum examples to store
            update_frequency: Update after this many examples
        """
        self.model = model
        self.word_to_index = word_to_index
        self.threshold = threshold
        self.device = next(model.parameters()).device

        # Online learning components
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Buffer for storing examples
        self.example_buffer = deque(maxlen=buffer_size)
        self.examples_since_update = 0
        self.update_frequency = update_frequency

        # Prediction cache for delayed feedback
        self.prediction_cache = {}
        self.cache_ttl = 60 * 60 * 24 * 7  # 7 days in seconds

        # Load TF-IDF scores and stopwords if available
        try:
            with open(TFIDF, "rb") as f:
                self.tfidf_scores = pickle.load(f)
            with open(STOP_WORDS, "rb") as f:
                self.stopwords_set = pickle.load(f)
            print(f"Loaded TF-IDF scores and stopwords")
        except FileNotFoundError:
            print("Warning: TF-IDF scores or stopwords not found")
            self.tfidf_scores = {}
            self.stopwords_set = set()

    def predict(self, text):
        """Make prediction and cache for possible feedback."""
        # Preprocess using existing pipeline
        input_tensor, tokens, original_tokens, position_mapping = self.preprocess_input(
            text
        )

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()
            label = prob >= self.threshold["best_threshold"]

        # Get attributions
        attributions = integrated_gradients(
            self.model, input_tensor, self.word_to_index, device=self.device
        )

        # Extract attribution values
        attribution_values = [a["attribution"] for a in attributions]

        # Map attributions to original text positions
        highlighted_segments = []
        mapped_indices = set()

        # First pass: try to map each position to a processed token
        for i, pos_map in enumerate(position_mapping):
            if pos_map["processed"] is not None:
                # Find the index of this processed token in our tokens list
                try:
                    token_idx = tokens.index(pos_map["processed"])
                    if (
                        token_idx < len(attribution_values)
                        and token_idx not in mapped_indices
                    ):
                        highlighted_segments.append(
                            {
                                "original_text": pos_map["original"]["text"],
                                "start": pos_map["original"]["start"],
                                "end": pos_map["original"]["end"],
                                "attribution": attribution_values[token_idx],
                                "processed_token": pos_map["processed"],
                            }
                        )
                        mapped_indices.add(token_idx)
                except ValueError:
                    # Token not found in processed list (might have been truncated)
                    pass

        # Sort segments by start position
        highlighted_segments.sort(key=lambda x: x["start"])

        # Enhance attributions with TF-IDF if available
        if hasattr(self, "tfidf_scores") and hasattr(self, "stopwords_set"):
            enhanced_attributions = get_explanation(
                tokens, attribution_values, self.tfidf_scores, self.stopwords_set
            )
        else:
            enhanced_attributions = attributions

        # Generate ID and cache prediction
        prediction_id = str(uuid.uuid4())
        self.prediction_cache[prediction_id] = {
            "input_tensor": input_tensor.clone(),
            "timestamp": time.time(),
            "text": text,
        }

        # Clean old cache entries
        self._clean_prediction_cache()

        return {
            "prediction_id": prediction_id,
            "text": text,
            "label": bool(label),
            "probability": prob,
            "tokens": tokens,
            "original_tokens": original_tokens,
            "explanation": enhanced_attributions,
            "highlighted_segments": highlighted_segments,
        }

    def submit_feedback(self, prediction_id, true_label):
        """Process feedback for a previous prediction."""
        # Check if prediction exists
        if prediction_id not in self.prediction_cache:
            return {"status": "error", "message": "Prediction ID not found or expired"}

        # Get cached prediction
        cached_data = self.prediction_cache[prediction_id]
        input_tensor = cached_data["input_tensor"]

        # Convert label to tensor
        true_label_tensor = torch.tensor([[float(true_label)]], dtype=torch.float32)

        # Add to buffer and possibly update
        self._add_to_buffer(input_tensor, true_label_tensor)
        updated = self._maybe_update_model()

        return {
            "status": "success",
            "message": "Feedback processed" + (" and model updated" if updated else ""),
            "text": cached_data["text"],
            "true_label": true_label,
        }

    def preprocess_input(self, text):
        """Preprocess text using existing pipeline."""
        
        # Get tokenized representation with position mapping
        indices, tokens, original_tokens, position_mapping = InferenceDataPipeline(text, self.word_to_index, 100)
        
        # Convert to tensor (indices are already processed by InferenceDataPipeline)
        tensor_input = torch.tensor([indices], dtype=torch.long, device=self.device)
        
        return tensor_input, tokens, original_tokens, position_mapping
    def _clean_prediction_cache(self):
        """Remove expired predictions from cache."""
        current_time = time.time()
        expired_ids = [
            pid
            for pid, data in self.prediction_cache.items()
            if current_time - data["timestamp"] > self.cache_ttl
        ]
        for pid in expired_ids:
            del self.prediction_cache[pid]

    def _add_to_buffer(self, input_tensor, true_label):
        """Add example to buffer."""
        self.example_buffer.append((input_tensor, true_label))
        self.examples_since_update += 1

    def _maybe_update_model(self):
        """Update model if enough examples collected."""
        if self.examples_since_update >= self.update_frequency and self.example_buffer:
            self._update_model()
            self.examples_since_update = 0
            return True
        return False

    def _update_model(self):
        """Update model weights using buffered examples."""
        # Set to training mode
        self.model.train()

        # Sample a batch
        batch_size = min(32, len(self.example_buffer))
        batch = random.sample(list(self.example_buffer), batch_size)

        # Reset gradients
        self.optimizer.zero_grad()

        # Calculate loss for each example
        for input_tensor, true_label in batch:
            # Forward pass
            outputs = self.model(input_tensor)

            # Calculate loss
            loss = self.criterion(outputs, true_label)

            # Backpropagate
            loss.backward()

        # Update weights
        self.optimizer.step()

        # Switch back to eval mode
        self.model.eval()

        print(f"Updated model with {batch_size} examples")
