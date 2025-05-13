# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import pickle
from pathlib import Path
import logging
# Import your model
from app.model import HateSpeechDetectorFNN
from app.online_learning import OnlineHateSpeechDetector
from app.feedback_queue import FeedbackQueue
from fastapi.middleware.cors import CORSMiddleware


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# API models
class TextInput(BaseModel):
    text: str


class FeedbackInput(BaseModel):
    prediction_id: str
    true_label: int

class SkipFeedbackInput(BaseModel):
    prediction_id: str
# Create FastAPI app
app = FastAPI(
    title="Hate Speech Detection API",
    description="API for detecting hate speech in text",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variables
online_detector = None
feedback_queue = None


# Load at startup
@app.on_event("startup")
def load_resources():
    global online_detector
    global feedback_queue
    # Path
    BASE_DIR = Path(__file__).parent
    MODEL_WEIGHTS = BASE_DIR / "artifacts/hate_speech_model-nn-focal.pth"
    VOCAB_PATH = BASE_DIR / "artifacts/vocab.pkl"
    HYPERPARAMS = BASE_DIR / "artifacts/best_hyperparams-nn-focal.json"
    BEST_THRESHOLD = BASE_DIR / "artifacts/hate_speech_model_bestthr.json"

    # Load resources
    with open(HYPERPARAMS, "r") as f:
        params = json.load(f)
    with open(VOCAB_PATH, "rb") as f:
        word_to_index = pickle.load(f)
    with open(BEST_THRESHOLD, "r") as f:
        threshold = json.load(f)

    # Create model
    model = HateSpeechDetectorFNN(
        vocab_size=params["vocab_size"],
        embed_dim=params["embed_dim"],
        hid_dim=params["hid_dim"],
        output_dim=1,
        pad_idx=params["pad_idx"],
        max_seq_length=params["max_seq_length"],
    )
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"))
    model.eval()

    # Initialize online detector
    online_detector = OnlineHateSpeechDetector(
        model=model, word_to_index=word_to_index, threshold=threshold
    )

    feedback_queue = FeedbackQueue(max_size=500)
    print("Model loaded successfully")
    logger.info("Model loaded successfully")


# Endpoints
@app.post("/predict")
def predict_endpoint(input_data: TextInput):
    """Endpoint for prediction, returns ID for later feedback."""
    try:
        # Debug information
        logger.info(f"Text to analyze: {input_data.text}")

        # Check if online_detector is initialized
        if online_detector is None:
            logger.error("online_detector is None! It wasn't properly initialized.")
            return {"error": "Model not initialized"}

        # Print detector information
        logger.info(f"Type of detector: {type(online_detector)}")
        logger.info(
            f"Available methods: {[m for m in dir(online_detector) if not m.startswith('_')]}"
        )

        # Try calling the predict method
        result = online_detector.predict(input_data.text)

        # Log successful prediction
        logger.info(
            f"Prediction successful: {result['label']} with probability {result['probability']}"
        )

        # Add to feedback queue
        feedback_queue.add_prediction(
            result["prediction_id"],
            result["text"],
            result["label"],
            result["probability"],
        )
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/next")
def get_next_feedback_item():
    """Get the next item needing feedback."""

    item = feedback_queue.get_next_feedback_item()

    if not item:

        return {"status": "empty", "message": "No items pending feedback"}

    return {
        "status": "success",
        "prediction_id": item["prediction_id"],
        "text": item["text"],
        "predicted_label": item["predicted_label"],
        "probability": item["probability"],
    }


@app.post("/feedback")
def feedback_endpoint(input_data: FeedbackInput):
    """Submit feedback for a prediction."""
    try:
        # Process the feedback with the online learning model
        result = online_detector.submit_feedback(
            input_data.prediction_id, input_data.true_label
        )
        # Mark as completed in the queue
        feedback_queue.mark_feedback_complete(
            input_data.prediction_id, bool(input_data.true_label)
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/skip")
def skip_feedback(input_data: SkipFeedbackInput):
    if feedback_queue.mark_feedback_skipped(input_data.prediction_id):
        return {"status": "success", "message": "Feedback skipped"}
    else:
        raise HTTPException(status_code=404, detail="Prediction ID not found")

@app.get("/feedback/stats")
def get_feedback_stats():
    """Get statistics about feedback collection."""
    return feedback_queue.get_feedback_stats()


@app.get("/")
def root():
    return {"message": "Welcome to the Hate Speech Detection API!"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "API is running!"}


# Run the app with: uvicorn app.main:app --reload
if __name__ == "__main__":

    port = int(os.getenv("PORT", 8000))

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
