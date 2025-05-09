#%%
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from app.pipeline import InferenceDataPipeline
#%%
class HateSpeechDetectorFNN(nn.Module):
    def __init__(self, vocab_size, hid_dim, embed_dim, output_dim, pad_idx, max_seq_length):
        super(HateSpeechDetectorFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim * max_seq_length, hid_dim)
        self.fc2 = nn.Linear(hid_dim, output_dim)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.view(embedded.shape[0], -1)
        hidden = self.fc1(embedded)
        hidden = self.ln1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output
#%%

# 2. Model/Artifact loading at startup
BASE_DIR = Path(__file__).parent.parent

MODEL_WEIGHTS = BASE_DIR / "artifacts/hate_speech_model-nn-focal.pth"
VOCAB_PATH    = BASE_DIR / "artifacts/vocab.pkl"
HYPERPARAMS   = BASE_DIR / "artifacts/best_hyperparams-nn-focal.json"

# Hyperparameters needed for model instantiation
with open(HYPERPARAMS, "r") as f:
    params = json.load(f)
with open(VOCAB_PATH, "rb") as f:
    word_to_index = pickle.load(f)

# Model instantiation (match code train args order)
def get_model():
    model = HateSpeechDetectorFNN(
        vocab_size=params["vocab_size"],
        embed_dim=params["embed_dim"],
        hid_dim=params["hid_dim"],
        output_dim=1,  # assume binary
        pad_idx=params["pad_idx"],
        max_seq_length=params["max_seq_length"]
    )
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"))
    model.eval()
    return model

model = get_model()  # global model instance
#%%
def preprocess_input(text, word_to_index, max_seq_length=100):
    i, tokens = InferenceDataPipeline(text, word_to_index, max_seq_length)
    indices = [word_to_index.get(tok, word_to_index['<UNK>']) for tok in tokens]
    # Pad or truncate
    if len(indices) < max_seq_length:
        indices += [word_to_index['<PAD>']] * (max_seq_length - len(indices))
    else:
        indices = indices[:max_seq_length]
    tensor_input = torch.tensor([indices], dtype=torch.long)  # shape (1, 100)
    return tensor_input, tokens
#%%
def predict_from_tensor(input_tensor):
    """input_tensor: torch (1, max_seq_length, dtype=long)"""
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()  # scalar
        label = prob >= 0.5
    return label, prob
#%%
test_case = "Islam and ISIS should go to hell. All Muslims should be immediately sent to their country, because they are all intolerant criminals. If we do so, Britain will be a safer place."
ten_i, tokens = preprocess_input(test_case, word_to_index=word_to_index)

def predict(text):
    ten_i, tokens = preprocess_input(text, word_to_index=word_to_index)
    label, prob = predict_from_tensor(ten_i)
    return {
        "label": label,
        "prob": prob,
        "tokens": tokens
    }

# %%
predict_from_tensor(ten_i)
# %%
