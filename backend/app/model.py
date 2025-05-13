# %%
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from app.pipeline import InferenceDataPipeline
from app.highlighting import get_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HateSpeechDetectorFNN(nn.Module):
    def __init__(
        self, vocab_size, hid_dim, embed_dim, output_dim, pad_idx, max_seq_length
    ):
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


# %%

# 2. Model/Artifact loading at startup
BASE_DIR = Path(__file__).parent.parent

MODEL_WEIGHTS = BASE_DIR / "artifacts/hate_speech_model-nn-focal.pth"
VOCAB_PATH = BASE_DIR / "artifacts/vocab.pkl"
HYPERPARAMS = BASE_DIR / "artifacts/best_hyperparams-nn-focal.json"
BEST_THRESHOLD = BASE_DIR / "artifacts/hate_speech_model_bestthr.json"
# Hyperparameters needed for model instantiation
with open(HYPERPARAMS, "r") as f:
    params = json.load(f)
with open(VOCAB_PATH, "rb") as f:
    word_to_index = pickle.load(f)
with open(BEST_THRESHOLD, "r") as f:
    threshold = json.load(f)


# Model instantiation (match code train args order)
def get_model():
    model = HateSpeechDetectorFNN(
        vocab_size=params["vocab_size"],
        embed_dim=params["embed_dim"],
        hid_dim=params["hid_dim"],
        output_dim=1,  # assume binary
        pad_idx=params["pad_idx"],
        max_seq_length=params["max_seq_length"],
    )
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"))
    model.eval()
    return model


model = get_model()  # global model instance


# %%
def preprocess_input(text, word_to_index, max_seq_length=100):
    i, tokens, original_tokens, position_mapping = InferenceDataPipeline(
        text, word_to_index, max_seq_length)
    indices = [word_to_index.get(tok, word_to_index["<UNK>"]) for tok in tokens]
    # Pad or truncate
    if len(indices) < max_seq_length:
        indices += [word_to_index["<PAD>"]] * (max_seq_length - len(indices))
    else:
        indices = indices[:max_seq_length]
    tensor_input = torch.tensor([indices], dtype=torch.long)  # shape (1, 100)
    return tensor_input, tokens, original_tokens, position_mapping


# %%


def compute_token_saliency(model, input_indices, vocab, device="cpu"):
    # input_indices: (seq_len,) torch.LongTensor
    # vocab: your index to word (or word to index)

    model.eval()
    input_indices = input_indices.unsqueeze(0).to(device)  # (1, seq_len)
    # Enable grad on embeddings for THIS sample
    embedding_layer = model.embedding
    embedding_layer.weight.requires_grad = True

    # Forward
    output_logits = model(input_indices)  # (1, 1) shape
    logit = output_logits.squeeze()  # scalar

    # Backward wrt logit
    model.zero_grad()
    logit.backward()

    # Now embedding_layer.weight.grad is (vocab_size, emb_dim)
    # But we want the grad for **just the tokens in this input**
    input_tokens = input_indices[0]  # (seq_len,)

    # For each position, get grad for embedding of that token
    grads = embedding_layer.weight.grad  # (vocab_size, emb_dim)

    # For each position in seq, select grad row corresponding to index at that pos
    saliency = []
    index_to_word = {idx: word for word, idx in vocab.items()}
    for idx in input_tokens:
        idx = idx.item()
        token_grad = grads[idx]  # (emb_dim,)
        sal_score = token_grad.abs().sum().item()
        tok = index_to_word.get(idx, "<UNK>")
        saliency.append({"text": tok, "saliency": sal_score})

    # Normalize saliency to [0,1] across input tokens
    sal_values = [s["saliency"] for s in saliency]
    max_sal = max(sal_values) if max(sal_values) > 0 else 1.0
    min_sal = min(sal_values)
    for s in saliency:
        s["saliency"] = (
            (s["saliency"] - min_sal) / (max_sal - min_sal)
            if max_sal > min_sal
            else 0.0
        )

    return saliency


def integrated_gradients(
    model,
    input_indices,  # Tensor: shape (1, seq_len)
    word_to_index,
    target_class=None,  # Not needed for binary. We'll use model output before sigmoid.
    baseline_indices=None,
    m_steps=50,
    device="cpu",
):
    """
    Returns a list of (token, attribution) for your input.
    input_indices: torch.tensor shape (1, seq_len)
    """
    model.eval()
    input_indices = input_indices.to(device)
    batch_size, seq_len = input_indices.shape

    embedding_layer = model.embedding
    # Prepare baseline (all <PAD> or <UNK>)
    if baseline_indices is None:
        baseline_indices = torch.full_like(input_indices, word_to_index["<PAD>"])
    else:
        baseline_indices = baseline_indices.to(device)

    # Get embedding for baseline and input
    baseline_emb = embedding_layer(baseline_indices)  # (1, seq_len, emb_dim)
    input_emb = embedding_layer(input_indices)  # (1, seq_len, emb_dim)

    # Integrated gradients
    interpolated_embeds = []
    alphas = torch.linspace(0, 1, m_steps + 1).to(device)  # from 0 to 1

    total_gradients = torch.zeros_like(input_emb)

    for alpha in alphas:
        step_emb = baseline_emb + alpha * (input_emb - baseline_emb)
        step_emb = step_emb.clone().detach().requires_grad_(True)

        # FNN expects flat input, so flatten
        step_flat = step_emb.view(batch_size, -1)
        output = model.fc2(model.dropout(model.relu(model.ln1(model.fc1(step_flat)))))
        # [OPTIONAL] If your model has more complicated forward(), wrap as one function
        # output = model(input_embs_as_input) # but you'll need to change model to accept embeddings

        score = output.squeeze()  # Get logit for class 1
        model.zero_grad()
        if step_emb.grad is not None:
            step_emb.grad.zero_()
        score.backward(retain_graph=True)

        # Get grad wrt step_emb
        grads = step_emb.grad.detach()  # shape (1, seq_len, emb_dim)
        total_gradients += grads

    # Average gradient
    avg_gradients = total_gradients / (m_steps + 1)

    # IG = (input_emb - baseline_emb) * avg_gradients ; sum over emb_dim for saliency per token
    attributions = (input_emb - baseline_emb) * avg_gradients  # (1, seq_len, emb_dim)
    attributions = attributions.abs().sum(dim=2).squeeze(0)  # (seq_len,)

    # Normalize for pretty plotting
    attributions = attributions.detach().cpu().numpy()
    max_attr = attributions.max() if attributions.max() > 0 else 1.0
    attributions_norm = attributions / max_attr

    # Build output: map input indices (unpad) to tokens and attributions
    input_tokens = input_indices[0].cpu().numpy()
    idx_to_word = {v: k for k, v in word_to_index.items()}  # index: token
    result = []
    for idx, attr in zip(input_tokens, attributions_norm):
        result.append(
            {"text": idx_to_word.get(idx, "<UNK>"), "attribution": float(attr)}
        )
    return result


def predict_from_tensor(input_tensor):
    """input_tensor: torch (1, max_seq_length, dtype=long)"""
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()  # scalar
        label = prob >= threshold["best_threshold"]
    return label, prob


# %%
test_case = "Islam and ISIS should go to hell. All Muslims should be immediately sent to their country, because they are all intolerant criminals. If we do so, Britain will be a safer place."
ten_i, tokens, original_tokens, position_mapping = preprocess_input(
    test_case, word_to_index=word_to_index
)


def predict(text):
    ten_i, tokens, original_tokens, position_mapping = preprocess_input(text, word_to_index=word_to_index)
    label, prob = predict_from_tensor(ten_i)
    return {"label": label, "prob": prob, "tokens": tokens}


def explain(text):
    # Preprocess
    input_tensor, tokens, original_tokens, position_mapping = preprocess_input(
        text, word_to_index, max_seq_length=100
    )  # (1, seq_len)
    attributions = integrated_gradients(
        model, input_tensor, word_to_index, device=device
    )

    tokens = [a["text"] for a in attributions]
    attributions = [a["attribution"] for a in attributions]
    result = get_explanation(tokens, attributions)
    return result, original_tokens
