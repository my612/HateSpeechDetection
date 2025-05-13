import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
from ray import tune
from ray.air import session  # Fix for tune.report issue
import json
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance (should be HIGH for minority class!)
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # Per class alpha (high for positive/minority class)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


class HateSpeechDataset(Dataset):
    def __init__(self, numerical_texts, labels, max_seq_length, pad_idx):
        self.numerical_texts = numerical_texts
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.numerical_texts)

    def __getitem__(self, idx):
        text = self.numerical_texts[idx][:self.max_seq_length]
        text = text[:self.max_seq_length] if len(text) > self.max_seq_length else text + [self.pad_idx] * (self.max_seq_length - len(text))
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


# Data Loading
parent = Path(__file__).parent.parent
df = pd.read_pickle(parent / "processed.pkl")
vocab_path = parent / "vocab.pkl"
with open(vocab_path, "rb") as f:
    word_to_index = pickle.load(f)

max_seq_length = 100
pad_idx = word_to_index['<PAD>']

numerical_texts = df["numerical_text"].tolist()
labels = df["labels"].tolist()

dataset = HateSpeechDataset(numerical_texts, labels, max_seq_length, pad_idx)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


def train_hatespeech(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = HateSpeechDetectorFNN(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hid_dim=config["hid_dim"],
        output_dim=1,
        pad_idx=config["pad_idx"],
        max_seq_length=config["max_seq_length"]
    ).to(device)

    criterion = FocalLoss(gamma=config['gamma'], alpha=config['alpha'])  # Corrected alpha handling!
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * texts.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        f_score = f1_score(all_labels, all_preds, average="macro")
        session.report({
            "f1": f_score,
            "train_loss": avg_train_loss,
        })

    torch.cuda.empty_cache()


def hyperparameter_search():
    config = {
        "vocab_size": len(word_to_index),
        "embed_dim": tune.choice([64, 128, 256]),
        "hid_dim": tune.choice([128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "num_epochs": 7,  # Increased for better learning
        "max_seq_length": max_seq_length,
        "pad_idx": pad_idx,
        "gamma": tune.choice([0.5, 1.0, 2.0]),          # Sweeps only to 2.0
        "alpha": tune.uniform(0.70, 0.95),              # Biases toward minority class
    }

    analysis = tune.run(
        train_hatespeech,
        config=config,
        metric="f1",
        mode="max",
        num_samples=30,
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
    )

    best_trial = analysis.get_best_trial("f1", "max", "last")
    return best_trial.config


def train_final_model(best_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

    model = HateSpeechDetectorFNN(
        vocab_size=best_params["vocab_size"],
        embed_dim=best_params["embed_dim"],
        hid_dim=best_params["hid_dim"],
        output_dim=1,
        pad_idx=best_params["pad_idx"],
        max_seq_length=best_params["max_seq_length"]
    ).to(device)

    criterion = FocalLoss(gamma=best_params['gamma'], alpha=best_params['alpha'])
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

    for epoch in range(10):  # More epochs for final fine-tune
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Save the trained model
    torch.save(model.state_dict(), "hate_speech_model-nn-focal.pth")
    print("Model saved as 'hate_speech_model-nn-focal.pth'")

    # Save best hyperparameters
    with open("best_hyperparams-nn-focal.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best hyperparameters saved as 'best_hyperparams-nn-focal.json'")

    # Generate classification report
    final_report = classification_report(all_labels, all_preds, digits=4)
    final_f1_score = f1_score(all_labels, all_preds, average='macro')
    print("\nFinal Classification Report:\n", final_report)
    print("Final F1 Score:", final_f1_score)
    with open("NN-focal-final_classification_report.txt", "w") as f:
        f.write(final_report)
        f.write(f"\nFinal F1 Score: {final_f1_score:.4f}\n")


if __name__ == "__main__":
    best_params = hyperparameter_search()
    train_final_model(best_params)