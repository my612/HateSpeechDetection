import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, average_precision_score
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

# ==== Model & Loss ==== #
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class HateSpeechDetectorFNN(nn.Module):
    def __init__(self, vocab_size, hid_dim, embed_dim, output_dim, pad_idx, max_seq_length):
        super().__init__()
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
        text = text if len(text) > self.max_seq_length else text + [self.pad_idx] * (self.max_seq_length - len(text))
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# ==== Data Load ==== #
def get_dataset():
    parent = Path(__file__).parent.parent
    df = pd.read_pickle(parent / "processed.pkl")
    vocab_path = parent / "vocab.pkl"
    with open(vocab_path, "rb") as f:
        word_to_index = pickle.load(f)
    pad_idx = word_to_index['<PAD>']
    max_seq_length = 100
    numerical_texts = df["numerical_text"].tolist()
    labels = df["labels"].tolist()
    dataset = HateSpeechDataset(numerical_texts, labels, max_seq_length, pad_idx)
    return dataset, word_to_index, pad_idx, max_seq_length

def make_weighted_loader(dataset, batch_size):
    labels_in_dataset = [dataset[i][1].item() for i in range(len(dataset))]
    class_sample_count = np.array([len([l for l in labels_in_dataset if l == t]) for t in [0,1]])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(l)] for l in labels_in_dataset])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ==== Main Train/Fine-tune Function ==== #
def train_or_finetune(finetune=True, n_epochs=10):
    # Load data/hyperparams/model/checkpoint
    dataset, word_to_index, pad_idx, max_seq_length = get_dataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Load best params:
    with open('best_hyperparams-nn-focal.json', 'r') as f:
        params = json.load(f)
    batch_size = params["batch_size"]
    train_loader = make_weighted_loader(train_dataset, batch_size)
    test_loader = make_loader(test_dataset, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = HateSpeechDetectorFNN(
        vocab_size=params["vocab_size"],
        embed_dim=params["embed_dim"],
        hid_dim=params["hid_dim"],
        output_dim=1,
        pad_idx=params["pad_idx"],
        max_seq_length=params["max_seq_length"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"] * (0.1 if finetune else 1.0))
    criterion = FocalLoss(alpha=params["alpha"], gamma=params["gamma"])

    # Resume weights if fine-tuning
    if finetune and os.path.exists('hate_speech_model-nn-focal.pth'):
        model.load_state_dict(torch.load('hate_speech_model-nn-focal.pth', map_location=device))
        print("Loaded previous model for fine-tuning.")
    else:
        print("Starting fresh training.")

    # LR SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # TRAIN LOOP
    best_f1 = 0
    patience = 3
    wait = 0
    for epoch in range(n_epochs):
        model.train(); tot_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(texts).squeeze(1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * texts.size(0)
        avg_loss = tot_loss / len(train_loader.dataset)
        # EVAL on test
        model.eval(); all_logits = []; all_y = []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts).squeeze(1)
                all_logits.extend(logits.cpu().numpy())
                all_y.extend(labels.cpu().numpy())
        all_logits = np.array(all_logits); all_y = np.array(all_y)
        all_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        # Threshold sweep
        epoch_best_f1, epoch_best_thr = 0, 0.5
        for t in np.arange(0.1, 0.91, 0.02):
            preds = (all_probs >= t).astype(float)
            f1 = f1_score(all_y, preds, average="macro")
            if f1 > epoch_best_f1:
                epoch_best_f1 = f1
                epoch_best_thr = t
        print(f"Epoch {epoch+1}/{n_epochs}: TrainLoss {avg_loss:.4f}, Macro F1 {epoch_best_f1:.4f} (Thr={epoch_best_thr:.2f})")
        scheduler.step(epoch_best_f1)
        # EARLY STOP
        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            wait = 0
            torch.save(model.state_dict(), "hate_speech_model-nn-focal.pth")
            with open("hate_speech_model_bestthr.json", "w") as f:
                json.dump(dict(best_threshold=float(epoch_best_thr)), f)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # FINAL EVAL
    model.load_state_dict(torch.load("hate_speech_model-nn-focal.pth", map_location=device))
    thr = 0.5
    if os.path.exists("hate_speech_model_bestthr.json"):
        with open("hate_speech_model_bestthr.json", "r") as f:
            thr = json.load(f).get("best_threshold", 0.5)
    model.eval(); all_logits=[]; all_y=[]
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts).squeeze(1)
            all_logits.extend(logits.cpu().numpy())
            all_y.extend(labels.cpu().numpy())
    all_logits = np.array(all_logits); all_y = np.array(all_y)
    all_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    final_preds = (all_probs >= thr).astype(float)
    report = classification_report(all_y, final_preds, digits=4)
    roc_auc = roc_auc_score(all_y, all_probs)
    pr_auc = average_precision_score(all_y, all_probs)
    print("Final results with best threshold:")
    print(report)
    print(f"ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")
    with open("NN-focal-final_classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nBest threshold: {thr:.2f}\nMacro F1: {f1_score(all_y, final_preds, average='macro'):.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\nPR-AUC: {pr_auc:.4f}\n")
    print("Results saved.")

if __name__ == "__main__":
    # Set finetune=True to resume, False for fresh
    train_or_finetune(finetune=True, n_epochs=5)