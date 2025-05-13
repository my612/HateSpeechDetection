import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import unicodedata
import contractions
import spacy
from collections import Counter
import pickle
import spacy.cli
from tqdm import tqdm
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to artifacts
artifacts_path = os.path.join(os.path.dirname(current_dir), "artifacts", "offensive_en.txt")

# Use the path
with open(artifacts_path) as f:
    offensive_words = set(word.strip() for word in f)

# You do NOT need to load vocab.pkl here unless you're doing inference.
# It will be created anew during training prep.

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[\d_]+", " ", text)  # Remove numbers & underscores
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = contractions.fix(text)
    text = normalize_text(text)
    # (Optional: Handle emojis here)
    # text = emoji.demojize(text) 
    # Remove punctuation, replace non-words with space
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\W+", " ", text)  # Remove special characters
    text = text.strip()
    return text

def dataCleaningPipeline(df):
    df.drop(df[df["text"].apply(type) == float].index, inplace=True)
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(clean_text)
    return df

def saveBadWords(text, offensive_words):
    words = text.split()
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if phrase in offensive_words:
            text = text.replace(phrase, phrase.replace(" ", "_"))
    return text

def lemmatize_texts(df, column="text", offensive_words=None):
    texts = df[column].tolist()
    lemmatized_batches = []
    for doc in tqdm(nlp.pipe(texts, batch_size=500), total=len(texts), desc="Lemmatizing"):
        tokens = [
            token.lemma_ if "_" not in token.text and (offensive_words is None or token.text not in offensive_words) else token.text
            for token in doc
        ]
        lemmatized_batches.append([w for w in tokens if w.strip()])
    df[column] = lemmatized_batches
    return df

class TextPreprocessor:
    def __init__(self):
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
    def fit(self, texts, min_freq=3):
        counter = Counter(word for sent in texts for word in sent)
        # Retain starter tokens explicitly
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word_to_index:
                self.word_to_index[word] = len(self.word_to_index)
        # Build inverse dict
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
    def transform(self, texts):
        return [[self.word_to_index.get(word, 1) for word in sentence] for sentence in texts]
    def save_vocab(self, path="vocab.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.word_to_index, f)
    def load_vocab(self, path="vocab.pkl"):
        with open(path, "rb") as f:
            self.word_to_index = pickle.load(f)
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
    def get_vocab(self):
        return self.word_to_index

def dataPipeline(df, min_freq=3):
    df = dataCleaningPipeline(df)
    # Only need to pass in offensives once
    df["text"] = df["text"].apply(lambda x: saveBadWords(x, offensive_words))
    # Lemmatize and remove empty tokens
    df = lemmatize_texts(df, column="text", offensive_words=offensive_words)
    # Remove any lingering empty tokens
    df["text"] = df["text"].apply(lambda x: [word for word in x if word.strip()])
    preprocessor = TextPreprocessor()
    preprocessor.fit(df["text"], min_freq=min_freq)
    df["numerical_text"] = preprocessor.transform(df["text"])
    vocab = preprocessor.get_vocab()
    return df, vocab

def InferenceDataPipeline(text, word_to_index, max_seq_length=100):
    # Store original text
    original_text = text
    
    # Split into original tokens for reference
    original_tokens = text.split()
    
    # Initial position tracking
    position_mapping = []
    current_pos = 0
    for orig_token in original_tokens:
        # Find actual position in original text (accounting for whitespace)
        while current_pos < len(original_text) and original_text[current_pos].isspace():
            current_pos += 1
        
        word_start = current_pos
        word_end = word_start + len(orig_token)
        current_pos = word_end
        
        # Store original token info
        position_mapping.append({
            "original": {
                "text": orig_token,
                "start": word_start,
                "end": word_end
            },
            "processed": None  # Will fill this later
        })
    
    # Continue with regular processing
    text = text.lower()  # Lowercase
    text = clean_text(text)
    text = saveBadWords(text, offensive_words)  # Use global set already loaded
    
    # Lemmatize (returns list of tokens)
    doc = nlp(text)
    tokens = [
        token.lemma_ if "_" not in token.text and token.text not in offensive_words else token.text
        for token in doc
    ]
    tokens = [word for word in tokens if word.strip()]
    
    # Map processed tokens back to original tokens
    # This is tricky since lemmatization may change token count
    # We'll use a best-effort approach based on token similarity
    
    # If counts match exactly, use direct mapping
    if len(tokens) == len(original_tokens):
        for i, processed_token in enumerate(tokens):
            position_mapping[i]["processed"] = processed_token
    else:
        # Use more complex alignment based on string matching
        # This is a simplified approach - you may need a more sophisticated alignment algorithm
        orig_index = 0
        for processed_token in tokens:
            # Skip empty tokens
            if not processed_token.strip():
                continue
                
            # Find best matching original token
            while orig_index < len(position_mapping):
                orig_token_lower = position_mapping[orig_index]["original"]["text"].lower()
                # Check if this processed token could have come from this original token
                if (processed_token in orig_token_lower or 
                    orig_token_lower in processed_token or
                    levenshtein_distance(processed_token, orig_token_lower) <= 2):
                    position_mapping[orig_index]["processed"] = processed_token
                    orig_index += 1
                    break
                orig_index += 1
                
            # If we run out of original tokens, stop mapping
            if orig_index >= len(position_mapping):
                break
    
    # Transform tokens to indices with word_to_index
    indices = [word_to_index.get(t, word_to_index['<UNK>']) for t in tokens]
    
    # Pad or truncate
    if len(indices) < max_seq_length:
        indices += [word_to_index['<PAD>']] * (max_seq_length - len(indices))
    else:
        indices = indices[:max_seq_length]
        tokens = tokens[:max_seq_length]
    
    return indices, tokens, original_tokens, position_mapping

# Helper function for string similarity
def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
def getTrainingProcessedData(df):
    df, vocab = dataPipeline(df)
    return df, vocab

# Example usage:
# df = pd.read_csv('../../../../HateSpeechDetector/en_hf_112024.csv')
# df, vocab = dataPipeline(df)
# df.to_pickle("processed.pkl")
# with open("vocab.pkl", "wb") as f:
    # pickle.dump(vocab, f)
#     
