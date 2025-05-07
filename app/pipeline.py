import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import unicodedata
import contractions
import spacy
from collections import defaultdict
import pickle


def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    return text



def dataCleaningPipeline(df):
    df.drop(df[df["text"].apply(type) == float].index, inplace=True)
    # Conversion to lowercase
    df = df.map(lambda s: s.lower() if type(s) == str else s)

    df['text'] = df['text'].apply(lambda text: contractions.fix(text))
    #Deleted a NaN from text

    df["text"] = df["text"].apply(normalize_text)


    df["text"] = df["text"].apply(lambda x: re.sub(r"_+", " ", x))  # Replace multiple underscores with space
    df['text'] = df['text'].apply(lambda text: clean_text(text))
    
    return df

def saveBadWords(text, offensive_words=None):
    words = text.split()
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if phrase in offensive_words:
            text = text.replace(phrase, phrase.replace(" ", "_"))
            # print(f"Replaced '{phrase}' with '{phrase.replace(' ', '_')}'")
    return text


def lemmatize(text, offensive_words=None, nlp=None):
    doc = nlp(text)
    tokens = []
    words = text.split()
    for token in doc:
        if "_" not in token.text and token.text not in offensive_words:  # Skip multi-word phrases
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)
    return tokens
def lemmatize_texts(df, column="text", offensive_words=None):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    texts = df[column].tolist()  # Extract column as a list
    lemmatized_batches = []

    for i, doc in enumerate(nlp.pipe(texts, batch_size=500)):
        tokens = [token.lemma_ if "_" not in token.text and token.text not in offensive_words else token.text
                  for token in doc]
        lemmatized_batches.append(tokens)

        if (i + 1) % 500 == 0:
            print(f"Batch {i + 1} done...")

    df[column] = lemmatized_batches  # Assign back to DataFrame
    return df




class TextPreprocessor:
    def __init__(self):
        self.word_to_index = defaultdict(lambda: 1)
        self.index_to_word = {}
        self.word_to_index['<UNK>'] = 1
        self.word_to_index['<PAD>'] = 0
        
    def _default_index(self):
        return self.word_to_indexU['<UNK>']    
    
    def fit(self, texts):
        for sentence in texts:
            for word in sentence:
                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)    
        self.index_to_word = {idx: word for idx, word in self.word_to_index.items()} 
        
    def transform(self, texts):
        return [[self.word_to_index[word] for word in sentence] for sentence in texts]
    
    def save_vocab(self, path='vocab.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.word_to_index), f)
        
    def load_vocab(self, path='vocab.pkl'):
        with open(path, 'rb') as f:
            self.word_to_index = pickle.load(path, f)
            self.index_to_word = {idx: word for idx, word in self.word_to_index.items()}
    def get_vocab(self):
        return self.word_to_index
def dataPipeline(df):
    df = dataCleaningPipeline(df)
    with open('offensive_en.txt') as f:
        offensive_words = set(word.strip() for word in f)
    df["text"] = df["text"].apply(lambda x: saveBadWords(x, offensive_words=offensive_words))

    print("Saving bad words")
    #nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    #df["text"] = df["text"].apply(lambda x: lemmatize(x, offensive_words=offensive_words, nlp=nlp))
    df = df.pipe(lemmatize_texts, column="text", offensive_words=offensive_words)

    print("Lemmatizing")
    df["text"] = df["text"].apply(lambda x: [word for word in x if word.strip()])
    
    preprocessor = TextPreprocessor()

    preprocessor.fit(df["text"])

    df["numerical_text"] = preprocessor.transform(df["text"])
    
    vocab = preprocessor.get_vocab()
    
    return df, vocab


def getTrainingProcessedData(df):
    df, vocab = dataPipeline(df)
    return df, vocab

def getPredictionProcessedData(text):
    df = pd.DataFrame({"text": [text]})
    df, vocab = dataPipeline(df)
    return df, vocab
