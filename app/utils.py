import os
import json
import pickle
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression

# Shared model for embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

import nltk

nltk.data.path.append("./nltk_data")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir="./nltk_data")

from nltk.tokenize import sent_tokenize
def load_transcript(file):
    content = file.read().decode("utf-8")
    # Break into proper sentences using punctuation
    sentences = sent_tokenize(content)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# def load_transcript(file):
#     content = file.read().decode("utf-8")
#     sentences = [line.strip() for line in content.split("\n") if line.strip()]
#     return sentences

def embed_sentences(sentences):
    return embedder.encode(sentences, convert_to_tensor=True)

def generate_candidates(transcript_sentences, example_sentences, top_n=25):
    transcript_embeddings = embed_sentences(transcript_sentences)
    example_embeddings = embed_sentences(example_sentences)

    scores = util.cos_sim(transcript_embeddings, example_embeddings).max(axis=1).values.cpu().numpy()
    idxs = np.argsort(-scores)[:top_n]

    return [{"text": transcript_sentences[i]} for i in idxs]  # Make sure we return dicts, not strings

def train_classifier(candidates, labels, examples, model_path, model):
    texts = [c if isinstance(c, str) else c["text"] for c in candidates]  # Fallback for any str
    sentence_embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()

    clf = LogisticRegression()
    clf.fit(sentence_embeddings, labels)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

def save_concept(name, examples, model_path):
    os.makedirs("data/concepts", exist_ok=True)
    concept_obj = {
        "name": name,
        "examples": examples,
        "model_path": model_path
    }
    with open(f"data/concepts/{name.lower().replace(' ', '_')}.json", "w") as f:
        json.dump(concept_obj, f)

def get_concepts():
    concept_files = os.listdir("data/concepts")
    all_concepts = []
    for file in concept_files:
        if file.endswith(".json"):
            with open(f"data/concepts/{file}", "r") as f:
                all_concepts.append(json.load(f))
    return all_concepts

def load_concept_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_concept_matches(sentences, model, examples):
    embeddings = embedder.encode(sentences, convert_to_tensor=True).cpu().numpy()
    preds = model.predict(embeddings)
    return [s for s, label in zip(sentences, preds) if label == 1]
