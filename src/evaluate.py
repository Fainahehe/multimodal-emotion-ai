
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

FEATURE_PATH = Path(
    "data/MELD.Features.Models/features/audio_emotion.pkl"
)

# Load the saved model (no retraining)
bundle = joblib.load("models/multimodal_model.joblib")

tfidf = bundle["tfidf"]
scaler = bundle["scaler"]
model = bundle["classifier"]
audio_weight = bundle["audio_weight"]

# Load MELD test labels and utterances
test = pd.read_csv(BASE_URL + "test_sent_emo.csv")

# Load precomputed MELD audio embeddings
# Only load pickle files from trusted sources
with open(FEATURE_PATH, "rb") as f:
    audio_splits = pickle.load(f, encoding="latin1")

audio_dict = audio_splits[2]

# Align audio features with test utterances
audio_vectors = []

for row in test.itertuples(index=False):
    dialogue_id = str(row.Dialogue_ID)
    utterance_id = int(row.Utterance_ID)

    if dialogue_id not in audio_dict:
        raise ValueError(f"Missing dialogue {dialogue_id}")

    dialogue_features = np.asarray(audio_dict[dialogue_id])

    if not 0 <= utterance_id < len(dialogue_features):
        raise ValueError(
            f"Missing utterance {utterance_id} "
            f"in dialogue {dialogue_id}"
        )

    audio_vectors.append(dialogue_features[utterance_id])

audio_test = np.asarray(audio_vectors, dtype=np.float32)

assert audio_test.shape == (len(test), 300)

# Transform text using the trained vectorizer
X_text = tfidf.transform(test["Utterance"].fillna(""))

# Normalize audio using the training-fitted scaler
X_audio = scaler.transform(audio_test)

# Combine both modalities
X_test = hstack([
    X_text,
    csr_matrix(X_audio * audio_weight)
]).tocsr()

# Predict test emotions
predictions = model.predict(X_test)
labels = list(model.classes_)

accuracy = accuracy_score(test["Emotion"], predictions)
macro_f1 = f1_score(
    test["Emotion"],
    predictions,
    average="macro",
    zero_division=0
)

print("\n=== MELD TEST RESULTS ===")
print(f"Test examples: {len(test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification report:")
print(classification_report(
    test["Emotion"],
    predictions,
    labels=labels,
    zero_division=0
))

print("\nConfusion matrix:")
print("Label order:", labels)
print(confusion_matrix(
    test["Emotion"],
    predictions,
    labels=labels
))
