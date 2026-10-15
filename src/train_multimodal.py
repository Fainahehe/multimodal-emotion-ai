
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

FEATURE_PATH = Path(
    "data/MELD.Features.Models/features/audio_emotion.pkl"
)

# Load official MELD text and labels
train = pd.read_csv(BASE_URL + "train_sent_emo.csv")
dev = pd.read_csv(BASE_URL + "dev_sent_emo.csv")
test = pd.read_csv(BASE_URL + "test_sent_emo.csv")

# Load trusted MELD audio feature archive
with open(FEATURE_PATH, "rb") as f:
    audio_splits = pickle.load(f, encoding="latin1")


def align_audio(df, audio_dict):
    """Match each utterance to its dialogue audio vector."""
    vectors = []

    for row in df.itertuples(index=False):
        dialogue_id = str(row.Dialogue_ID)
        utterance_id = int(row.Utterance_ID)

        if dialogue_id not in audio_dict:
            raise ValueError(
                f"Missing dialogue {dialogue_id}"
            )

        dialogue_features = np.asarray(
            audio_dict[dialogue_id]
        )

        if utterance_id >= len(dialogue_features):
            raise ValueError(
                f"Missing utterance {utterance_id} "
                f"in dialogue {dialogue_id}"
            )

        vectors.append(dialogue_features[utterance_id])

    return np.asarray(vectors, dtype=np.float32)


# Expected order: train, validation, test.
# Validate all dialogue and utterance references.
audio_train = align_audio(train, audio_splits[0])
audio_dev = align_audio(dev, audio_splits[1])
audio_test = align_audio(test, audio_splits[2])

print("Audio train:", audio_train.shape)
print("Audio validation:", audio_dev.shape)
print("Audio test:", audio_test.shape)

assert audio_train.shape == (len(train), 300)
assert audio_dev.shape == (len(dev), 300)
assert audio_test.shape == (len(test), 300)

# Text feature extraction
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_text_train = tfidf.fit_transform(
    train["Utterance"].fillna("")
)

X_text_dev = tfidf.transform(
    dev["Utterance"].fillna("")
)

# Normalize audio features using training data only
scaler = StandardScaler()

X_audio_train = scaler.fit_transform(audio_train)
X_audio_dev = scaler.transform(audio_dev)


# Control the relative influence of audio
AUDIO_WEIGHT = 0.1

X_train = hstack([
    X_text_train,
    csr_matrix(X_audio_train * AUDIO_WEIGHT)
]).tocsr()

X_dev = hstack([
    X_text_dev,
    csr_matrix(X_audio_dev * AUDIO_WEIGHT)
]).tocsr()


# Train classifier
model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    random_state=42
)

print("\nTraining multimodal classifier...")
model.fit(X_train, train["Emotion"])

# Evaluate
predictions = model.predict(X_dev)

print("\nMultimodal validation accuracy:")
print(accuracy_score(dev["Emotion"], predictions))

print("\nClassification report:")
print(classification_report(
    dev["Emotion"],
    predictions,
    zero_division=0
))

# Save all components needed for inference
Path("models").mkdir(exist_ok=True)


joblib.dump({
    "tfidf": tfidf,
    "scaler": scaler,
    "classifier": model,
    "audio_weight": AUDIO_WEIGHT
}, "models/multimodal_model.joblib")


print("\nMultimodal model saved!")
