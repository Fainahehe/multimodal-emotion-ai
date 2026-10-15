
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Load MELD data
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

train = pd.read_csv(BASE_URL + "train_sent_emo.csv")
dev = pd.read_csv(BASE_URL + "dev_sent_emo.csv")

X_train = train["Utterance"].fillna("")
y_train = train["Emotion"]

X_dev = dev["Utterance"].fillna("")
y_dev = dev["Emotion"]

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model
print("Training text emotion classifier...")
model.fit(X_train, y_train)

# Evaluate on validation data
predictions = model.predict(X_dev)

print("\nValidation accuracy:")
print(accuracy_score(y_dev, predictions))

print("\nClassification report:")
print(classification_report(y_dev, predictions, zero_division=0))

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/text_model.joblib")

print("\nModel saved to models/text_model.joblib")

# Test a few custom sentences
examples = [
    "I'm so happy to see you!",
    "Why would you do something like that?",
    "I'm really scared right now.",
    "I don't know what to do anymore."
]

print("\nExample predictions:")
for sentence in examples:
    emotion = model.predict([sentence])[0]
    print(f"{sentence} -> {emotion}")
