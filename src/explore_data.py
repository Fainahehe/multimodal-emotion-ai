
import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

train = pd.read_csv(BASE_URL + "train_sent_emo.csv")
dev = pd.read_csv(BASE_URL + "dev_sent_emo.csv")
test = pd.read_csv(BASE_URL + "test_sent_emo.csv")

print("Training examples:", len(train))
print("Validation examples:", len(dev))
print("Test examples:", len(test))

print("\nFirst five examples:")
print(train[["Utterance", "Emotion"]].head())

print("\nEmotion distribution:")
print(train["Emotion"].value_counts())

