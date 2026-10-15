
import pickle
from pathlib import Path

feature_path = Path(
    "data/MELD.Features.Models/features/audio_emotion.pkl"
)

# Only load pickle files from trusted sources.
with open(feature_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Data type:", type(data))

if isinstance(data, dict):
    print("\nDictionary keys:")
    for key, value in data.items():
        print(key, type(value), getattr(value, "shape", ""))

elif isinstance(data, (list, tuple)):
    print("\nNumber of elements:", len(data))

    for i, item in enumerate(data):
        print(
            f"Element {i}:",
            type(item),
            getattr(item, "shape", ""),
        )

        if isinstance(item, dict):
            print("Sample keys:", list(item.keys())[:5])

else:
    print("Shape:", getattr(data, "shape", "Unknown"))


import numpy as np

print("\n--- Detailed feature inspection ---")

for i, split in enumerate(data):
    print(f"\nDictionary {i}:")
    print("Number of dialogue IDs:", len(split))

    first_key = next(iter(split))
    features = np.asarray(split[first_key])

    print("Example dialogue ID:", first_key)
    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)

    print("First 5 dialogue IDs:", list(split.keys())[:5])

