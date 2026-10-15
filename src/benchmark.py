
import os
import time

import joblib
import psutil

MODEL_PATH = "models/multimodal_model.joblib"

process = psutil.Process(os.getpid())

# Measure memory before loading
memory_before = process.memory_info().rss / (1024 ** 2)

start = time.perf_counter()
bundle = joblib.load(MODEL_PATH)
load_time = time.perf_counter() - start

memory_after = process.memory_info().rss / (1024 ** 2)

print("=== MODEL RESOURCE BENCHMARK ===")
print(f"Model load time: {load_time:.3f} seconds")
print(f"Process RAM before load: {memory_before:.2f} MB")
print(f"Process RAM after load: {memory_after:.2f} MB")
print(f"RAM increase: {memory_after - memory_before:.2f} MB")

print(
    f"Model file size: "
    f"{os.path.getsize(MODEL_PATH) / (1024 ** 2):.2f} MB"
)

print(
    "Classifier learned coefficients:",
    bundle["classifier"].coef_.size
)

print(
    "Classifier learned intercepts:",
    bundle["classifier"].intercept_.size
)


import numpy as np
from scipy.sparse import csr_matrix, hstack

# Reuse the loaded model components
tfidf = bundle["tfidf"]
scaler = bundle["scaler"]
classifier = bundle["classifier"]
audio_weight = bundle["audio_weight"]

# Use a synthetic example to benchmark local inference.
# This measures text preprocessing, audio scaling,
# feature fusion, and classification.
# It does not include raw audio feature extraction.

text = "I am really happy to see you!"
audio = np.zeros((1, 300), dtype=np.float32)

def predict_once():
    text_features = tfidf.transform([text])
    audio_features = scaler.transform(audio)

    combined = hstack([
        text_features,
        csr_matrix(audio_features * audio_weight)
    ]).tocsr()

    return classifier.predict(combined)[0]


# Warm up the pipeline before measuring
for _ in range(20):
    predict_once()

latencies = []

for _ in range(100):
    start = time.perf_counter()
    predict_once()
    latencies.append(
        (time.perf_counter() - start) * 1000
    )

print("\n=== INFERENCE LATENCY BENCHMARK ===")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Median: {np.median(latencies):.2f} ms")
print(f"P95: {np.percentile(latencies, 95):.2f} ms")
print(f"Maximum: {np.max(latencies):.2f} ms")
