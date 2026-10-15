# Multimodal Emotion AI
### Low-Latency Emotion Recognition for Human–Robot Interaction

A local multimodal machine learning prototype that combines conversational text with pretrained audio embeddings to predict emotions and generate short, emotion-aware robot responses.

Developed as a technical exploration of emotion-aware human–computer interaction using the **Multimodal EmotionLines Dataset (MELD)**.

## Overview

Emotion recognition from text alone is challenging because the same words can convey different emotions depending on vocal tone and delivery.

This project explores multimodal emotion classification using two complementary inputs:

- **Text:** The spoken utterance, represented using TF-IDF features.
- **Audio:** Precomputed 300-dimensional audio embeddings supplied with MELD.

The system predicts one of seven emotion categories and produces a short, rule-based conversational response.

A Streamlit interface demonstrates sequential processing of MELD dialogue utterances, structured JSON output, and measured local inference latency.

## Architecture

```text
MELD Utterance Text
        |
        v
 TF-IDF Vectorizer
        |
        +----------------------+
                               |
MELD Audio Embedding           |
        |                      |
        v                      |
 StandardScaler                |
        |                      |
        v                      |
 Audio Weight (0.1)            |
        |                      |
        +----------------------+
                    |
                    v
          Feature Concatenation
                    |
                    v
       Logistic Regression Model
                    |
                    v
         Predicted MELD Emotion
                    |
                    v
        Rule-Based Response Policy
                    |
                    v
         Streamlit Interface
           + JSON Output
```

### Model Design

- Text representation: TF-IDF, unigrams and bigrams, maximum 10,000 features.
- Audio representation: pretrained 300-dimensional MELD audio features.
- Audio preprocessing: StandardScaler fitted on training data.
- Fusion strategy: weighted early fusion, with audio weight 0.1.
- Classifier: multinomial logistic regression with balanced class weights.
- Emotion categories: anger, disgust, fear, joy, neutral, sadness, surprise.

The model is trained and evaluated locally using scikit-learn.

## Dataset

This project uses the [MELD dataset](https://github.com/declare-lab/MELD), containing emotion-annotated conversations from the television series *Friends*.

| Split | Utterances |
|---|---:|
| Training | 9,989 |
| Validation | 1,109 |
| Test | 2,610 |

The official dataset splits are preserved.

Large MELD archives and extracted features are excluded from this repository.

## Evaluation

The classifier was trained on the MELD training split, with model configuration evaluated using the validation split. Final evaluation was performed on the held-out test split.

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Accuracy | 41.23% |
| Macro F1 | 0.3189 |
| Weighted F1 | Approximately 0.43 |
| Test samples | 2,610 |

### Per-Class F1 Scores

| Emotion | F1 |
|---|---:|
| Anger | 0.37 |
| Disgust | 0.13 |
| Fear | 0.07 |
| Joy | 0.39 |
| Neutral | 0.51 |
| Sadness | 0.30 |
| Surprise | 0.46 |

The classifier performs better on neutral and surprise than on fear and disgust.

Class imbalance and overlapping emotional signals remain significant challenges.

A majority-class baseline predicting neutral would achieve approximately 48% accuracy on the test split, exceeding the current classifier's accuracy. This highlights the need for improved calibration, feature representations, and class-imbalance handling.

## Latency and Resource Benchmark

Benchmarks were collected on a Windows laptop using local CPU inference.

| Measurement | Observed result |
|---|---:|
| Mean inference latency | 0.72 ms |
| Median latency | 0.67 ms |
| P95 latency | 1.00 ms |
| Maximum latency | 1.96 ms |
| Saved model size | 0.92 MB |
| Additional process RAM during model loading | 118.80 MB |
| Model load time (measured run) | 1.390 seconds |
| Logistic regression learned parameters | 72,107 |

Latency was measured over 100 warmed-up predictions using synthetic text and a zero-valued 300-dimensional audio feature vector.

These measurements cover text vectorization, audio normalization, feature fusion, and classification. They exclude raw audio capture, audio feature extraction, dataset loading, and Streamlit rendering.

The reported RAM increase reflects the Python process during model loading, including library allocations, rather than model memory alone.

The local classifier is substantially smaller than the assignment's six-billion-parameter ceiling.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Fainahehe/multimodal-emotion-ai.git
cd multimodal-emotion-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Obtain MELD data

Download the MELD feature archive from the [MELD project](https://github.com/declare-lab/MELD).

Extract the feature files so the audio embeddings are available at:

```text
data/MELD.Features.Models/features/audio_emotion.pkl
```

The training and evaluation scripts also require access to the official MELD CSV files from the public repository.

**Note:** The interactive replay application also requires the locally available MELD features. The large dataset files are intentionally not distributed in this Git repository.

### 5. Train the multimodal model (optional)

```bash
python src/train_multimodal.py
```

### 6. Evaluate

```bash
python src/evaluate.py
```

### 7. Launch the interface

```bash
python -m streamlit run app.py
```

The application opens in a local browser, typically at `http://localhost:8501`.

## Interactive Demonstration

The Streamlit application allows a user to select a MELD dialogue and process its utterances sequentially.

For each utterance, it displays:

- Input text.
- Predicted emotion.
- Short robot response.
- Structured JSON state.
- Local inference latency.
- Ground-truth MELD emotion label.

Example output:

```json
{
  "emotion": "joy",
  "response": "That's wonderful! Tell me more!",
  "inference_latency_ms": 13.44
}
```

This example illustrates the output format, not a guarantee of prediction correctness or typical latency.

## Design Decisions and Trade-offs

**Lightweight classical ML:** Logistic regression was selected for fast local inference, interpretability, and low hardware requirements.

**Early multimodal fusion:** Text and audio features are concatenated to provide a simple, reproducible multimodal baseline.

**Precomputed audio embeddings:** These enable multimodal experiments without expensive audio encoder training, but prevent arbitrary live microphone input from being processed by the current model.

**Rule-based response generation:** A small deterministic response policy keeps the interaction lightweight and avoids requiring a large language model. Responses are limited in conversational depth.

**Sequential interaction:** Dialogue utterances are processed in order through the interface. This is an offline replay demonstration rather than continuous live audio streaming.

## Limitations and Future Work

The current prototype has several limitations:

1. It uses precomputed MELD audio embeddings rather than live microphone input.
2. Its latency measurements exclude raw audio processing.
3. Its emotion recognition accuracy is limited, especially for minority classes.
4. Robot responses use a simple rule-based policy rather than contextual language generation.
5. The model does not explicitly incorporate previous dialogue turns as predictive features.
6. The application is a research demonstration rather than a production-ready human–robot interaction system.

Potential extensions include live audio feature extraction, stronger text and audio encoders, contextual emotion recognition, confidence calibration, and a more capable response policy.

## Tools and Technologies

Python, pandas, NumPy, SciPy, scikit-learn, joblib, Streamlit, and MELD.

## Dataset Attribution

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., and Mihalcea, R. (2018). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.*

Dataset repository: https://github.com/declare-lab/MELD

## Development Notes

AI-assisted development tools were used during implementation. Model design decisions, experiments, evaluation, integration, and reported results are documented for reproducibility.