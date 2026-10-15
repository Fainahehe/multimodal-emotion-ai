
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

st.set_page_config(
    page_title="Multimodal Emotion AI",
    page_icon="🤖",
    layout="centered"
)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

AUDIO_PATH = (
    "data/MELD.Features.Models/features/audio_emotion.pkl"
)

RESPONSES = {
    "anger": "That sounds frustrating. What happened?",
    "disgust": "That sounds unpleasant. Want to tell me more?",
    "fear": "I'm here with you. What's worrying you?",
    "joy": "That's wonderful! Tell me more!",
    "neutral": "I see. What would you like to talk about?",
    "sadness": "I'm sorry to hear that. Want to talk about it?",
    "surprise": "Oh, that's unexpected! What happened?"
}


@st.cache_resource
def load_model():
    return joblib.load("models/multimodal_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv(BASE_URL + "dev_sent_emo.csv")

    # Only load trusted MELD pickle files.
    with open(AUDIO_PATH, "rb") as f:
        audio_splits = pickle.load(f, encoding="latin1")

    return df, audio_splits[1]


bundle = load_model()
df, audio_dict = load_data()

st.title("🤖 Multimodal Emotion AI")
st.write(
    "An emotion-aware robot prototype combining "
    "utterance text and pretrained MELD audio features."
)

st.info(
    "Offline MELD replay demo: this prototype uses "
    "precomputed audio embeddings, not live microphone audio."
)

# Select a dialogue
dialogue_ids = sorted(df["Dialogue_ID"].unique())

dialogue_id = st.selectbox(
    "Select a MELD dialogue",
    dialogue_ids
)

dialogue = (
    df[df["Dialogue_ID"] == dialogue_id]
    .sort_values("Utterance_ID")
    .reset_index(drop=True)
)

if (
    "active_dialogue" not in st.session_state
    or st.session_state.active_dialogue != dialogue_id
):
    st.session_state.active_dialogue = dialogue_id
    st.session_state.position = 0

st.write(
    f"Utterance {st.session_state.position + 1} "
    f"of {len(dialogue)}"
)

if st.button("Process next utterance"):
    row = dialogue.iloc[st.session_state.position]

    text = str(row["Utterance"])
    utterance_id = int(row["Utterance_ID"])

    audio_vector = np.asarray(
        audio_dict[str(dialogue_id)][utterance_id],
        dtype=np.float32
    ).reshape(1, -1)

    start = time.perf_counter()

    text_features = bundle["tfidf"].transform([text])

    audio_features = bundle["scaler"].transform(
        audio_vector
    )

    audio_features *= bundle["audio_weight"]

    combined = hstack([
        text_features,
        csr_matrix(audio_features)
    ]).tocsr()

    emotion = bundle["classifier"].predict(combined)[0]

    response = RESPONSES[emotion]

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    st.subheader("Input")
    st.write(text)

    st.subheader("Predicted emotion")
    st.success(emotion.capitalize())

    st.subheader("Robot response")
    st.write(response)

    st.json({
        "emotion": emotion,
        "response": response,
        "inference_latency_ms": round(latency_ms, 2)
    })

    st.caption(
        f"Ground-truth MELD label: {row['Emotion']}"
    )

    st.session_state.position = (
        st.session_state.position + 1
    ) % len(dialogue)
