"""Streamlit dashboard for SoundSense."""

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from soundsense.config import CLASS_NAMES, PATHS
from soundsense.features.audio import load_audio, log_mel
from soundsense.inference.predictor import Predictor

st.set_page_config(page_title="SoundSense", page_icon="🔊", layout="wide")
st.markdown(
    """<style>
.stApp {background: #0b1020; color: #edf2ff}
.block-container {padding-top: 2rem}
[data-testid="stMetric"] {
    background: #141d33;
    border: 1px solid #273653;
    padding: 1rem;
    border-radius: 14px;
}
</style>""",
    unsafe_allow_html=True,
)
st.title("🔊 SoundSense")
st.caption("Urban environmental sound classification · UrbanSound8K · 10 categories")
pages = ["Overview", "Audio Classification", "Model Performance", "About"]
page = st.sidebar.radio("Navigate", pages)

if page == "Overview":
    st.header("Understand the sounds around you")
    st.write(
        "Upload an environmental recording to classify it into one of ten UrbanSound8K categories."
    )
    a, b, c = st.columns(3)
    a.metric("Sound categories", "10")
    b.metric("Audio input", "WAV / MP3 / FLAC")
    c.metric("Primary model", "Log-Mel CNN")
    st.subheader("Supported categories")
    st.write(" · ".join(CLASS_NAMES))
    summary_path = PATHS["results"] / "dataset_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.subheader("Dataset summary")
        st.write(
            f"{summary['metadata_rows']:,} metadata rows · "
            f"{summary['missing_audio_files']} missing files · "
            f"{len(summary.get('corrupt_audio_files', []))} unreadable files"
        )
        st.bar_chart(summary.get("class_distribution", {}))
    else:
        st.info("Dataset summary unavailable. Run `python scripts/prepare_data.py` after setup.")
    st.write(
        "Model status: "
        + (
            "CNN checkpoint available"
            if (PATHS["models"] / "cnn_best.pt").is_file()
            else "CNN checkpoint not trained yet"
        )
    )
elif page == "Audio Classification":
    st.header("Classify a recording")
    upload = st.file_uploader("Choose audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if upload:
        st.audio(upload)
        raw = upload.getvalue()
        temp_handle = tempfile.NamedTemporaryFile(
            suffix=Path(upload.name).suffix or ".audio", delete=False
        )
        temp = Path(temp_handle.name)
        try:
            temp_handle.write(raw)
        finally:
            temp_handle.close()
        try:
            audio = load_audio(temp)
            mel = log_mel(audio)
            left, right = st.columns(2)
            with left:
                fig, ax = plt.subplots()
                ax.plot(np.arange(len(audio)) / 22050, audio, lw=0.55)
                ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
                st.pyplot(fig)
                plt.close(fig)
            with right:
                fig, ax = plt.subplots()
                ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
                ax.set(xlabel="Frame", ylabel="Mel band", title="Log-Mel spectrogram")
                st.pyplot(fig)
                plt.close(fig)
            checkpoint = PATHS["models"] / "cnn_best.pt"
            if not checkpoint.is_file():
                st.warning(
                    "CNN model is not available yet. Run `python scripts/train_cnn.py` "
                    "after preparing the official dataset."
                )
            else:
                result = Predictor(checkpoint).predict_array(audio)
                st.success(
                    f"Prediction: {result['predicted_class']} · "
                    f"confidence {result['confidence']:.1%}"
                )
                st.write("Top three")
                st.dataframe(result["top_three"], use_container_width=True, hide_index=True)
                st.bar_chart(result["probabilities"])
                st.caption(
                    "Scores are softmax model outputs and have not been calibrated; "
                    "confidence is not a guarantee of correctness."
                )
        except Exception as exc:
            st.error(f"Could not analyze this audio: {exc}")
        finally:
            temp.unlink(missing_ok=True)
elif page == "Model Performance":
    st.header("Held-out evaluation")
    metric_files = list(PATHS["metrics"].glob("*_metrics.json"))
    if not metric_files:
        st.info("No evaluation metrics yet. Train the baseline and CNN to populate this page.")
    else:
        for path in metric_files:
            result = json.loads(path.read_text())
            st.subheader(result["model"])
            metrics = result.get("test", {})
            cols = st.columns(4)
            for col, key, label in zip(
                cols,
                ["accuracy", "macro_f1", "weighted_f1", "macro_recall"],
                ["Accuracy", "Macro F1", "Weighted F1", "Macro recall"],
                strict=True,
            ):
                col.metric(label, f"{metrics.get(key,0):.3f}")
            if "confusion_matrix" in metrics:
                matrix = pd.DataFrame(
                    metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES
                )
                st.dataframe(matrix, use_container_width=True)
            report = metrics.get("classification_report", {})
            class_rows = [
                {
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "F1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
                for label in CLASS_NAMES
                if label in report
            ]
            if class_rows:
                st.subheader("Per-class performance")
                class_metrics = pd.DataFrame(class_rows).set_index("class")
                st.dataframe(class_metrics, use_container_width=True)
                st.bar_chart(class_metrics[["precision", "recall", "F1"]])
            if result.get("history"):
                history = pd.DataFrame(result["history"]).set_index("epoch")
                st.line_chart(history[["train_loss", "val_loss"]])
                st.line_chart(history[["train_accuracy", "val_accuracy"]])
elif page == "About":
    st.header("About SoundSense")
    st.write(
        "UrbanSound8K contains 8,732 labeled excerpts, arranged in ten predefined "
        "folds. SoundSense trains on folds 1–8, selects checkpoints using fold 9, "
        "and reserves fold 10 for final evaluation."
    )
    st.write(
        "The CNN uses log-scaled Mel spectrograms. Mel bands group frequency in a "
        "way that roughly follows human pitch perception, making time-frequency "
        "patterns visible to a compact image-like model."
    )
    st.write(
        "Limitations: recordings can contain multiple sources, classes are not "
        "perfectly balanced, and softmax scores are not calibrated probabilities. "
        "Predictions may not generalize to sounds outside the dataset."
    )
