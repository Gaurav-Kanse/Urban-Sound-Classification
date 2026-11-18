"""Generate dataset EDA figures from the official audio collection."""

import json
import logging

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from soundsense.config import AUDIO, PATHS
from soundsense.data.dataset import DatasetError, validate_metadata
from soundsense.features.audio import feature_vector, load_audio, log_mel


def main() -> int:
    try:
        frame, missing = validate_metadata(PATHS["metadata"], PATHS["dataset"])
    except DatasetError as exc:
        logging.error("%s", exc)
        return 2
    if missing:
        logging.error("Cannot run complete EDA: %d audio paths are missing.", len(missing))
        return 2
    out = PATHS["figures"]
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=frame, x="class", order=sorted(frame["class"].unique()), ax=ax)
    ax.set(title="UrbanSound8K samples by class", xlabel="Class", ylabel="Recordings")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=frame, x="fold", ax=ax)
    ax.set(title="Audio samples by predefined fold", xlabel="Fold", ylabel="Recordings")
    fig.tight_layout()
    fig.savefig(out / "fold_distribution.png", dpi=160)
    plt.close(fig)
    durations, feature_rows, corrupt = [], [], []
    example = None
    for _, row in frame.iterrows():
        try:
            audio = load_audio(row["audio_path"])
            if example is None:
                example = audio
            duration = float(librosa.get_duration(path=row["audio_path"]))
            durations.append({"duration": duration, "class": row["class"]})
            values = feature_vector(audio)
            feature_rows.append(
                {"class": row["class"], **{f"feature_{i}": v for i, v in enumerate(values)}}
            )
        except (ValueError, OSError) as exc:
            logging.warning("Skipping unreadable %s: %s", row["audio_path"], exc)
            corrupt.append({"path": row["audio_path"], "error": str(exc)})
    if example is None:
        logging.error("No readable audio recordings were available for EDA.")
        return 2
    duration_df = pd.DataFrame(durations)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=duration_df, x="duration", bins=35, ax=ax)
    ax.set(title="Audio duration distribution", xlabel="Duration (seconds)", ylabel="Recordings")
    fig.tight_layout()
    fig.savefig(out / "duration_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3))
    times = np.arange(len(example)) / AUDIO.sample_rate
    ax.plot(times, example, linewidth=0.5)
    ax.set(title="Example waveform", xlabel="Time (s)", ylabel="Amplitude")
    fig.tight_layout()
    fig.savefig(out / "sample_waveform.png", dpi=160)
    plt.close(fig)
    for name, matrix, title, ylabel in [
        ("sample_mel.png", log_mel(example), "Example log-Mel spectrogram", "Mel band"),
        (
            "sample_mfcc.png",
            librosa.feature.mfcc(y=example, sr=AUDIO.sample_rate, n_mfcc=20),
            "Example MFCCs",
            "MFCC",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
        ax.set(title=title, xlabel="Frame", ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
    if feature_rows:
        feature_df = pd.DataFrame(feature_rows)
        numeric = feature_df.drop(columns="class")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set(title="Extracted feature correlation")
        fig.tight_layout()
        fig.savefig(out / "feature_correlation.png", dpi=160)
        plt.close(fig)
        # Compare class-wise aggregate summary feature means using class x feature matrix.
        class_means = feature_df.groupby("class").mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(class_means, cmap="viridis", ax=ax)
        ax.set(title="Class-wise audio feature comparison", xlabel="Feature index", ylabel="Class")
        fig.tight_layout()
        fig.savefig(out / "class_feature_comparison.png", dpi=160)
        plt.close(fig)
    PATHS["results"].mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": len(frame),
        "missing": 0,
        "corrupt_audio_files": corrupt,
        "duration_count": len(durations),
        "duration_mean_seconds": float(duration_df.duration.mean()) if durations else None,
        "class_distribution": {
            str(k): int(v) for k, v in frame["class"].value_counts().sort_index().items()
        },
        "fold_distribution": {
            str(k): int(v) for k, v in frame["fold"].value_counts().sort_index().items()
        },
    }
    (PATHS["results"] / "eda_statistics.json").write_text(json.dumps(stats, indent=2))
    observations = [
        f"Metadata contains {len(frame)} rows and {len(durations)} readable duration measurements."
    ]
    if durations:
        observations.append(
            f"Mean excerpt duration is {duration_df.duration.mean():.2f} seconds; "
            f"median is {duration_df.duration.median():.2f} seconds."
        )
    counts = frame["class"].value_counts()
    observations.append(
        f"Class counts range from {int(counts.min())} to {int(counts.max())} excerpts."
    )
    report = (
        "# Dataset analysis\n\n"
        + "\n\n".join(observations)
        + "\n\nAll statements above are computed from the validated metadata and "
        "readable audio files during this run.\n"
    )
    (PATHS["results"] / "dataset_analysis.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
