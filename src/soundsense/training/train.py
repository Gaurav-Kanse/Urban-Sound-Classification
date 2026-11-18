"""Fold-aware baseline and CNN training/evaluation routines."""

import json
import logging
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from soundsense.config import AUDIO, CLASS_NAMES, PATHS, audio_config_dict
from soundsense.data.dataset import validate_metadata
from soundsense.features.audio import feature_vector, load_audio, log_mel
from soundsense.models.cnn import SoundCNN

LOG = logging.getLogger(__name__)


def _save_evaluation_artifacts(name: str, result: dict, history: list[dict] | None = None) -> None:
    """Save per-class CSV and readable evaluation plots from computed metrics."""
    PATHS["metrics"].mkdir(parents=True, exist_ok=True)
    PATHS["figures"].mkdir(parents=True, exist_ok=True)
    test_metrics = result["test"]
    class_rows = []
    for label in CLASS_NAMES:
        values = test_metrics["classification_report"][label]
        class_rows.append(
            {
                "class": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"],
            }
        )
    pd.DataFrame(class_rows).to_csv(PATHS["metrics"] / f"{name}_per_class.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        np.asarray(test_metrics["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set(title=f"{name} test confusion matrix", xlabel="Predicted class", ylabel="True class")
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(PATHS["figures"] / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=pd.DataFrame(class_rows), x="class", y="f1", color="#5271ff", ax=ax)
    ax.set(title=f"{name} test F1 by class", xlabel="Class", ylabel="F1 score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(PATHS["figures"] / f"{name}_per_class.png", dpi=160)
    plt.close(fig)
    if history:
        history_df = pd.DataFrame(history)
        for y_cols, title, filename, ylabel in [
            (
                ["train_loss", "val_loss"],
                "Training and validation loss",
                f"{name}_loss.png",
                "Cross-entropy",
            ),
            (
                ["train_accuracy", "val_accuracy"],
                "Training and validation accuracy",
                f"{name}_accuracy.png",
                "Accuracy",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            for column in y_cols:
                ax.plot(history_df["epoch"], history_df[column], label=column.replace("_", " "))
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
            ax.legend()
            fig.tight_layout()
            fig.savefig(PATHS["figures"] / filename, dpi=160)
            plt.close(fig)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use only official folds: 1-8 train, 9 validation, 10 test."""
    return frame[frame.fold <= 8], frame[frame.fold == 9], frame[frame.fold == 10]


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Build label-complete metrics so absent classes remain visible."""
    labels = np.arange(len(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _feature_table(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in frame.itertuples():
        X.append(feature_vector(load_audio(row.audio_path)))
        y.append(int(row.classID))
    return np.asarray(X), np.asarray(y)


def train_baseline() -> dict:
    """Train Random Forest on training folds and choose best validation macro-F1."""
    frame, missing = validate_metadata(PATHS["metadata"], PATHS["dataset"])
    if missing:
        raise FileNotFoundError(
            f"Dataset has {len(missing)} missing audio files; see prepare_data.py."
        )
    train, val, test = split_records(frame)
    X_train, y_train = _feature_table(train)
    X_val, y_val = _feature_table(val)
    X_test, y_test = _feature_table(test)
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    PATHS["metrics"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATHS["models"] / "baseline.joblib")
    PATHS["configs"].mkdir(parents=True, exist_ok=True)
    (PATHS["configs"] / "baseline.json").write_text(
        json.dumps(
            {
                "model": "RandomForest",
                "audio_config": audio_config_dict(),
                "features": "MFCC mean/std and spectral summary mean/std",
                "training_folds": list(range(1, 9)),
                "validation_fold": 9,
                "test_fold": 10,
                "random_state": 42,
            },
            indent=2,
        )
    )
    results = {
        "model": "RandomForest",
        "validation": metrics_for(y_val, val_pred),
        "test": metrics_for(y_test, test_pred),
    }
    (PATHS["metrics"] / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    _save_evaluation_artifacts("baseline", results)
    return results


class MelDataset(Dataset):
    """On-demand log-Mel data with augmentation restricted to training samples."""

    def __init__(self, frame: pd.DataFrame, augment: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        audio = load_audio(row.audio_path)
        if self.augment:
            shift = np.random.randint(
                -int(0.15 * AUDIO.sample_rate), int(0.15 * AUDIO.sample_rate) + 1
            )
            audio = np.roll(audio, shift)
            audio = audio * np.random.uniform(0.8, 1.2) + np.random.normal(
                0, 0.003, audio.shape
            ).astype(np.float32)
            audio = np.clip(audio, -1, 1)
        return torch.from_numpy(log_mel(audio)).unsqueeze(0), torch.tensor(
            int(row.classID), dtype=torch.long
        )


def train_cnn(epochs: int = 25, batch_size: int = 32, patience: int = 6, seed: int = 42) -> dict:
    """Train with validation macro F1 selection and evaluate the held-out fold once."""
    seed_everything(seed)
    frame, missing = validate_metadata(PATHS["metadata"], PATHS["dataset"])
    if missing:
        raise FileNotFoundError(
            f"Dataset has {len(missing)} missing audio files; see prepare_data.py."
        )
    train, val, test = split_records(frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        MelDataset(train, True), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(MelDataset(val), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(MelDataset(test), batch_size=batch_size, shuffle=False, num_workers=0)
    model = SoundCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    PATHS["metrics"].mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        val_y = []
        val_p = []
        val_loss = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss.append(float(criterion(logits, y).item()))
                val_y.extend(y.cpu().tolist())
                val_p.extend(logits.argmax(1).cpu().tolist())
        val_f1 = float(f1_score(val_y, val_p, average="macro", zero_division=0))
        scheduler.step(val_f1)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / max(total, 1),
            "val_loss": float(np.mean(val_loss)),
            "val_accuracy": float(accuracy_score(val_y, val_p)),
            "val_macro_f1": val_f1,
        }
        history.append(row)
        LOG.info("epoch %d/%d: %s", epoch + 1, epochs, row)
        if val_f1 > best:
            best, stale = val_f1, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "audio_config": audio_config_dict(),
                    "classes": CLASS_NAMES,
                    "epoch": epoch + 1,
                    "val_macro_f1": best,
                    "training_config": {
                        "optimizer": "AdamW",
                        "learning_rate": 1e-3,
                        "batch_size": batch_size,
                        "seed": seed,
                        "training_samples": len(train),
                    },
                },
                PATHS["models"] / "cnn_best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break
    # Best checkpoint is reloaded before touching fold 10.
    payload = torch.load(PATHS["models"] / "cnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    test_y = []
    test_p = []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            test_y.extend(y.tolist())
            test_p.extend(logits.argmax(1).cpu().tolist())
    result = {
        "model": "SoundCNN",
        "device": str(device),
        "best_epoch": payload["epoch"],
        "validation_macro_f1": best,
        "training": {
            "epochs_completed": len(history),
            "training_samples_per_epoch": len(train),
            "validation_samples": len(val),
            "test_samples": len(test),
            "seed": seed,
            "batch_size": batch_size,
            "patience": patience,
        },
        "test": metrics_for(np.array(test_y), np.array(test_p)),
        "history": history,
    }
    (PATHS["metrics"] / "cnn_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(history).to_csv(PATHS["metrics"] / "training_history.csv", index=False)
    _save_evaluation_artifacts("cnn", result, history)
    baseline_path = PATHS["metrics"] / "baseline_metrics.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        comparison = pd.DataFrame(
            [
                {
                    "model": "Random Forest",
                    "validation_macro_f1": baseline["validation"]["macro_f1"],
                    "test_macro_f1": baseline["test"]["macro_f1"],
                },
                {
                    "model": "CNN",
                    "validation_macro_f1": best,
                    "test_macro_f1": result["test"]["macro_f1"],
                },
            ]
        )
        comparison.to_csv(PATHS["metrics"] / "model_comparison.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        comparison.set_index("model").plot(
            kind="bar", ax=ax, ylim=(0, 1), color=["#6887ff", "#31b99b"]
        )
        ax.set(title="Baseline and CNN performance", ylabel="Macro F1", xlabel="Model")
        ax.legend(title="Split")
        fig.tight_layout()
        fig.savefig(PATHS["figures"] / "model_comparison.png", dpi=160)
        plt.close(fig)
    return result
