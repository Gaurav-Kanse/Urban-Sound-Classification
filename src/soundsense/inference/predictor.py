"""Reusable checkpoint-backed audio prediction."""

from pathlib import Path

import numpy as np
import torch

from soundsense.config import AUDIO, CLASS_NAMES, AudioConfig
from soundsense.features.audio import load_audio, log_mel
from soundsense.models.cnn import SoundCNN


class Predictor:
    """Load a CNN once and predict audio files with the training preprocessing."""

    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = SoundCNN(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.audio_config = AudioConfig(**payload.get("audio_config", AUDIO.__dict__))

    @torch.inference_mode()
    def predict_array(self, audio: np.ndarray) -> dict:
        """Predict an already loaded waveform and return named scores."""
        features = (
            torch.from_numpy(log_mel(audio, self.audio_config))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        logits = self.model(features)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        order = np.argsort(probabilities)[::-1]
        return {
            "predicted_class": CLASS_NAMES[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
            "top_three": [
                {"class": CLASS_NAMES[int(i)], "probability": float(probabilities[i])}
                for i in order[:3]
            ],
        }

    def predict_file(self, path: str | Path) -> dict:
        """Load and predict an audio path."""
        return self.predict_array(load_audio(path, self.audio_config))
