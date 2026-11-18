"""CNN tensor-shape tests."""

import pytest
import torch

from soundsense.config import CLASS_NAMES
from soundsense.inference.predictor import Predictor
from soundsense.models.cnn import SoundCNN


def test_cnn_forward_shape():
    model = SoundCNN(num_classes=10)
    assert model(torch.randn(2, 1, 128, 173)).shape == (2, 10)
    assert model(torch.randn(1, 128, 173)).shape == (1, 10)


def test_prediction_pipeline_returns_distribution(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": SoundCNN().state_dict()}, checkpoint)
    predictor = Predictor(checkpoint, device="cpu")
    result = predictor.predict_array(torch.zeros(22050, dtype=torch.float32))
    assert result["predicted_class"] in CLASS_NAMES
    assert len(result["probabilities"]) == 10
    assert len(result["top_three"]) == 3
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
