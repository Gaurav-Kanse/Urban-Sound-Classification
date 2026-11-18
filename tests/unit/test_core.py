"""Lightweight tests that do not require UrbanSound8K or trained artifacts."""

import numpy as np
import pandas as pd
import pytest

from soundsense.data.dataset import DatasetError, validate_metadata
from soundsense.features.audio import feature_vector, fixed_length, load_audio, log_mel


def test_metadata_schema_validation(tmp_path):
    path = tmp_path / "UrbanSound8K.csv"
    pd.DataFrame(
        {"slice_file_name": ["x.wav"], "classID": [0], "class": ["air_conditioner"], "fold": [1]}
    ).to_csv(path, index=False)
    frame, missing = validate_metadata(path, tmp_path)
    assert len(frame) == 1
    assert missing == [tmp_path / "audio/fold1/x.wav"]


def test_metadata_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wrong\n1\n")
    with pytest.raises(DatasetError, match="missing required"):
        validate_metadata(path, tmp_path)


def test_audio_preprocessing_shapes_and_finite_values():
    signal = np.sin(np.linspace(0, 50, 22050, dtype=np.float32))
    fixed = fixed_length(signal)
    assert fixed.shape == (88200,)
    mel = log_mel(fixed)
    assert mel.shape[0] == 128 and np.isfinite(mel).all()
    assert feature_vector(fixed).shape == (48,)


def test_audio_rejects_empty():
    with pytest.raises(ValueError):
        fixed_length(np.array([], dtype=np.float32))


def test_audio_loading_reports_corrupt_file(tmp_path):
    audio_path = tmp_path / "corrupt.wav"
    audio_path.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Could not read audio file"):
        load_audio(audio_path)
