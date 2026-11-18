"""Shared audio loading, fixed-duration normalization, and feature extraction."""

from pathlib import Path

import librosa
import numpy as np

from soundsense.config import AUDIO, AudioConfig


def load_audio(path: str | Path, config: AudioConfig = AUDIO) -> np.ndarray:
    """Load an audio file as mono float32 at the configured sample rate."""
    try:
        audio, _ = librosa.load(path, sr=config.sample_rate, mono=True, duration=None)
    except Exception as exc:
        raise ValueError(f"Could not read audio file '{path}': {exc}") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio file '{path}' is empty or contains invalid samples.")
    return fixed_length(audio, config)


def fixed_length(audio: np.ndarray, config: AudioConfig = AUDIO) -> np.ndarray:
    """Normalize peak amplitude, then center crop or zero-pad to a fixed duration."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Audio must contain finite samples.")
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    target = config.samples
    if signal.size > target:
        start = (signal.size - target) // 2
        signal = signal[start : start + target]
    elif signal.size < target:
        pad = target - signal.size
        signal = np.pad(signal, (pad // 2, pad - pad // 2))
    return signal.astype(np.float32, copy=False)


def log_mel(audio: np.ndarray, config: AudioConfig = AUDIO) -> np.ndarray:
    """Return normalized log-Mel array (n_mels, frames) used by CNN and inference."""
    signal = fixed_length(audio, config)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    return np.clip((db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def feature_vector(audio: np.ndarray, config: AudioConfig = AUDIO) -> np.ndarray:
    """Compute compact summary statistics for baseline classifiers."""
    signal = fixed_length(audio, config)
    mfcc = librosa.feature.mfcc(y=signal, sr=config.sample_rate, n_mfcc=config.n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=config.sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=config.sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=config.sample_rate)
    stats = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    stats += [np.array([x.mean(), x.std()]) for x in (zcr, centroid, bandwidth, rolloff)]
    return np.concatenate(stats).astype(np.float32)
