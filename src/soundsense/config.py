"""Central configuration and class mapping."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
CLASS_NAMES = [name.replace("_", " ").title() for name in CLASSES]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.duration)


AUDIO = AudioConfig()
PATHS = {
    "dataset": ROOT / "data/raw/UrbanSound8K",
    "metadata": ROOT / "data/raw/UrbanSound8K/metadata/UrbanSound8K.csv",
    "processed": ROOT / "data/processed",
    "models": ROOT / "artifacts/models",
    "metrics": ROOT / "artifacts/metrics",
    "configs": ROOT / "artifacts/configs",
    "figures": ROOT / "reports/figures",
    "results": ROOT / "reports/results",
}


def audio_config_dict() -> dict[str, int | float]:
    """Return serializable preprocessing configuration."""
    return asdict(AUDIO)
