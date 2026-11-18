"""UrbanSound8K metadata validation and fold-aware records."""

from pathlib import Path

import pandas as pd

from soundsense.config import CLASSES

REQUIRED_COLUMNS = {"slice_file_name", "classID", "class", "fold"}


class DatasetError(ValueError):
    """Raised when dataset files are absent or malformed."""


def validate_metadata(
    path: str | Path, dataset_root: str | Path
) -> tuple[pd.DataFrame, list[Path]]:
    """Read metadata, validate schema and identify missing audio paths."""
    metadata_path, root = Path(path), Path(dataset_root)
    if not metadata_path.is_file():
        raise DatasetError(
            f"UrbanSound8K metadata not found at {metadata_path}. Download the official archive "
            "from https://urbansounddataset.weebly.com/urbansound8k.html and extract it so that "
            "data/raw/UrbanSound8K/metadata/UrbanSound8K.csv and audio/fold1..fold10 exist."
        )
    frame = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise DatasetError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise DatasetError("Metadata contains no rows.")
    if not pd.api.types.is_integer_dtype(frame["fold"]):
        raise DatasetError("Fold values must be integers from 1 through 10.")
    if not pd.api.types.is_integer_dtype(frame["classID"]):
        raise DatasetError("classID values must be integers from 0 through 9.")
    if not frame["fold"].between(1, 10).all():
        raise DatasetError("Fold values must be integers from 1 through 10.")
    if not frame["classID"].between(0, 9).all():
        raise DatasetError("classID values must be integers from 0 through 9.")
    if not frame["class"].isin(CLASSES).all():
        raise DatasetError("Metadata contains class names outside the ten UrbanSound8K classes.")
    expected_classes = frame["classID"].map(dict(enumerate(CLASSES)))
    if not frame["class"].reset_index(drop=True).equals(expected_classes.reset_index(drop=True)):
        raise DatasetError("Metadata class names do not match their classID values.")
    if frame["slice_file_name"].duplicated().any():
        raise DatasetError("Metadata contains duplicate slice_file_name entries.")
    if frame["slice_file_name"].isna().any():
        raise DatasetError("Metadata contains empty audio file names.")
    if frame["slice_file_name"].astype(str).str.contains(r"[/\\]").any():
        raise DatasetError("slice_file_name must contain filenames, not path components.")
    paths = [
        root / "audio" / f"fold{int(row.fold)}" / row.slice_file_name for row in frame.itertuples()
    ]
    missing = [p for p in paths if not p.is_file()]
    frame = frame.copy()
    frame["audio_path"] = [str(p) for p in paths]
    return frame, missing


def dataset_summary(frame: pd.DataFrame, missing: list[Path]) -> dict:
    """Summarize observed metadata counts; missing files are reported separately."""
    observed_folds = set(frame["fold"].astype(int))
    return {
        "metadata_rows": int(len(frame)),
        "missing_audio_files": len(missing),
        "represented_classes": sorted(frame["class"].unique().tolist()),
        "all_ten_classes_represented": set(frame["class"]) == set(CLASSES),
        "missing_folds": sorted(set(range(1, 11)) - observed_folds),
        "all_ten_folds_represented": observed_folds == set(range(1, 11)),
        "class_distribution": {
            str(k): int(v) for k, v in frame["class"].value_counts().sort_index().items()
        },
        "fold_distribution": {
            str(k): int(v) for k, v in frame["fold"].value_counts().sort_index().items()
        },
    }
