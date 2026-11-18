"""Verify the expected UrbanSound8K metadata and audio files."""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_ROWS = 8732
REQUIRED_COLUMNS = {"slice_file_name", "fold", "class"}
EXPECTED_CLASSES = {
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
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data/raw/UrbanSound8K"


@dataclass
class DatasetValidation:
    """Validation results for a local UrbanSound8K dataset."""

    metadata_rows: int = 0
    missing_audio: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors and not self.missing_audio


def validate_dataset(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> DatasetValidation:
    """Check metadata shape, all ten folds, and every metadata audio reference."""
    root = Path(dataset_dir)
    result = DatasetValidation()
    metadata_path = root / "metadata/UrbanSound8K.csv"

    if not metadata_path.is_file():
        result.errors.append(f"Metadata CSV not found: {metadata_path}")
        return result

    for fold in range(1, 11):
        fold_dir = root / "audio" / f"fold{fold}"
        if not fold_dir.is_dir():
            result.errors.append(f"Missing audio directory: {fold_dir}")

    try:
        with metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
            reader = csv.DictReader(metadata_file)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_COLUMNS - columns
            if missing_columns:
                result.errors.append(
                    f"Metadata is missing required columns: {', '.join(sorted(missing_columns))}"
                )
                return result

            classes: set[str] = set()
            folds: set[int] = set()
            filenames: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                result.metadata_rows += 1
                filename = (row.get("slice_file_name") or "").strip()
                class_name = (row.get("class") or "").strip()
                if class_name:
                    classes.add(class_name)

                try:
                    fold = int(row.get("fold", ""))
                except ValueError:
                    result.errors.append(f"Invalid fold value on metadata row {line_number}.")
                    continue
                if fold not in range(1, 11):
                    result.errors.append(
                        f"Fold must be 1 through 10 on metadata row {line_number}."
                    )
                    continue
                folds.add(fold)

                if not filename or Path(filename).name != filename or "\\" in filename:
                    result.errors.append(
                        f"Invalid audio filename on metadata row {line_number}: {filename!r}"
                    )
                    continue
                if filename in filenames:
                    result.errors.append(
                        f"Duplicate audio filename on metadata row {line_number}: {filename}"
                    )
                filenames.add(filename)
                audio_path = root / "audio" / f"fold{fold}" / filename
                if not audio_path.is_file():
                    result.missing_audio.append(str(audio_path))

            if result.metadata_rows != EXPECTED_ROWS:
                result.errors.append(
                    f"Expected {EXPECTED_ROWS:,} metadata rows; found {result.metadata_rows:,}."
                )
            if classes != EXPECTED_CLASSES:
                result.errors.append(
                    "Expected all ten UrbanSound8K classes; "
                    f"found {len(classes)} distinct class labels."
                )
            if folds != set(range(1, 11)):
                missing_folds = sorted(set(range(1, 11)) - folds)
                result.errors.append(f"Metadata does not represent folds: {missing_folds}.")
    except (OSError, UnicodeError, csv.Error) as exc:
        result.errors.append(f"Could not read metadata CSV: {exc}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()
    result = validate_dataset(args.dataset_dir)

    if result.valid:
        print(
            f"PASS: {result.metadata_rows:,} metadata rows, 10 classes, all 10 folds, "
            "and all referenced audio files are present."
        )
        return 0

    print(
        f"FAIL: {result.metadata_rows:,} metadata rows; "
        f"{len(result.missing_audio):,} referenced audio files missing."
    )
    for error in result.errors:
        print(f"  - {error}")
    if result.missing_audio:
        for path in result.missing_audio[:5]:
            print(f"  - Missing audio: {path}")
        if len(result.missing_audio) > 5:
            print(f"  - ... and {len(result.missing_audio) - 5:,} more missing audio files.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
