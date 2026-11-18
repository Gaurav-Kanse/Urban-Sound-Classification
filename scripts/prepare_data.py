"""Validate official UrbanSound8K download and write a dataset summary."""

import json
import logging

from soundsense.config import PATHS
from soundsense.data.dataset import DatasetError, dataset_summary, validate_metadata
from soundsense.features.audio import load_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> int:
    try:
        frame, missing = validate_metadata(PATHS["metadata"], PATHS["dataset"])
    except DatasetError as exc:
        logging.error("%s", exc)
        return 2
    summary = dataset_summary(frame, missing)
    corrupt = []
    for audio_path in frame.loc[~frame["audio_path"].isin(map(str, missing)), "audio_path"]:
        try:
            load_audio(audio_path)
        except (ValueError, OSError) as exc:
            corrupt.append({"path": audio_path, "error": str(exc)})
    summary["corrupt_audio_files"] = corrupt
    summary["represented_classes"] = sorted(frame["class"].unique().tolist())
    summary["all_ten_classes_represented"] = len(summary["represented_classes"]) == 10
    PATHS["results"].mkdir(parents=True, exist_ok=True)
    target = PATHS["results"] / "dataset_summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(
        "Validated %d metadata rows; %d audio paths missing; %d files unreadable.",
        len(frame),
        len(missing),
        len(corrupt),
    )
    logging.info("Class distribution: %s", summary["class_distribution"])
    logging.info("Fold distribution: %s", summary["fold_distribution"])
    return (
        1
        if missing
        or corrupt
        or not summary["all_ten_classes_represented"]
        or not summary["all_ten_folds_represented"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
