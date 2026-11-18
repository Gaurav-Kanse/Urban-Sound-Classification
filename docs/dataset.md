# Dataset

UrbanSound8K is distributed separately from this repository because the official archive is about 6 GB and the raw data is not suitable for the repository or submission package. The project includes scripts and documentation for downloading and validating the data.

## Official source and layout

Use the official [Zenodo record](https://zenodo.org/records/1203745). From the repository root, activate the project virtual environment, then run:

```bash
python scripts/download_dataset.py
python scripts/verify_dataset.py
```

The downloader resolves the archive URL and checksum from Zenodo, supports partial-download resume where available, and safely extracts only the expected dataset tree. It retains the archive in `data/downloads/` and installs the dataset under:

```text
data/raw/UrbanSound8K/
├── audio/
│   ├── fold1/ ... fold10/
└── metadata/UrbanSound8K.csv
```

UrbanSound8K contains 8,732 labeled excerpts in ten predefined folds and ten sound classes. The verifier checks the metadata columns and row count, class/fold representation, all fold directories, and all referenced audio paths. Run `python scripts/prepare_data.py` after verification to check audio readability and write a summary to `reports/results/dataset_summary.json`.

## Folds and categories

Folds 1–8 are training, fold 9 validation, fold 10 final test. Classes are air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music.

The collection consists of short excerpts from Freesound, with varied recording conditions, background sources, clip lengths, and class counts. Fold-aware evaluation is required to preserve the official split protocol. Data usage remains subject to the original dataset terms and source licenses.
