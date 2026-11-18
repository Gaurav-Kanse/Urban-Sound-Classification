# SoundSense: Urban Sound Classification

SoundSense is a Python project for classifying short urban audio recordings into the ten UrbanSound8K categories. It includes dataset preparation and EDA, a Random Forest feature baseline, a compact PyTorch CNN, shared audio preprocessing, a FastAPI inference service, and a Streamlit dashboard.

## Dataset

SoundSense uses the official [UrbanSound8K dataset](https://zenodo.org/records/1203745). It contains 8,732 labeled audio excerpts, ten sound classes, and ten predefined folds. The dataset is large (the official archive is about 6 GB) and is not included in this repository because of its size and submission limits. This repository contains the project code and setup scripts, not the raw recordings.

The archive is distributed under the license and terms shown on the official Zenodo record. Please cite Salamon, Jacoby, and Bello, “A Dataset and Taxonomy for Urban Sound Research,” ACM Multimedia, 2014, when using it.

## Installation

Python 3.13 is used for the current project environment (the package supports Python 3.10–3.13). From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Keep the virtual environment active for the commands below. On Fedora, audio decoding may also require system `libsndfile` and FFmpeg packages.

## Download and verify the dataset

Ensure you have enough free disk space for the archive and extracted audio. The setup script obtains the archive URL, expected size, and checksum from the official Zenodo record. It shows download progress, preserves an interrupted partial download for resuming, checks the archive before extraction, and safely extracts into a staging directory. It does not replace a valid existing dataset.

```bash
python scripts/download_dataset.py
python scripts/verify_dataset.py
```

The verifier checks that the metadata CSV is readable and has the expected columns and dataset row count, all ten audio folders exist, the ten classes and folds are represented, and every metadata-referenced audio file is present. It exits nonzero and lists a concise failure summary if validation fails.

Expected layout:

```text
data/
├── downloads/
│   └── UrbanSound8K.tar.gz
└── raw/
    └── UrbanSound8K/
        ├── audio/
        │   ├── fold1/
        │   ├── fold2/
        │   ├── ...
        │   └── fold10/
        └── metadata/
            └── UrbanSound8K.csv
```

The compressed archive is retained in `data/downloads/` so it can be reused. Dataset files are ignored by Git.

## Preparation and EDA

After download and successful verification, these project scripts validate audio readability and produce the dataset summary and EDA outputs:

```bash
python scripts/prepare_data.py
python scripts/run_eda.py
```

The preparation summary is written to `reports/results/dataset_summary.json`; EDA figures and statistics are written under `reports/figures/` and `reports/results/`.

## Training and evaluation

The commands below require the dataset to be downloaded and verified first. The established fold split is folds 1–8 for training, fold 9 for validation/model selection, and fold 10 for final testing.

```bash
python scripts/train_baseline.py
python scripts/train_cnn.py --epochs 25 --batch-size 32 --patience 6
python scripts/evaluate.py
```

The Random Forest script trains and evaluates the baseline; the CNN script trains the CNN, selects its validation checkpoint, and evaluates it on the held-out fold. `evaluate.py` prints the metric artifacts created by training. Checkpoints, configurations, metrics, and figures are stored under `artifacts/` and `reports/`. No models are claimed to have been trained and no experimental results are reported by this repository setup documentation; results remain pending until the dataset is downloaded and experiments are actually run.

## API

After training the CNN, start FastAPI from the repository root:

```bash
python -m uvicorn soundsense.api.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for Swagger. Prediction routes return HTTP 503 until a trained CNN checkpoint is available. Set `SOUNDSENSE_MODEL` to select a checkpoint.

## Dashboard

In another terminal with the virtual environment active:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard reports when dataset summaries or trained artifacts are not available.

## Troubleshooting

- **Insufficient disk space:** Free space before retrying. The archive is about 6 GB; extraction needs additional space. The downloader checks available space before downloading and extraction.
- **Download interrupted:** Run `python scripts/download_dataset.py` again. The `.part` file is retained and a byte-range resume is attempted when supported by the server.
- **Archive extraction or checksum failure:** The downloader verifies the official Zenodo checksum and extracts to a temporary staging location. Review the error, ensure sufficient space, and retry. A failed extraction is not installed as a dataset.
- **Missing metadata CSV:** Confirm that the archive came from the official [Zenodo record](https://zenodo.org/records/1203745), then rerun the download. The expected path is `data/raw/UrbanSound8K/metadata/UrbanSound8K.csv`.
- **Missing audio folds or recordings:** Run `python scripts/verify_dataset.py`. If validation fails after extraction, move the incomplete `data/raw/UrbanSound8K/` directory aside and rerun the download; the script will not overwrite an existing invalid destination.
- **Incorrect Python environment:** Activate the project environment with `source .venv/bin/activate`, then verify `python --version` and reinstall project dependencies with `python -m pip install -e '.[dev]'`.

## Repository layout

```text
src/soundsense/       reusable package: data, features, models, training, inference, API
scripts/              download, verification, preparation, EDA, training, evaluation
dashboard/app.py      Streamlit user interface
data/                 local dataset and processed data (ignored by Git)
artifacts/            checkpoints, metrics, and model configurations
reports/              final report, figures, result summaries, checklist
docs/                 dataset and methodology documentation
tests/                unit tests, including offline dataset setup tests
```

## Limitations

UrbanSound8K has class imbalance, recording variation, and clips with overlapping sources. The classifier is closed-set and has no calibrated probabilities or out-of-distribution detection. Generalization beyond the dataset has not been established.

## Author

Student project submission. Add the author's name, course, and institution before academic submission.

## License

Project source is provided under MIT; dataset use is governed by the original dataset terms and source licenses. See [LICENSE](LICENSE).
