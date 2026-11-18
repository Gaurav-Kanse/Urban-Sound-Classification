# SoundSense submission checklist

Status below reflects work completed in the inspected workspace on 2026-11-18. “Pending” means the official dataset or a local runtime artifact is still required; no data or metrics have been invented.

| Item | Status | Evidence / remaining action |
|---|---|---|
| Project audit and Python 3.13 environment | Completed | Active `.venv` uses Python 3.13.18; editable `soundsense` package and dependencies are installed. |
| Dataset delivery | Completed | The raw dataset is distributed separately from the official Zenodo record; this repository contains code and setup scripts, not the large audio archive or extracted data. |
| Dataset setup instructions | Completed | README documents installation, `scripts/download_dataset.py`, `scripts/verify_dataset.py`, expected layout, preparation, EDA, training, evaluation, API, dashboard, and troubleshooting. |
| Dataset verified locally | Pending | No claim is made that the archive has been downloaded or the local dataset verified. Run `python scripts/download_dataset.py` and `python scripts/verify_dataset.py`. |
| Dataset summary and corruption scan | Pending | `scripts/prepare_data.py` needs the separately downloaded and verified official files. |
| EDA figures | Pending | `scripts/run_eda.py` needs the downloaded dataset; no generated data figures are claimed. |
| Baseline trained | Pending | Requires the separately downloaded and verified UrbanSound8K dataset. |
| CNN trained | Pending | Requires the separately downloaded and verified UrbanSound8K dataset. |
| Final test evaluation | Pending | Requires trained models and validation-based selection; no claim is made that fold 10 has been used. |
| Model artifacts and measured metrics | Pending | Actual training and evaluation results remain pending until the dataset is downloaded and experiments are run. |
| Inference pipeline | Completed | Checkpoint-backed prediction returns class, uncalibrated confidence, top three, and all ten scores; preprocessing config is loaded from the checkpoint. |
| API route/handler checks | Completed | Health/info, unsupported upload extension, and batch count checks are covered by unit tests. Predictions require trained CNN weights. |
| Live API verification | Completed, model-dependent predictions blocked | Uvicorn launched; `/`, `/health`, `/model-info` returned HTTP 200, and invalid `/predict`/over-limit `/predict/batch` uploads returned HTTP 415/400. No checkpoint is available for successful predictions. |
| Dashboard implementation | Completed | Overview, classification, performance, and About views handle missing artifacts. |
| Live dashboard verification | Completed, visual review pending | Streamlit served HTTP 200; Streamlit AppTest ran all four pages with zero exceptions. No screenshot is included. |
| Unit tests | Completed | `17 passed`; new download tests mock Zenodo responses and temporary archives/directories. No real dataset download is performed. |
| Ruff | Completed | `python -m ruff check .` passes. |
| Black | Completed | `python -m black .` applied formatting; `python -m black --check .` passes. |
| Compile check | Completed | `python -m compileall -q src scripts dashboard tests` passes. |
| Documentation | Completed with data-dependent sections pending | README, methodology, dataset notes, final report, and this checklist are present; add empirical results and author/course details after experiments. |
| Git tracking audit | Limited | The checkout has no tracked/indexed files; project files appear untracked. No archive or extracted dataset is present, and ignore rules cover all requested dataset/checkpoint paths. |

## Commands

Run these from the project root:

```bash
# Activate the project environment
source .venv/bin/activate

# Install project and development dependencies into the active environment
python -m pip install -e '.[dev]'

# Download the official dataset separately and validate its layout and contents
python scripts/download_dataset.py
python scripts/verify_dataset.py

# Check audio readability and summarize the dataset
python scripts/prepare_data.py

# Generate EDA
python scripts/run_eda.py

# Train both models and evaluate on the held-out test fold
python scripts/train_baseline.py
python scripts/train_cnn.py --epochs 25 --batch-size 32 --patience 6
python scripts/evaluate.py

# Run quality checks
python -m pytest -v
python -m ruff check .
python -m black --check .
python -m compileall -q src scripts dashboard tests

# Launch the API
python -m uvicorn soundsense.api.app:app --host 127.0.0.1 --port 8000

# Launch the dashboard in a second terminal
python -m streamlit run dashboard/app.py
```

The preparation script returns a nonzero exit status when paths are missing, files are unreadable, or one of the ten classes is absent. Inspect `reports/results/dataset_summary.json` before training. The evaluation and final report must be updated from actual files generated under `artifacts/metrics/` and `reports/figures/`; do not present expected, placeholder, or invented metrics as completed experiments.

The raw dataset is excluded from Git and submission packages because of its size. Dataset setup is documented in the README and `docs/dataset.md`. This checklist does not claim that training or evaluation has been completed.
