# SoundSense: Urban Sound Classification

## Abstract

SoundSense implements a fold-aware urban audio classification workflow for UrbanSound8K, comparing a feature-based Random Forest with a compact log-Mel convolutional neural network. The repository also contains reusable audio preprocessing, dataset setup and verification scripts, evaluation and visualization scripts, a FastAPI service, and a Streamlit dashboard. UrbanSound8K is downloaded separately from the official Zenodo record; the repository contains project code and scripts, not the large raw dataset. The dataset and trained weights were not available in the inspected workspace, so empirical dataset summaries, model scores, and data-derived figures cannot be reported. Unit tests validate code behavior and do not establish model performance.

## 1. Introduction

Urban sound classification assigns short environmental recordings to categories such as traffic, machinery, and human activity. It is a useful benchmark for audio feature engineering and convolutional learning because recordings vary in duration, loudness, background noise, and source overlap.

## 2. Problem statement

Given one recording, predict one of the ten known UrbanSound8K labels. The system must apply the same audio transformation in training and inference and preserve the dataset's official folds to avoid using held-out recordings for model selection.

## 3. Objectives

- Validate the official dataset metadata and referenced audio files.
- Compare a standardized Random Forest against a compact CNN.
- Select CNN checkpoints using validation fold 9 and reserve fold 10 for final evaluation.
- Report class-sensitive metrics, especially macro F1.
- Expose audio predictions through an API and dashboard with clear artifact status.

## 4. Dataset description

UrbanSound8K comprises 8,732 short excerpts organized into ten predefined folds and ten classes: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music. The official source is the [UrbanSound8K Zenodo record](https://zenodo.org/records/1203745). The raw dataset is not included in this repository because of its size and submission limits; `scripts/download_dataset.py` obtains it separately and `scripts/verify_dataset.py` checks the resulting files. The required files were absent from this workspace at verification time. Consequently, the class and fold counts, missing/corrupt recording counts, and duration statistics are unavailable here.

## 4.1 Dataset setup

The README documents environment installation, one-command download, dataset verification, the expected `data/raw/UrbanSound8K/` layout, and troubleshooting. The archive is stored under ignored `data/downloads/`; extracted raw data is stored under ignored `data/raw/`. The repository submission contains the setup code and documentation only, not the archive or extracted recordings.

## 5. Exploratory data analysis

The EDA script generates class and fold count plots, duration distribution, example waveform, log-Mel and MFCC views, and a feature correlation heatmap. These are computed from the real dataset and are not checked in as mock figures. EDA was not run because the metadata and audio files were not present.

## 6. Preprocessing and features

Librosa loads recordings as mono at 22,050 Hz. Samples are peak-normalized, center-cropped or symmetrically zero-padded to four seconds. The CNN input is a 128-band power Mel spectrogram using FFT size 2,048 and hop length 512. Power is converted to dB relative to the clip peak and mapped into a bounded 0–1 range. The preprocessing configuration is stored in the selected CNN checkpoint and reused by inference.

The baseline feature vector contains per-coefficient MFCC means and standard deviations, plus mean and standard deviation for zero-crossing rate, spectral centroid, spectral bandwidth, and spectral rolloff.

## 7. Model architecture

The classical baseline is a `StandardScaler` followed by a class-balanced Random Forest. The CNN has four convolutional blocks with 16, 32, 64, and 128 channels. Each block uses convolution, batch normalization, ReLU, max pooling, and dropout. Adaptive global average pooling and a fully connected layer produce ten logits.

## 8. Training methodology

Folds 1–8 are used for training, fold 9 for validation and model selection, and fold 10 for the final test only. CNN optimization uses AdamW and cross-entropy. A ReduceLROnPlateau scheduler monitors validation macro F1; early stopping and checkpointing also use validation macro F1. Python, NumPy, and PyTorch seeds are set. Small time shifts, gain changes, and Gaussian noise are applied only to training audio. The scripts select CUDA when available and otherwise use CPU.

No training run was possible because the official dataset was absent. Epoch counts and samples processed are therefore not applicable for this checkout.

## 9. Evaluation metrics

The evaluation code computes accuracy, macro precision, macro recall, macro F1, weighted F1, per-class precision/recall/F1/support, and a fixed-label confusion matrix. Macro F1 is the primary metric because it gives each category equal weight. Fold 10 is used only after model selection. No metrics are reported without a completed run.

## 10. Experimental results

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Status |
|---|---:|---:|---:|---|
| Random Forest | — | — | — | Not trained: dataset unavailable |
| SoundCNN | — | — | — | Not trained: dataset unavailable |

The dashes indicate unavailable measurements, not zero performance. No classification report, result JSON/CSV, confusion matrix, or training curve was generated because there were no recordings to process.

## 11. Discussion

The repository has a consistent preprocessing implementation, explicit fold-aware training code, metrics and visualization writers, prediction interfaces, and automated dataset download and verification. The unit tests cover mocked network responses, temporary datasets, synthetic inputs, and code paths only. They cannot support claims about recognition quality, class-wise behavior, or generalization.

## 12. Limitations

The core external dependency is the separately downloaded official UrbanSound8K dataset. Actual training and evaluation results remain pending until the dataset is downloaded, verified, and experiments are run. The current project has no empirical results or real-data plots. UrbanSound8K recordings may contain overlapping sounds and varied recording conditions; a ten-class closed-set model may mislabel out-of-domain audio. Softmax outputs are uncalibrated scores. Author, course, and institution information also remains to be supplied.

## 13. Conclusion

SoundSense provides reproducible implementation, dataset setup instructions, and tested lightweight code paths, but it is not yet empirically submission-complete. Download and verify the official dataset separately, run preparation and EDA, train both models, inspect the held-out results, and fill in author information before making performance claims or submitting a results-based report. No model is claimed to have been trained.

## 14. References

- Salamon, J., Jacoby, C., and Bello, J. P. “A Dataset and Taxonomy for Urban Sound Research.” ACM Multimedia, 2014.
- UrbanSound8K official Zenodo record: <https://zenodo.org/records/1203745>.
