# Methodology

## Data split

The original UrbanSound8K fold labels are retained. Folds 1–8 form the training set, fold 9 is used to select the CNN checkpoint and compare configurations, and fold 10 is evaluated once after selection. No excerpt is moved between folds.

## Audio preprocessing and features

Librosa decodes each file as mono at 22,050 Hz. A peak-normalized centered crop or symmetric zero padding makes the signal four seconds long. The primary CNN input is a 128-band power Mel spectrogram (FFT 2,048, hop 512), converted to dB relative to its peak and bounded to a stable [0, 1] range using an 80 dB window. The same implementation serves training, API, and dashboard inference. The baseline summarizes MFCC means and standard deviations plus mean and standard deviation of zero-crossing rate, spectral centroid, bandwidth, and rolloff.

MFCCs compactly describe the spectral envelope after a perceptual frequency warp; they are useful compact inputs for classical models. A Mel spectrogram displays energy over time and perceptually spaced frequency bands, which lets a CNN learn localized spectral-temporal patterns.

## CNN and training

The CNN stacks four Conv2D, BatchNorm, ReLU, max-pool, and dropout blocks with channel widths 16, 32, 64, and 128. Adaptive global average pooling feeds a dropout and ten-logit linear classifier. Cross-entropy and AdamW are used, with ReduceLROnPlateau on validation macro F1 and early stopping. Python, NumPy, and PyTorch random generators are seeded. Training-only augmentation applies small circular time shifts, gain changes, and low-level Gaussian noise.

## Evaluation

Random Forest features are standardized in a scikit-learn pipeline. Reported metrics include accuracy, macro precision/recall/F1, weighted F1, per-class report, and a fixed-label confusion matrix. Macro F1 weights each class equally, so frequent classes cannot hide poor performance on rare categories; accuracy alone can conceal this imbalance. Softmax outputs are model scores and are not described as calibrated probabilities.
