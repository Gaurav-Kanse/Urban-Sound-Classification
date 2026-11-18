"""Compact convolutional network for log-Mel spectrograms."""

import torch
from torch import nn


class SoundCNN(nn.Module):
    """Four-stage CNN with global average pooling for ten sound classes."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in (16, 32, 64, 128):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.10),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for cross entropy."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.pool(self.features(x)))
