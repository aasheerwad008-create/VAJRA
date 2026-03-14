"""
VAJRA Voice AI — Codec Artifact Detector Model.

Lightweight 1-D CNN that classifies audio waveforms into:
    - HUMAN (genuine speech)
    - ENCODEC (Meta EnCodec neural codec artifacts)
    - SOUNDSTREAM (Google SoundStream artifacts)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CodecDetectorModel(nn.Module):
    """
    1-D CNN for neural codec artifact detection.

    Input: raw waveform at 16 kHz, 2 seconds (32 000 samples).
    Output: 3-class logits [HUMAN, ENCODEC, SOUNDSTREAM].

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 3).
    base_channels : int
        Number of channels in the first conv layer (default 32).
    dropout : float
        Dropout rate before the final linear layer (default 0.3).
    """

    def __init__(
        self,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, T) → (32, T/4)
            nn.Conv1d(1, base_channels, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),

            # Block 2: (32, T/4) → (64, T/16)
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Block 3: (64, T/16) → (128, T/64)
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Block 4: (128, T/256)
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        self._num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        waveform : Tensor, shape (B, T)
            Raw audio waveform.

        Returns
        -------
        Tensor, shape (B, num_classes) — class logits.
        """
        x = waveform.unsqueeze(1)  # (B, 1, T)
        features = self.features(x)  # (B, base_channels*4)
        return self.classifier(features)  # (B, num_classes)
