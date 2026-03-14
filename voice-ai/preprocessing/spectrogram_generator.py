"""
VAJRA Voice AI — Mel Spectrogram Generator.

Generates 128×128 mel spectrograms for the EfficientNet-B0 classifier.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


class SpectrogramGenerator:
    """
    Generate normalized mel spectrograms from audio waveforms.

    Parameters
    ----------
    sample_rate : int
        Input sample rate (default 16 000).
    n_mels : int
        Number of mel filter banks (default 128).
    n_fft : int
        FFT window size (default 1024).
    hop_length : int
        Hop length between frames (default 512).
    target_size : tuple
        (height, width) to resize the spectrogram (default (128, 128)).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        target_size: tuple = (128, 128),
    ) -> None:
        self.sample_rate = sample_rate
        self.target_size = target_size

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amp_to_db = T.AmplitudeToDB()

    def generate(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate a normalized mel spectrogram.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
            Audio waveform at ``self.sample_rate`` Hz.

        Returns
        -------
        ndarray, shape (1, H, W) — normalized dB-scale mel spectrogram.
        """
        tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        mel = self.mel_transform(tensor)  # (1, n_mels, time)
        mel = self.amp_to_db(mel)

        # Resize to target size
        mel = mel.unsqueeze(0)  # (1, 1, n_mels, time)
        mel = torch.nn.functional.interpolate(
            mel, size=self.target_size, mode="bilinear", align_corners=False
        )
        mel = mel.squeeze(0)  # (1, H, W)

        # Normalize to [0, 1]
        mel_np = mel.numpy()
        mel_min = mel_np.min()
        mel_max = mel_np.max()
        if mel_max - mel_min > 0:
            mel_np = (mel_np - mel_min) / (mel_max - mel_min)

        return mel_np.astype(np.float32)

    def generate_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Generate mel spectrograms for a batch.

        Parameters
        ----------
        waveforms : Tensor, shape (B, T)

        Returns
        -------
        Tensor, shape (B, 1, H, W)
        """
        mel = self.mel_transform(waveforms)  # (B, n_mels, time)
        mel = self.amp_to_db(mel)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time)
        mel = torch.nn.functional.interpolate(
            mel, size=self.target_size, mode="bilinear", align_corners=False
        )
        return mel
