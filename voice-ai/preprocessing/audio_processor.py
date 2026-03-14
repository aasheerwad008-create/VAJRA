"""
VAJRA Voice AI — Audio Preprocessing Pipeline.

Provides a reusable audio processor for loading, resampling,
normalizing, and preparing audio for model consumption.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_DURATION_S = 2.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)


class AudioProcessor:
    """
    Standardized audio preprocessing pipeline.

    Converts any audio input to 16 kHz mono float32, trimmed or padded
    to a fixed duration.

    Parameters
    ----------
    sample_rate : int
        Target sample rate (default 16 000).
    max_duration_s : float
        Target duration in seconds (default 2.0).
    normalize : bool
        If True, peak-normalize waveform to [-1, 1] (default True).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_duration_s: float = MAX_DURATION_S,
        normalize: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)
        self.normalize = normalize

    def process_file(self, path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess an audio file. Returns (T,) float32 array."""
        waveform, sr = torchaudio.load(str(path))
        return self._process_tensor(waveform, sr)

    def process_bytes(self, raw: bytes) -> np.ndarray:
        """Load and preprocess raw audio bytes. Returns (T,) float32 array."""
        import librosa

        buf = io.BytesIO(raw)
        waveform, sr = librosa.load(buf, sr=self.sample_rate, mono=True)
        return self._finalize(waveform)

    def process_tensor(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocess a waveform tensor.

        Parameters
        ----------
        waveform : Tensor, shape (C, T) or (T,)
        sr : int, source sample rate

        Returns
        -------
        Tensor, shape (T,) — preprocessed waveform
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return torch.from_numpy(self._process_tensor(waveform, sr))

    def _process_tensor(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Internal: process a (C, T) tensor to a (T,) numpy array."""
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        arr = waveform.squeeze(0).numpy().astype(np.float32)
        return self._finalize(arr)

    def _finalize(self, waveform: np.ndarray) -> np.ndarray:
        """Trim/pad and optionally normalize."""
        # Trim or pad
        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]
        elif len(waveform) < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        # Peak normalization
        if self.normalize:
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak

        return waveform.astype(np.float32)
