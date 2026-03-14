"""
VAJRA Voice AI — Advanced Audio Augmentations.

Implements data augmentation techniques for training robustness:
    - Gaussian noise injection
    - Pitch shifting
    - Time stretching
    - Background noise mixing
    - Reverberation simulation
    - Time masking

Augmentations are applied only to **real (bonafide) speech** samples
to increase training diversity without generating synthetic artifacts.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class AudioAugmentor:
    """
    Audio augmentation pipeline for training.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (default 16 000).
    augment_prob : float
        Probability of applying each augmentation (default 0.5).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        augment_prob: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob

    def augment(
        self,
        waveform: np.ndarray,
        is_real: bool = True,
    ) -> np.ndarray:
        """
        Apply random augmentations to a waveform.

        Only real (bonafide) samples are augmented to avoid creating
        misleading synthetic artifacts in fake samples.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        is_real : bool
            If False, returns waveform unchanged.

        Returns
        -------
        ndarray, shape (T,)
        """
        if not is_real:
            return waveform

        if random.random() < self.augment_prob:
            waveform = self.add_gaussian_noise(waveform)
        if random.random() < self.augment_prob:
            waveform = self.time_mask(waveform)
        if random.random() < self.augment_prob * 0.5:
            waveform = self.pitch_shift(waveform)
        if random.random() < self.augment_prob * 0.5:
            waveform = self.time_stretch(waveform)
        if random.random() < self.augment_prob * 0.3:
            waveform = self.add_reverb(waveform)

        return waveform

    def add_gaussian_noise(
        self,
        waveform: np.ndarray,
        snr_db: float = 20.0,
    ) -> np.ndarray:
        """
        Add Gaussian white noise at a given SNR.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        snr_db : float
            Signal-to-noise ratio in dB (default 20).
        """
        signal_power = np.mean(waveform ** 2)
        if signal_power < 1e-10:
            return waveform

        # Randomize SNR between snr_db - 10 and snr_db + 5
        actual_snr = snr_db + random.uniform(-10, 5)
        noise_power = signal_power / (10 ** (actual_snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
        return (waveform + noise).astype(np.float32)

    def pitch_shift(
        self,
        waveform: np.ndarray,
        semitones_range: tuple = (-2, 2),
    ) -> np.ndarray:
        """
        Apply pitch shifting using phase vocoder approximation.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        semitones_range : tuple
            Range of semitones to shift (default (-2, 2)).
        """
        try:
            import librosa

            semitones = random.uniform(*semitones_range)
            shifted = librosa.effects.pitch_shift(
                waveform, sr=self.sample_rate, n_steps=semitones
            )
            return shifted.astype(np.float32)
        except ImportError:
            return waveform

    def time_stretch(
        self,
        waveform: np.ndarray,
        rate_range: tuple = (0.9, 1.1),
    ) -> np.ndarray:
        """
        Apply time stretching.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        rate_range : tuple
            Range of stretch rates (default (0.9, 1.1)).
        """
        try:
            import librosa

            rate = random.uniform(*rate_range)
            stretched = librosa.effects.time_stretch(waveform, rate=rate)
            # Maintain original length
            orig_len = len(waveform)
            if len(stretched) > orig_len:
                stretched = stretched[:orig_len]
            elif len(stretched) < orig_len:
                stretched = np.pad(stretched, (0, orig_len - len(stretched)))
            return stretched.astype(np.float32)
        except ImportError:
            return waveform

    def add_reverb(
        self,
        waveform: np.ndarray,
        decay: float = 0.3,
    ) -> np.ndarray:
        """
        Simulate simple reverberation using exponential decay.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        decay : float
            Reverb decay factor (default 0.3).
        """
        decay = random.uniform(0.1, decay)
        delay_samples = int(self.sample_rate * random.uniform(0.01, 0.05))

        reverb = np.zeros_like(waveform)
        if delay_samples < len(waveform):
            reverb[delay_samples:] = waveform[:-delay_samples] * decay

        result = waveform + reverb
        # Prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak
        return result.astype(np.float32)

    def time_mask(
        self,
        waveform: np.ndarray,
        max_mask_pct: float = 0.15,
    ) -> np.ndarray:
        """Zero out a random contiguous segment of the waveform."""
        T = len(waveform)
        mask_len = int(T * max_mask_pct * random.random())
        if mask_len == 0:
            return waveform
        start = random.randint(0, max(0, T - mask_len))
        result = waveform.copy()
        result[start : start + mask_len] = 0.0
        return result

    def add_background_noise(
        self,
        waveform: np.ndarray,
        noise_waveform: np.ndarray,
        snr_db: float = 15.0,
    ) -> np.ndarray:
        """
        Mix background noise into the signal at a given SNR.

        Parameters
        ----------
        waveform : ndarray, shape (T,)
        noise_waveform : ndarray, shape (T_noise,)
            A noise waveform to mix in.
        snr_db : float
            Signal-to-noise ratio in dB.
        """
        # Trim or loop noise to match waveform length
        if len(noise_waveform) < len(waveform):
            repeats = len(waveform) // len(noise_waveform) + 1
            noise_waveform = np.tile(noise_waveform, repeats)
        noise_waveform = noise_waveform[: len(waveform)]

        signal_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise_waveform ** 2)
        if noise_power < 1e-10 or signal_power < 1e-10:
            return waveform

        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        result = waveform + scale * noise_waveform
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak
        return result.astype(np.float32)
