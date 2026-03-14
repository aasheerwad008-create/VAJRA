"""
VAJRA Voice AI — Audio utility functions.

Provides reusable helpers for:
  - Loading and resampling audio (any format → 16 kHz mono float32)
  - Mel-spectrogram and MFCC feature extraction
  - Waveform padding / trimming to fixed length
  - Cosine similarity for speaker embeddings
  - RMS energy measurement

All functions are pure numpy/torchaudio — no side effects.
"""
from __future__ import annotations

import io
from typing import Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# ── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000
N_MELS: int = 128
N_MFCC: int = 40
N_FFT: int = 1024
HOP_LENGTH: int = 512
CHUNK_SECONDS: float = 2.0
CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_SECONDS)


# ── Audio loading ──────────────────────────────────────────────────────────

def load_audio(
    source: Union[str, bytes, io.IOBase],
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio from a file path, raw bytes, or file-like object.
    Returns a float32 numpy array normalised to [-1, 1] at *target_sr* Hz, mono.

    Parameters
    ----------
    source:
        Path string, raw audio bytes, or seekable file-like object.
    target_sr:
        Output sample rate (default 16 000 Hz).

    Returns
    -------
    np.ndarray, shape (T,), dtype float32.
    """
    import librosa

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    waveform, sr = librosa.load(source, sr=target_sr, mono=True)
    return waveform.astype(np.float32)


# ── Waveform helpers ───────────────────────────────────────────────────────

def pad_or_trim(waveform: np.ndarray, length: int = CHUNK_SAMPLES) -> np.ndarray:
    """
    Ensure *waveform* is exactly *length* samples.
    Trims from the end if too long; zero-pads from the end if too short.
    """
    if len(waveform) >= length:
        return waveform[:length]
    return np.pad(waveform, (0, length - len(waveform)), mode="constant")


def rms_energy(waveform: np.ndarray) -> float:
    """Compute root-mean-square energy of a waveform (float in [0, 1])."""
    rms = float(np.sqrt(np.mean(waveform ** 2) + 1e-9))
    return min(1.0, rms)


# ── Feature extraction ─────────────────────────────────────────────────────

def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    to_db: bool = True,
) -> np.ndarray:
    """
    Compute a mel-spectrogram from a 1-D waveform array.

    Parameters
    ----------
    waveform:    float32 numpy array, shape (T,).
    sample_rate: source sample rate.
    n_mels:      number of mel filter banks.
    n_fft:       FFT window size.
    hop_length:  number of samples between successive frames.
    to_db:       if True convert power to dB scale.

    Returns
    -------
    np.ndarray, shape (n_mels, time_frames), dtype float32.
    """
    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(tensor)  # (1, n_mels, time)

    if to_db:
        mel = T.AmplitudeToDB()(mel)

    return mel.squeeze(0).numpy()  # (n_mels, time)


def compute_mfcc(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    Compute MFCCs from a 1-D waveform array.

    Returns
    -------
    np.ndarray, shape (n_mfcc, time_frames), dtype float32.
    """
    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_mels": n_mels},
    )
    mfcc = mfcc_transform(tensor)  # (1, n_mfcc, time)
    return mfcc.squeeze(0).numpy()  # (n_mfcc, time)


# ── Similarity ─────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D embedding vectors.
    Returns a float in [-1, 1].
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))
