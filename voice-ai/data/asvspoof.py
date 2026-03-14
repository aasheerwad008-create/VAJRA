"""
VAJRA Voice AI — ASVspoof 2024 Dataset Loader.

Provides a PyTorch Dataset class for the ASVspoof 2024 corpus
(Track 1: deepfake speech detection) and helper utilities for
loading the protocol metadata files.

Expected directory layout:
    data/ASVspoof2024/
        LA/
            ASVspoof2024_LA_cm_protocols/
                ASVspoof2024.LA.cm.train.trn.txt
                ASVspoof2024.LA.cm.dev.trl.txt
                ASVspoof2024.LA.cm.eval.trl.txt
            ASVspoof2024_LA_train/
                flac/  *.flac
            ASVspoof2024_LA_dev/
                flac/  *.flac
            ASVspoof2024_LA_eval/
                flac/  *.flac

Usage:
    from data.asvspoof import ASVspoof2024Dataset

    train_ds = ASVspoof2024Dataset(
        root="/data/ASVspoof2024/LA",
        split="train",
        sample_rate=16_000,
    )
    waveform, label, meta = train_ds[0]
    # label: 0 = bonafide (REAL), 1 = spoof (FAKE)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


# ── Metadata ───────────────────────────────────────────────────────────────

@dataclass
class UtteranceMeta:
    """Metadata for a single ASVspoof utterance."""
    utt_id: str
    speaker_id: str
    system_id: str     # spoofing system or '-' for bonafide
    label: int         # 0 = bonafide, 1 = spoof
    label_str: str     # "bonafide" | "spoof"


# ── Protocol parser ────────────────────────────────────────────────────────

_SPLIT_PROTOCOL_MAP: Dict[str, str] = {
    "train": "ASVspoof2024.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2024.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2024.LA.cm.eval.trl.txt",
}

_SPLIT_AUDIO_DIR_MAP: Dict[str, str] = {
    "train": "ASVspoof2024_LA_train",
    "dev":   "ASVspoof2024_LA_dev",
    "eval":  "ASVspoof2024_LA_eval",
}


def load_asvspoof_metadata(
    protocol_path: str | Path,
) -> List[UtteranceMeta]:
    """
    Parse an ASVspoof 2024 protocol file.

    Each line has the format:
        SPEAKER_ID UTT_ID - SYSTEM_ID LABEL

    Returns a list of UtteranceMeta objects sorted by utt_id.
    """
    path = Path(protocol_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    records: List[UtteranceMeta] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, _, system_id, label_str = parts[:5]
            label = 0 if label_str.lower() == "bonafide" else 1
            records.append(
                UtteranceMeta(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    system_id=system_id,
                    label=label,
                    label_str=label_str.lower(),
                )
            )
    return sorted(records, key=lambda r: r.utt_id)


# ── Dataset ────────────────────────────────────────────────────────────────

class ASVspoof2024Dataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2024 (LA track, countermeasure task).

    Parameters
    ----------
    root:
        Path to the ``LA/`` directory downloaded from the ASVspoof website.
    split:
        One of ``"train"``, ``"dev"``, or ``"eval"``.
    sample_rate:
        Target sample rate; audio is resampled if necessary (default 16 000).
    max_duration_s:
        Clips longer than this are trimmed (default 4.0 s).
    augment:
        If True, apply simple time-masking augmentation during training.

    Returns (per __getitem__)
    -------------------------
    waveform  : torch.Tensor, shape (1, T)   — float32 in [-1, 1]
    label     : int  — 0 = bonafide, 1 = spoof
    meta      : UtteranceMeta
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = 16_000,
        max_duration_s: float = 4.0,
        augment: bool = False,
    ) -> None:
        if split not in _SPLIT_PROTOCOL_MAP:
            raise ValueError(f"split must be one of {list(_SPLIT_PROTOCOL_MAP)}; got {split!r}")

        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)
        self.augment = augment and split == "train"

        # Protocol
        protocol_dir = self.root / "ASVspoof2024_LA_cm_protocols"
        protocol_file = protocol_dir / _SPLIT_PROTOCOL_MAP[split]
        self.records: List[UtteranceMeta] = load_asvspoof_metadata(protocol_file)

        # Audio directory
        self.audio_dir = self.root / _SPLIT_AUDIO_DIR_MAP[split] / "flac"

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, UtteranceMeta]:
        meta = self.records[idx]
        audio_path = self.audio_dir / f"{meta.utt_id}.flac"

        waveform, sr = torchaudio.load(str(audio_path))

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Trim / pad to fixed length
        T = waveform.shape[-1]
        if T > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif T < self.max_samples:
            pad = self.max_samples - T
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # Training augmentation: time masking
        if self.augment:
            waveform = _time_mask(waveform, max_mask_pct=0.15)

        return waveform, meta.label, meta

    # ------------------------------------------------------------------
    def label_distribution(self) -> Dict[str, int]:
        """Return count of bonafide vs spoof utterances."""
        counts: Dict[str, int] = {"bonafide": 0, "spoof": 0}
        for r in self.records:
            counts[r.label_str] += 1
        return counts

    def get_speaker_ids(self) -> List[str]:
        """Return sorted list of unique speaker IDs."""
        return sorted({r.speaker_id for r in self.records})


# ── Augmentation helpers ───────────────────────────────────────────────────

def _time_mask(
    waveform: torch.Tensor,
    max_mask_pct: float = 0.15,
) -> torch.Tensor:
    """Zero out a random contiguous segment of the waveform."""
    T = waveform.shape[-1]
    mask_len = int(T * max_mask_pct * torch.rand(1).item())
    if mask_len == 0:
        return waveform
    start = int((T - mask_len) * torch.rand(1).item())
    masked = waveform.clone()
    masked[:, start : start + mask_len] = 0.0
    return masked
