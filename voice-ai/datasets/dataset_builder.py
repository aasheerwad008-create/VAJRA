"""
VAJRA Voice AI — Unified Dataset Builder.

Converts raw dataset directories into a unified ``real/`` + ``fake/``
structure suitable for training deepfake detection models.

Supported source formats:
    - ASVspoof 2019 / 2024 (protocol files + FLAC audio)
    - LibriSpeech (all utterances treated as real speech)
    - Raw directory of WAV/FLAC files with ``real/`` and ``fake/`` sub-dirs
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class DatasetBuilder:
    """
    Build a unified binary-label dataset from heterogeneous sources.

    The output directory will contain:
        ``<output_dir>/real/``  — bonafide speech files
        ``<output_dir>/fake/``  — spoofed / deepfake files

    Parameters
    ----------
    output_dir : str
        Target directory for the unified dataset.
    sample_rate : int
        Target sample rate (default 16 000 Hz).
    max_duration_s : float
        Maximum clip duration in seconds (default 4.0).
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        sample_rate: int = SAMPLE_RATE,
        max_duration_s: float = 4.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)

        for sub in ("real", "fake"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def build_from_asvspoof(self, root: str, version: str = "2024") -> Dict[str, int]:
        """
        Build from ASVspoof protocol files.

        Parameters
        ----------
        root : str
            Path to the ``LA/`` directory.
        version : str
            ``"2019"`` or ``"2024"`` (affects protocol file names).

        Returns
        -------
        Dict with counts of real and fake files processed.
        """
        root_path = Path(root)
        counts = {"real": 0, "fake": 0}

        prefix = f"ASVspoof{version}"
        protocol_dir = root_path / f"{prefix}_LA_cm_protocols"

        for split in ("train", "dev"):
            suffix = "trn" if split == "train" else "trl"
            protocol_file = protocol_dir / f"{prefix}.LA.cm.{split}.{suffix}.txt"
            if not protocol_file.exists():
                log.warning("Protocol file not found: %s", protocol_file)
                continue

            audio_dir_name = f"{prefix}_LA_{split}"
            audio_dir = root_path / audio_dir_name / "flac"

            with protocol_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _speaker, utt_id, _, _system, label_str = parts[:5]
                    is_real = label_str.lower() == "bonafide"
                    src = audio_dir / f"{utt_id}.flac"
                    if not src.exists():
                        continue

                    dest_sub = "real" if is_real else "fake"
                    dest = self.output_dir / dest_sub / f"{utt_id}.flac"
                    if not dest.exists():
                        shutil.copy2(str(src), str(dest))

                    key = "real" if is_real else "fake"
                    counts[key] += 1

        log.info("ASVspoof %s: %s", version, counts)
        return counts

    def build_from_librispeech(self, root: str) -> Dict[str, int]:
        """
        Import LibriSpeech utterances as real (bonafide) speech.

        Parameters
        ----------
        root : str
            Path to the LibriSpeech root (containing ``train-clean-100/``).

        Returns
        -------
        Dict with count of real files added.
        """
        root_path = Path(root)
        count = 0

        for flac_file in root_path.rglob("*.flac"):
            dest = self.output_dir / "real" / f"libri_{flac_file.stem}.flac"
            if not dest.exists():
                shutil.copy2(str(flac_file), str(dest))
            count += 1

        log.info("LibriSpeech: added %d real utterances", count)
        return {"real": count, "fake": 0}

    def build_from_directory(self, root: str) -> Dict[str, int]:
        """
        Import from a pre-organized ``real/`` + ``fake/`` directory.

        Parameters
        ----------
        root : str
            Directory containing ``real/`` and ``fake/`` sub-directories.
        """
        root_path = Path(root)
        counts = {"real": 0, "fake": 0}

        for label in ("real", "fake"):
            src_dir = root_path / label
            if not src_dir.exists():
                continue
            for audio_file in src_dir.glob("*"):
                if audio_file.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                    dest = self.output_dir / label / audio_file.name
                    if not dest.exists():
                        shutil.copy2(str(audio_file), str(dest))
                    counts[label] += 1

        log.info("Directory import: %s", counts)
        return counts

    def get_stats(self) -> Dict[str, int]:
        """Return counts of real and fake files in the output directory."""
        real_count = len(list((self.output_dir / "real").glob("*")))
        fake_count = len(list((self.output_dir / "fake").glob("*")))
        return {"real": real_count, "fake": fake_count}


class UnifiedAudioDataset(Dataset):
    """
    PyTorch Dataset that loads from the unified ``real/`` + ``fake/`` structure.

    Parameters
    ----------
    root : str
        Path to the unified dataset directory (containing ``real/`` and ``fake/``).
    sample_rate : int
        Target sample rate.
    max_duration_s : float
        Max clip duration in seconds.
    augment : bool
        Whether to apply augmentation (for training).
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = SAMPLE_RATE,
        max_duration_s: float = 4.0,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)
        self.augment = augment

        self.files: List[Tuple[Path, int]] = []
        for audio_file in sorted((self.root / "real").glob("*")):
            if audio_file.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                self.files.append((audio_file, 0))
        for audio_file in sorted((self.root / "fake").glob("*")):
            if audio_file.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                self.files.append((audio_file, 1))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.files[idx]
        waveform, sr = torchaudio.load(str(path))

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Trim / pad
        T = waveform.shape[-1]
        if T > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif T < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - T))

        return waveform.squeeze(0), label  # (T,), label

    def label_distribution(self) -> Dict[str, int]:
        """Return count of real vs fake files."""
        real = sum(1 for _, l in self.files if l == 0)
        fake = sum(1 for _, l in self.files if l == 1)
        return {"real": real, "fake": fake}
