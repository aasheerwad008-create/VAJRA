"""
Unit tests for VAJRA Voice AI — datasets/dataset_builder.py

Covers:
    - DatasetBuilder creation and directory layout
    - build_from_directory (real/ + fake/ import)
    - UnifiedAudioDataset loading, trimming, padding
    - get_stats / label_distribution
    - Empty directory edge cases
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from datasets.dataset_builder import DatasetBuilder, UnifiedAudioDataset

SAMPLE_RATE = 16_000

_HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import AudioDecoder  # noqa: F401
except ImportError:
    _HAS_TORCHCODEC = False


def _write_wav(path: Path, duration_s: float = 1.0, sr: int = SAMPLE_RATE) -> None:
    """Write a short synthetic WAV file using soundfile."""
    n_samples = int(sr * duration_s)
    wav = np.random.randn(n_samples).astype(np.float32)
    sf.write(str(path), wav, sr)


# ═══════════════════════════════════════════════════════════════════════════
# DatasetBuilder
# ═══════════════════════════════════════════════════════════════════════════
class TestDatasetBuilder:

    def test_creates_output_dirs(self, tmp_path):
        output_dir = tmp_path / "processed"
        builder = DatasetBuilder(output_dir=str(output_dir))
        assert (output_dir / "real").is_dir()
        assert (output_dir / "fake").is_dir()

    def test_build_from_directory(self, tmp_path):
        # Prepare a source directory
        src = tmp_path / "source"
        (src / "real").mkdir(parents=True)
        (src / "fake").mkdir(parents=True)

        _write_wav(src / "real" / "r1.wav")
        _write_wav(src / "real" / "r2.wav")
        _write_wav(src / "fake" / "f1.wav")

        output_dir = tmp_path / "processed"
        builder = DatasetBuilder(output_dir=str(output_dir))
        counts = builder.build_from_directory(str(src))

        assert counts["real"] == 2
        assert counts["fake"] == 1
        assert (output_dir / "real" / "r1.wav").exists()
        assert (output_dir / "fake" / "f1.wav").exists()

    def test_get_stats(self, tmp_path):
        src = tmp_path / "source"
        (src / "real").mkdir(parents=True)
        (src / "fake").mkdir(parents=True)

        _write_wav(src / "real" / "a.wav")
        _write_wav(src / "fake" / "b.wav")
        _write_wav(src / "fake" / "c.wav")

        output_dir = tmp_path / "processed"
        builder = DatasetBuilder(output_dir=str(output_dir))
        builder.build_from_directory(str(src))

        stats = builder.get_stats()
        assert stats["real"] == 1
        assert stats["fake"] == 2

    def test_empty_source_directory(self, tmp_path):
        src = tmp_path / "empty_src"
        (src / "real").mkdir(parents=True)
        (src / "fake").mkdir(parents=True)

        output_dir = tmp_path / "processed"
        builder = DatasetBuilder(output_dir=str(output_dir))
        counts = builder.build_from_directory(str(src))
        assert counts["real"] == 0
        assert counts["fake"] == 0

    def test_skips_non_audio_files(self, tmp_path):
        src = tmp_path / "source"
        (src / "real").mkdir(parents=True)
        (src / "fake").mkdir(parents=True)

        _write_wav(src / "real" / "audio.wav")
        (src / "real" / "readme.txt").write_text("not audio")

        output_dir = tmp_path / "processed"
        builder = DatasetBuilder(output_dir=str(output_dir))
        counts = builder.build_from_directory(str(src))
        assert counts["real"] == 1  # Only the WAV file


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedAudioDataset
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available for torchaudio.load")
class TestUnifiedAudioDataset:

    @pytest.fixture()
    def dataset_dir(self, tmp_path):
        """Create a small unified dataset directory with synthetic audio."""
        (tmp_path / "real").mkdir()
        (tmp_path / "fake").mkdir()

        _write_wav(tmp_path / "real" / "r1.wav", duration_s=2.0)
        _write_wav(tmp_path / "real" / "r2.wav", duration_s=0.5)
        _write_wav(tmp_path / "fake" / "f1.wav", duration_s=3.0)

        return tmp_path

    def test_length(self, dataset_dir):
        ds = UnifiedAudioDataset(root=str(dataset_dir), max_duration_s=4.0)
        assert len(ds) == 3

    def test_item_shape(self, dataset_dir):
        ds = UnifiedAudioDataset(root=str(dataset_dir), max_duration_s=2.0)
        wav, label = ds[0]
        expected_samples = SAMPLE_RATE * 2
        assert wav.shape == (expected_samples,)
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_padding_short_audio(self, dataset_dir):
        ds = UnifiedAudioDataset(root=str(dataset_dir), max_duration_s=4.0)
        # r2.wav is 0.5 seconds — should be padded to 4.0 seconds
        wav, label = ds[1]  # sorted order: r1.wav, r2.wav
        expected = SAMPLE_RATE * 4
        assert wav.shape == (expected,)

    def test_trimming_long_audio(self, dataset_dir):
        ds = UnifiedAudioDataset(root=str(dataset_dir), max_duration_s=1.0)
        wav, _ = ds[0]
        expected = SAMPLE_RATE * 1
        assert wav.shape == (expected,)

    def test_label_distribution(self, dataset_dir):
        ds = UnifiedAudioDataset(root=str(dataset_dir))
        dist = ds.label_distribution()
        assert dist["real"] == 2
        assert dist["fake"] == 1

    def test_empty_dataset(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "fake").mkdir()
        ds = UnifiedAudioDataset(root=str(tmp_path))
        assert len(ds) == 0
