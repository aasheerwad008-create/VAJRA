"""
Unit tests for VAJRA Voice AI — pretrained weight management.

Covers:
    - WeightRegistry singleton and entry listing
    - WeightEntry data fields
    - Registry filtering (by model, source_type)
    - verify_weights for init-type and missing files
    - download_weights dispatch for init-type
    - setup_all with init-only entries
    - status() reporting
    - Timm download (with mocked timm)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from pretrained.registry import WeightEntry, WeightRegistry
from pretrained.downloader import download_weights, setup_all, status, verify_weights


# ═══════════════════════════════════════════════════════════════════════════
# WeightRegistry
# ═══════════════════════════════════════════════════════════════════════════
class TestWeightRegistry:

    def test_singleton(self):
        r1 = WeightRegistry()
        r2 = WeightRegistry()
        assert r1 is r2

    def test_list_all_non_empty(self):
        reg = WeightRegistry()
        entries = reg.list_all()
        assert len(entries) >= 5

    def test_names_returns_strings(self):
        reg = WeightRegistry()
        names = reg.names()
        assert all(isinstance(n, str) for n in names)
        assert "efficientnet_b0_imagenet" in names
        assert "ecapa_tdnn_voxceleb" in names
        assert "rawnet2_init" in names
        assert "codec_detector_init" in names

    def test_get_existing(self):
        reg = WeightRegistry()
        entry = reg.get("efficientnet_b0_imagenet")
        assert isinstance(entry, WeightEntry)
        assert entry.model == "SpectrogramModel"

    def test_get_missing_raises(self):
        reg = WeightRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent_model")

    def test_filter_by_model(self):
        reg = WeightRegistry()
        entries = reg.filter_by_model("SpectrogramModel")
        assert len(entries) >= 1
        assert all(e.model == "SpectrogramModel" for e in entries)

    def test_filter_by_source_type_init(self):
        reg = WeightRegistry()
        inits = reg.filter_by_source_type("init")
        assert len(inits) >= 2
        assert all(e.source_type == "init" for e in inits)

    def test_filter_by_source_type_timm(self):
        reg = WeightRegistry()
        timm_entries = reg.filter_by_source_type("timm")
        assert len(timm_entries) >= 1
        assert all(e.source_type == "timm" for e in timm_entries)

    def test_model_dir_is_path(self):
        reg = WeightRegistry()
        assert isinstance(reg.model_dir, Path)


# ═══════════════════════════════════════════════════════════════════════════
# WeightEntry
# ═══════════════════════════════════════════════════════════════════════════
class TestWeightEntry:

    def test_entry_fields(self):
        reg = WeightRegistry()
        entry = reg.get("efficientnet_b0_imagenet")
        assert entry.name == "efficientnet_b0_imagenet"
        assert entry.source_type == "timm"
        assert entry.source == "efficientnet_b0"
        assert entry.trainable is True
        assert "backbone" in entry.tags

    def test_entry_frozen(self):
        """WeightEntry is a frozen dataclass — immutable."""
        reg = WeightRegistry()
        entry = reg.get("rawnet2_init")
        with pytest.raises(AttributeError):
            entry.name = "something_else"  # type: ignore[misc]

    def test_ecapa_entry_not_trainable(self):
        reg = WeightRegistry()
        entry = reg.get("ecapa_tdnn_voxceleb")
        assert entry.trainable is False

    def test_init_entries_have_empty_filename(self):
        reg = WeightRegistry()
        for entry in reg.filter_by_source_type("init"):
            assert entry.filename == ""

    def test_non_init_entries_have_filename(self):
        reg = WeightRegistry()
        for entry in reg.list_all():
            if entry.source_type != "init":
                assert entry.filename != ""


# ═══════════════════════════════════════════════════════════════════════════
# verify_weights
# ═══════════════════════════════════════════════════════════════════════════
class TestVerifyWeights:

    def test_init_entry_always_passes(self):
        reg = WeightRegistry()
        entry = reg.get("rawnet2_init")
        assert verify_weights(entry) is True

    def test_missing_file_returns_false(self, tmp_path):
        entry = WeightEntry(
            name="test_missing",
            model="TestModel",
            source="http://example.com/weights.pt",
            source_type="url",
            description="Test",
            filename="missing_file.pt",
        )
        assert verify_weights(entry, model_dir=tmp_path) is False

    def test_existing_file_no_checksum(self, tmp_path):
        (tmp_path / "test.pt").write_bytes(b"fake weights")
        entry = WeightEntry(
            name="test_exists",
            model="TestModel",
            source="test",
            source_type="url",
            description="Test",
            filename="test.pt",
            sha256="",
        )
        assert verify_weights(entry, model_dir=tmp_path) is True

    def test_existing_file_wrong_checksum(self, tmp_path):
        (tmp_path / "test.pt").write_bytes(b"fake weights")
        entry = WeightEntry(
            name="test_bad_checksum",
            model="TestModel",
            source="test",
            source_type="url",
            description="Test",
            filename="test.pt",
            sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert verify_weights(entry, model_dir=tmp_path) is False

    def test_existing_file_correct_checksum(self, tmp_path):
        import hashlib
        data = b"test data for checksum"
        (tmp_path / "test.pt").write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        entry = WeightEntry(
            name="test_good_checksum",
            model="TestModel",
            source="test",
            source_type="url",
            description="Test",
            filename="test.pt",
            sha256=expected,
        )
        assert verify_weights(entry, model_dir=tmp_path) is True


# ═══════════════════════════════════════════════════════════════════════════
# download_weights
# ═══════════════════════════════════════════════════════════════════════════
class TestDownloadWeights:

    def test_init_entry_returns_base_dir(self, tmp_path):
        entry = WeightEntry(
            name="test_init",
            model="TestModel",
            source="kaiming_init",
            source_type="init",
            description="Test",
            filename="",
        )
        result = download_weights(entry, model_dir=tmp_path)
        assert result == tmp_path

    def test_cached_entry_skips_download(self, tmp_path):
        (tmp_path / "cached.pt").write_bytes(b"cached weights")
        entry = WeightEntry(
            name="test_cached",
            model="TestModel",
            source="test",
            source_type="url",
            description="Test",
            filename="cached.pt",
        )
        result = download_weights(entry, model_dir=tmp_path)
        assert result == tmp_path / "cached.pt"

    def test_unknown_source_type_returns_base(self, tmp_path):
        entry = WeightEntry(
            name="test_unknown",
            model="TestModel",
            source="test",
            source_type="unknown_provider",
            description="Test",
            filename="missing.pt",
        )
        result = download_weights(entry, model_dir=tmp_path)
        assert result == tmp_path

    def test_creates_model_dir(self, tmp_path):
        target = tmp_path / "subdir" / "weights"
        entry = WeightEntry(
            name="test_mkdir",
            model="TestModel",
            source="kaiming_init",
            source_type="init",
            description="Test",
            filename="",
        )
        download_weights(entry, model_dir=target)
        assert target.exists()

    def test_timm_download_mocked(self, tmp_path):
        """Test timm download path with mocked timm library."""
        import torch

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": torch.zeros(1)}

        entry = WeightEntry(
            name="test_timm",
            model="TestModel",
            source="efficientnet_b0",
            source_type="timm",
            description="Test timm download",
            filename="timm/test_model.pth",
        )

        with patch("pretrained.downloader.verify_weights", return_value=False):
            with patch("timm.create_model", return_value=mock_model):
                result = download_weights(entry, model_dir=tmp_path, force=True)

        assert result == tmp_path / "timm" / "test_model.pth"
        assert result.exists()


# ═══════════════════════════════════════════════════════════════════════════
# setup_all
# ═══════════════════════════════════════════════════════════════════════════
class TestSetupAll:

    def test_setup_returns_dict(self, tmp_path):
        """setup_all returns a dict mapping names to success booleans.

        Init-type entries always succeed; network-dependent ones may fail in
        CI but should not raise when skip_errors=True.
        """
        results = setup_all(model_dir=tmp_path, skip_errors=True)
        assert isinstance(results, dict)
        assert len(results) >= 5

        # Init-type entries should always succeed
        assert results["rawnet2_init"] is True
        assert results["codec_detector_init"] is True

    def test_setup_creates_model_dir(self, tmp_path):
        target = tmp_path / "my_weights"
        setup_all(model_dir=target, skip_errors=True)
        assert target.exists()


# ═══════════════════════════════════════════════════════════════════════════
# status
# ═══════════════════════════════════════════════════════════════════════════
class TestStatus:

    def test_status_returns_list(self, tmp_path):
        rows = status(model_dir=tmp_path)
        assert isinstance(rows, list)
        assert len(rows) >= 5

    def test_status_has_expected_keys(self, tmp_path):
        rows = status(model_dir=tmp_path)
        for row in rows:
            assert "name" in row
            assert "model" in row
            assert "source_type" in row
            assert "cached" in row
            assert "path" in row

    def test_init_entries_are_cached(self, tmp_path):
        rows = status(model_dir=tmp_path)
        for row in rows:
            if row["source_type"] == "init":
                assert row["cached"] is True
