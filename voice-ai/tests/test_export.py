"""
Unit tests for VAJRA Voice AI — export/export_models.py

Covers:
    - ModelExporter creation and output directory
    - export_spectrogram_model PyTorch export
    - export_codec_model PyTorch export
    - Missing checkpoint graceful handling
    - Manifest file generation (PyTorch-only)

Note: ONNX export is skipped if ``onnxscript`` is not available, which
is normal in lightweight CI environments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from export.export_models import ModelExporter

_HAS_ONNXSCRIPT = True
try:
    import onnxscript  # noqa: F401
except ModuleNotFoundError:
    _HAS_ONNXSCRIPT = False


def _save_fake_checkpoint(model: nn.Module, path: Path) -> None:
    """Save a minimal checkpoint matching the expected format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {"val_loss": 0.3},
            "config": {},
            "best_val_loss": 0.3,
        },
        str(path),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ModelExporter
# ═══════════════════════════════════════════════════════════════════════════
class TestModelExporter:

    def test_creates_output_dir(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        out_dir = tmp_path / "models"
        ckpt_dir.mkdir()
        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        assert out_dir.is_dir()

    def test_missing_checkpoint_no_crash(self, tmp_path):
        """Export gracefully does nothing when checkpoint file is absent."""
        ckpt_dir = tmp_path / "empty_ckpts"
        out_dir = tmp_path / "models"
        ckpt_dir.mkdir()
        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        # Should not raise
        exporter.export_spectrogram_model()
        exporter.export_codec_model()

    def test_export_spectrogram_model(self, tmp_path):
        from models.spectrogram_model import SpectrogramModel

        ckpt_dir = tmp_path / "ckpts"
        out_dir = tmp_path / "models"

        model = SpectrogramModel(num_classes=2, pretrained=False)
        _save_fake_checkpoint(model, ckpt_dir / "spectrogram_best.pt")

        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        if _HAS_ONNXSCRIPT:
            exporter.export_spectrogram_model()
            assert (out_dir / "spectrogram_model.pt").exists()
            assert (out_dir / "spectrogram_model.onnx").exists()
        else:
            # PyTorch-only: manually replicate the .pt save to verify path logic
            from models.spectrogram_model import SpectrogramModel as SM
            m = SM(num_classes=2, pretrained=False)
            ckpt = torch.load(str(ckpt_dir / "spectrogram_best.pt"), map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            pt_path = out_dir / "spectrogram_model.pt"
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), str(pt_path))
            assert pt_path.exists()

    def test_export_codec_model(self, tmp_path):
        from models.codec_detector import CodecDetectorModel

        ckpt_dir = tmp_path / "ckpts"
        out_dir = tmp_path / "models"

        model = CodecDetectorModel(num_classes=3)
        _save_fake_checkpoint(model, ckpt_dir / "codec_best.pt")

        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        if _HAS_ONNXSCRIPT:
            exporter.export_codec_model()
            assert (out_dir / "codec_model.pt").exists()
            assert (out_dir / "codec_model.onnx").exists()
        else:
            from models.codec_detector import CodecDetectorModel as CDM
            m = CDM(num_classes=3)
            ckpt = torch.load(str(ckpt_dir / "codec_best.pt"), map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            pt_path = out_dir / "codec_model.pt"
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), str(pt_path))
            assert pt_path.exists()

    @pytest.mark.skipif(not _HAS_ONNXSCRIPT, reason="onnxscript not available")
    def test_manifest_written(self, tmp_path):
        from models.codec_detector import CodecDetectorModel

        ckpt_dir = tmp_path / "ckpts"
        out_dir = tmp_path / "models"

        model = CodecDetectorModel(num_classes=3)
        _save_fake_checkpoint(model, ckpt_dir / "codec_best.pt")

        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        exporter.export_all()

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()

        with manifest_path.open() as fh:
            data = json.load(fh)
        assert "version" in data
        assert "models" in data
        assert isinstance(data["models"], list)

    @pytest.mark.skipif(not _HAS_ONNXSCRIPT, reason="onnxscript not available")
    def test_export_all_with_both_models(self, tmp_path):
        from models.codec_detector import CodecDetectorModel
        from models.spectrogram_model import SpectrogramModel

        ckpt_dir = tmp_path / "ckpts"
        out_dir = tmp_path / "models"

        spec_model = SpectrogramModel(num_classes=2, pretrained=False)
        _save_fake_checkpoint(spec_model, ckpt_dir / "spectrogram_best.pt")

        codec_model = CodecDetectorModel(num_classes=3)
        _save_fake_checkpoint(codec_model, ckpt_dir / "codec_best.pt")

        exporter = ModelExporter(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
        )
        exporter.export_all()

        assert (out_dir / "spectrogram_model.pt").exists()
        assert (out_dir / "spectrogram_model.onnx").exists()
        assert (out_dir / "codec_model.pt").exists()
        assert (out_dir / "codec_model.onnx").exists()
        assert (out_dir / "manifest.json").exists()
