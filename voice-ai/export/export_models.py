"""
VAJRA Voice AI — Model Export Utility.

Exports trained models in both PyTorch (.pt) and ONNX (.onnx) formats
for production deployment.

Usage:
    python -m export.export_models \
        --checkpoint-dir checkpoints \
        --output-dir models
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = SAMPLE_RATE * 2  # 2-second window


class ModelExporter:
    """
    Export trained models to PyTorch and ONNX formats.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing training checkpoints.
    output_dir : str
        Directory for exported models.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "models",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, opset_version: int = 17) -> None:
        """Export all available models."""
        log.info("Exporting models from %s → %s", self.checkpoint_dir, self.output_dir)

        self.export_spectrogram_model(opset_version=opset_version)
        self.export_codec_model(opset_version=opset_version)
        self._write_manifest()

    def export_spectrogram_model(self, opset_version: int = 17) -> None:
        """Export the spectrogram classifier."""
        ckpt_path = self.checkpoint_dir / "spectrogram_best.pt"
        if not ckpt_path.exists():
            log.warning("Spectrogram checkpoint not found: %s", ckpt_path)
            return

        from models.spectrogram_model import SpectrogramModel

        model = SpectrogramModel(num_classes=2, pretrained=False)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Save PyTorch format
        pt_path = self.output_dir / "spectrogram_model.pt"
        torch.save(model.state_dict(), str(pt_path))
        log.info("Exported SpectrogramModel (PyTorch) → %s", pt_path)

        # Save ONNX format
        onnx_path = self.output_dir / "spectrogram_model.onnx"
        dummy = torch.zeros(1, CHUNK_SAMPLES, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=opset_version,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes={
                "waveform": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )
        log.info("Exported SpectrogramModel (ONNX) → %s", onnx_path)

    def export_codec_model(self, opset_version: int = 17) -> None:
        """Export the codec detector."""
        ckpt_path = self.checkpoint_dir / "codec_best.pt"
        if not ckpt_path.exists():
            log.warning("Codec checkpoint not found: %s", ckpt_path)
            return

        from models.codec_detector import CodecDetectorModel

        model = CodecDetectorModel(num_classes=3)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Save PyTorch format
        pt_path = self.output_dir / "codec_model.pt"
        torch.save(model.state_dict(), str(pt_path))
        log.info("Exported CodecDetectorModel (PyTorch) → %s", pt_path)

        # Save ONNX format
        onnx_path = self.output_dir / "codec_model.onnx"
        dummy = torch.zeros(1, CHUNK_SAMPLES, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=opset_version,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes={
                "waveform": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )
        log.info("Exported CodecDetectorModel (ONNX) → %s", onnx_path)

    def _write_manifest(self) -> None:
        """Write a JSON manifest listing exported model files."""
        files = sorted(self.output_dir.glob("*"))
        manifest = {
            "version": "2.0",
            "training_strategy": "pretrained_fine_tuning",
            "models": [
                {
                    "name": f.stem,
                    "file": f.name,
                    "format": f.suffix.lstrip("."),
                    "size_bytes": f.stat().st_size,
                }
                for f in files
                if f.suffix in (".pt", ".onnx")
            ],
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        log.info("Manifest written → %s", manifest_path)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export trained models")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--opset-version", type=int, default=17)
    args = parser.parse_args()

    exporter = ModelExporter(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )
    exporter.export_all(opset_version=args.opset_version)


if __name__ == "__main__":
    main()
