"""
VAJRA Voice AI — ONNX Model Export Utility.

Exports the trained ensemble models to ONNX format for:
  - Faster CPU inference in production
  - Cross-platform deployment (edge devices, browser via onnxruntime-web)
  - Model versioning and reproducibility

Usage:
    python -m models.export --output-dir /app/onnx_models
    # or from Python:
    from models.export import export_all_models
    export_all_models(output_dir="/app/onnx_models")
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = SAMPLE_RATE * 2  # 2-second window

# ── Individual export functions ────────────────────────────────────────────

def export_spectrogram_classifier(
    model: nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    """
    Export the EfficientNet-B0 mel-spectrogram deepfake classifier to ONNX.

    Input shape:  (batch, num_samples)  — float32 waveform at 16 kHz
    Output shape: (batch, 2)            — [p_real, p_fake] logits
    """
    model.eval()
    dummy = torch.zeros(1, CHUNK_SAMPLES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={
            "waveform": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
    )
    log.info("Exported SpectrogramClassifier → %s", output_path)


def export_codec_detector(
    model: nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    """
    Export the 1-D CNN codec artifact detector to ONNX.

    Input shape:  (batch, num_samples)
    Output shape: (batch, 3)  — [p_human, p_encodec, p_soundstream]
    """
    model.eval()
    dummy = torch.zeros(1, CHUNK_SAMPLES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={
            "waveform": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
    )
    log.info("Exported CodecArtifactDetector → %s", output_path)


def export_speaker_embedder(
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    """
    Export the ECAPA-TDNN speaker embedder to ONNX.

    Uses SpeechBrain's ONNX export functionality if available,
    otherwise falls back to a torch.onnx.export call.

    Input shape:  (batch, num_samples)
    Output shape: (batch, 192)  — L2-normalised speaker embedding
    """
    try:
        import speechbrain as sb
        from speechbrain.pretrained import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )

        # Wrap encoder in a thin nn.Module for export
        class EmbedderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, wavs: torch.Tensor) -> torch.Tensor:
                embs = self.encoder.encode_batch(wavs)
                return embs.squeeze(1)

        wrapper = EmbedderWrapper(classifier).eval()
        dummy = torch.zeros(1, CHUNK_SAMPLES, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            opset_version=opset_version,
            input_names=["waveform"],
            output_names=["embedding"],
            dynamic_axes={"waveform": {0: "batch"}, "embedding": {0: "batch"}},
        )
        log.info("Exported SpeakerEmbedder (ECAPA-TDNN) → %s", output_path)

    except Exception as exc:
        log.warning("Could not export SpeakerEmbedder: %s (skipping)", exc)


# ── Bulk export ────────────────────────────────────────────────────────────

def export_all_models(
    output_dir: str | Path = "/app/onnx_models",
    weights_dir: Optional[str | Path] = None,
) -> None:
    """
    Export all three ensemble models to ONNX in *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory where .onnx files will be written.
    weights_dir:
        Optional directory containing pre-trained PyTorch weight files
        (``spec_classifier.pt``, ``codec_detector.pt``).
        If None, random weights are used (for CI / shape validation only).
    """
    from models.ensemble import SpectrogramClassifier, CodecArtifactDetector

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Spectrogram Classifier ────────────────────────────────────────────
    spec_model = SpectrogramClassifier()
    if weights_dir:
        ckpt = Path(weights_dir) / "spec_classifier.pt"
        if ckpt.exists():
            spec_model.load_state_dict(
                torch.load(str(ckpt), map_location="cpu", weights_only=True)
            )
            log.info("Loaded SpectrogramClassifier weights from %s", ckpt)
    export_spectrogram_classifier(spec_model, out / "spec_classifier.onnx")

    # ── Codec Detector ────────────────────────────────────────────────────
    codec_model = CodecArtifactDetector()
    if weights_dir:
        ckpt = Path(weights_dir) / "codec_detector.pt"
        if ckpt.exists():
            codec_model.load_state_dict(
                torch.load(str(ckpt), map_location="cpu", weights_only=True)
            )
            log.info("Loaded CodecArtifactDetector weights from %s", ckpt)
    export_codec_detector(codec_model, out / "codec_detector.onnx")

    # ── Speaker Embedder ─────────────────────────────────────────────────
    export_speaker_embedder(out / "speaker_embedder.onnx")

    log.info("All models exported to %s", out)
    _write_manifest(out)


def _write_manifest(output_dir: Path) -> None:
    """Write a JSON manifest listing exported files with sizes."""
    import json

    files = sorted(output_dir.glob("*.onnx"))
    manifest = {
        "version": "1.0",
        "models": [
            {"name": f.stem, "file": f.name, "size_bytes": f.stat().st_size}
            for f in files
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest written to %s", manifest_path)


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Export VAJRA Voice AI ensemble models to ONNX"
    )
    parser.add_argument(
        "--output-dir",
        default="/app/onnx_models",
        help="Directory to write .onnx files (default: /app/onnx_models)",
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Directory with pre-trained .pt weight files (optional)",
    )
    args = parser.parse_args()
    export_all_models(output_dir=args.output_dir, weights_dir=args.weights_dir)


if __name__ == "__main__":
    _main()
