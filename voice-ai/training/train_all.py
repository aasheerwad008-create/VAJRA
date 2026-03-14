"""
VAJRA Voice AI — Master Training Script.

Executes the complete ML pipeline:
    1. Download/prepare datasets
    2. Preprocess audio
    3. Train spectrogram model (pretrained + fine-tuning)
    4. Train codec detector
    5. Evaluate models
    6. Export models (PyTorch + ONNX)

Usage:
    python -m training.train_all --data-root /data/ASVspoof2024/LA

    # Skip evaluation/export (training only):
    python -m training.train_all --data-root /data/ASVspoof2024/LA --train-only
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def run_full_pipeline(
    data_root: str,
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "models",
    num_workers: int = 4,
    train_only: bool = False,
    skip_spectrogram: bool = False,
    skip_codec: bool = False,
) -> None:
    """
    Run the complete ML training pipeline.

    Parameters
    ----------
    data_root : str
        Path to the ASVspoof 2024 ``LA/`` directory.
    batch_size : int
        Mini-batch size for training.
    checkpoint_dir : str
        Directory for model checkpoints.
    output_dir : str
        Directory for exported models.
    num_workers : int
        DataLoader worker processes.
    train_only : bool
        If True, skip evaluation and export.
    skip_spectrogram : bool
        If True, skip spectrogram model training.
    skip_codec : bool
        If True, skip codec model training.
    """
    pipeline_start = time.perf_counter()
    log.info("=" * 70)
    log.info("VAJRA/KAVACHA Voice AI — Full Training Pipeline")
    log.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — Dataset Preparation
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n[1/6] Dataset preparation")
    log.info("Data root: %s", data_root)
    root = Path(data_root)
    if not root.exists():
        log.error("Data root does not exist: %s", root)
        raise FileNotFoundError(f"Data root not found: {root}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — Audio Preprocessing
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n[2/6] Audio preprocessing (handled by dataset loaders)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — Train Spectrogram Model (Pretrained + Fine-Tuning)
    # ══════════════════════════════════════════════════════════════════════
    spec_metrics = None
    if not skip_spectrogram:
        log.info("\n[3/6] Training spectrogram model (pretrained + fine-tuning)")
        from training.train_spectrogram import train_spectrogram

        spec_metrics = train_spectrogram(
            data_root=data_root,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            num_workers=num_workers,
        )
        if spec_metrics:
            best = max(spec_metrics, key=lambda m: m.val_acc)
            log.info(
                "Spectrogram best: val_acc=%.3f val_loss=%.4f (epoch %d)",
                best.val_acc,
                best.val_loss,
                best.epoch + 1,
            )
    else:
        log.info("\n[3/6] Skipping spectrogram model training")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — Train Codec Detector
    # ══════════════════════════════════════════════════════════════════════
    codec_metrics = None
    if not skip_codec:
        log.info("\n[4/6] Training codec detector")
        from training.train_codec import train_codec

        codec_metrics = train_codec(
            data_root=data_root,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            num_workers=num_workers,
        )
        if codec_metrics:
            best = max(codec_metrics, key=lambda m: m.val_acc)
            log.info(
                "Codec best: val_acc=%.3f val_loss=%.4f (epoch %d)",
                best.val_acc,
                best.val_loss,
                best.epoch + 1,
            )
    else:
        log.info("\n[4/6] Skipping codec model training")

    if train_only:
        elapsed = time.perf_counter() - pipeline_start
        log.info("\nTraining-only pipeline complete in %.1f seconds", elapsed)
        return

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — Evaluate Models
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n[5/6] Model evaluation")
    try:
        from evaluation.evaluate_models import evaluate_all_models

        evaluate_all_models(
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
        )
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 — Export Models
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n[6/6] Model export")
    try:
        from export.export_models import ModelExporter

        exporter = ModelExporter(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )
        exporter.export_all()
    except Exception as exc:
        log.warning("Export failed: %s", exc)

    elapsed = time.perf_counter() - pipeline_start
    log.info("\n" + "=" * 70)
    log.info("Full pipeline complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    log.info("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="VAJRA Voice AI — Master Training Pipeline"
    )
    parser.add_argument("--data-root", required=True, help="Path to ASVspoof 2024 LA/ directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-only", action="store_true", help="Skip evaluation and export")
    parser.add_argument("--skip-spectrogram", action="store_true")
    parser.add_argument("--skip-codec", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(
        data_root=args.data_root,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        train_only=args.train_only,
        skip_spectrogram=args.skip_spectrogram,
        skip_codec=args.skip_codec,
    )


if __name__ == "__main__":
    main()
