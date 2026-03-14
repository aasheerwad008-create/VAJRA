"""
VAJRA Voice AI — Model Evaluation Script.

Loads trained checkpoints and computes comprehensive metrics
on the validation/evaluation set.

Usage:
    python -m evaluation.evaluate_models \
        --checkpoint-dir checkpoints \
        --data-root /data/ASVspoof2024/LA
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import MetricsReport, compute_metrics

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def _collate_fn(batch):
    """Stack waveforms and labels, drop metadata."""
    waveforms = torch.stack([item[0].squeeze(0) for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> MetricsReport:
    """
    Evaluate a model and return comprehensive metrics.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    num_classes : int

    Returns
    -------
    MetricsReport
    """
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_scores: List[float] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        # Use class 1 probability as score for EER/ROC-AUC
        all_scores.extend(probs[:, 1].cpu().tolist())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    scores_arr = np.array(all_scores)

    return compute_metrics(labels_arr, preds_arr, scores_arr)


def evaluate_all_models(
    checkpoint_dir: str = "checkpoints",
    data_root: str = "",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, MetricsReport]:
    """
    Evaluate all available trained models.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing model checkpoints.
    data_root : str
        Path to ASVspoof 2024 LA/ directory.
    batch_size : int
    num_workers : int

    Returns
    -------
    Dict mapping model name to MetricsReport.
    """
    from data.asvspoof import ASVspoof2024Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir)
    results: Dict[str, MetricsReport] = {}

    # Build validation loader
    val_ds = ASVspoof2024Dataset(
        root=data_root,
        split="dev",
        sample_rate=SAMPLE_RATE,
        max_duration_s=4.0,
        augment=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    # ── Evaluate spectrogram model ────────────────────────────────────────
    spec_ckpt = ckpt_dir / "spectrogram_best.pt"
    if spec_ckpt.exists():
        log.info("Evaluating spectrogram model from %s", spec_ckpt)
        from models.spectrogram_model import SpectrogramModel

        model = SpectrogramModel(num_classes=2, pretrained=False)
        ckpt = torch.load(str(spec_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        report = evaluate_model(model, val_loader, device, num_classes=2)
        results["spectrogram"] = report
        log.info(
            "Spectrogram: acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f eer=%.4f",
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            report.roc_auc,
            report.eer,
        )
    else:
        log.warning("Spectrogram checkpoint not found: %s", spec_ckpt)

    # ── Evaluate codec model ──────────────────────────────────────────────
    codec_ckpt = ckpt_dir / "codec_best.pt"
    if codec_ckpt.exists():
        log.info("Evaluating codec model from %s", codec_ckpt)
        from models.codec_detector import CodecDetectorModel

        model = CodecDetectorModel(num_classes=3)
        ckpt = torch.load(str(codec_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        report = evaluate_model(model, val_loader, device, num_classes=3)
        results["codec"] = report
        log.info(
            "Codec: acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f",
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            report.roc_auc,
        )
    else:
        log.warning("Codec checkpoint not found: %s", codec_ckpt)

    # ── Save evaluation report ────────────────────────────────────────────
    report_path = ckpt_dir / "evaluation_report.json"
    report_data = {}
    for name, r in results.items():
        report_data[name] = {
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "roc_auc": r.roc_auc,
            "eer": r.eer,
            "num_samples": r.num_samples,
            "class_distribution": r.class_distribution,
        }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)
    log.info("Evaluation report saved → %s", report_path)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    evaluate_all_models(
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
