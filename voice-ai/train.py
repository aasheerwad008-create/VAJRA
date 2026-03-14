"""
VAJRA Voice AI — Model Training Pipeline.

Trains the SpectrogramClassifier and CodecArtifactDetector for
deepfake speech detection.

This module supports two training strategies:

1. **Pretrained + Fine-Tuning** (default, recommended):
   Uses ImageNet-pretrained EfficientNet-B0 with two-stage transfer learning.
   Stage 1: Freeze backbone, train classifier head (10 epochs, lr=1e-4).
   Stage 2: Unfreeze last layers, fine-tune all (5 epochs, lr=1e-5).

2. **Legacy from-scratch** training:
   Train all parameters from random initialization (for benchmarking).

Usage:
    # Pretrained + fine-tuning (recommended)
    python train.py --model spectrogram --data-root /data/ASVspoof2024/LA

    # Train codec detector
    python train.py --model codec --data-root /data/ASVspoof2024/LA

    # Train both models
    python train.py --model all --data-root /data/ASVspoof2024/LA

    # Full pipeline (train + evaluate + export)
    python train.py --model all --data-root /data/ASVspoof2024/LA --full-pipeline

    # Legacy from-scratch mode
    python train.py --model spectrogram --data-root /data/ASVspoof2024/LA --no-pretrained
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.asvspoof import ASVspoof2024Dataset
from models.ensemble import CodecArtifactDetector, SpectrogramClassifier

log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_CHECKPOINT_DIR = "checkpoints"


# ── Metrics ────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Metrics collected for a single epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_eer: float
    lr: float
    elapsed_s: float


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER) from binary labels and prediction scores.

    Parameters
    ----------
    labels : array of int, shape (N,)
        Ground truth labels (0 = bonafide, 1 = spoof).
    scores : array of float, shape (N,)
        Model scores (higher → more likely spoof).

    Returns
    -------
    float
        The EER value in [0, 1].
    """
    if len(labels) == 0:
        return 0.0

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Compute FAR and FRR at every unique threshold
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]

    tp = 0
    prev_far, prev_frr = 0.0, 1.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        fp = (i + 1) - tp
        fn = n_pos - tp

        far = fp / n_neg       # false acceptance rate
        frr = fn / n_pos       # false rejection rate

        # Check if FAR has crossed FRR (was below, now at or above)
        if far >= frr:
            # Linear interpolation to find the crossing point
            denom = (far - prev_far) + (prev_frr - frr)
            if denom > 0:
                alpha = (prev_frr - prev_far) / denom
                eer = prev_far + alpha * (far - prev_far)
            else:
                eer = far
            return float(np.clip(eer, 0.0, 1.0))

        prev_far, prev_frr = far, frr

    # If we never crossed, return the last FAR (edge case)
    return float(np.clip(prev_far, 0.0, 1.0))


# ── Data helpers ───────────────────────────────────────────────────────────

def _collate_fn(
    batch: List[Tuple[torch.Tensor, int, object]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate: stack waveforms and labels, drop metadata.

    The dataset returns ``(waveform, label, meta)`` where waveform is
    shape ``(1, T)``.  Models expect ``(B, T)`` so we squeeze channel dim.
    """
    waveforms = torch.stack([item[0].squeeze(0) for item in batch])  # (B, T)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


def build_dataloaders(
    data_root: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    max_duration_s: float = 4.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from ASVspoof 2024.

    Parameters
    ----------
    data_root : str
        Path to the ``LA/`` directory of the ASVspoof 2024 corpus.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    max_duration_s : float
        Maximum audio clip duration in seconds.

    Returns
    -------
    (train_loader, val_loader)
    """
    train_ds = ASVspoof2024Dataset(
        root=data_root,
        split="train",
        sample_rate=SAMPLE_RATE,
        max_duration_s=max_duration_s,
        augment=True,
    )
    val_ds = ASVspoof2024Dataset(
        root=data_root,
        split="dev",
        sample_rate=SAMPLE_RATE,
        max_duration_s=max_duration_s,
        augment=False,
    )

    log.info(
        "Train set: %d samples — %s", len(train_ds), train_ds.label_distribution()
    )
    log.info("Val set:   %d samples — %s", len(val_ds), val_ds.label_distribution())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader


# ── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Run one training epoch with mixed precision and gradient clipping.

    Returns
    -------
    (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(waveforms)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate on a validation set.

    Returns
    -------
    (avg_loss, accuracy, eer)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[int] = []
    all_scores: List[float] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Collect scores for EER — use spoof probability (class 1)
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(probs[:, 1].cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    return avg_loss, accuracy, eer


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "train_loss": metrics.train_loss,
                "train_acc": metrics.train_acc,
                "val_loss": metrics.val_loss,
                "val_acc": metrics.val_acc,
                "val_eer": metrics.val_eer,
            },
        },
        str(path),
    )
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Load a training checkpoint. Returns the epoch number to resume from.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    log.info("Resumed from checkpoint %s (epoch %d)", path, start_epoch - 1)
    return start_epoch


# ── Main training driver ──────────────────────────────────────────────────

def train_model(
    model_name: str,
    data_root: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    resume: Optional[str] = None,
    num_workers: int = 4,
    patience: int = 7,
) -> List[EpochMetrics]:
    """
    Train a single model (legacy from-scratch mode) and return per-epoch metrics.

    Parameters
    ----------
    model_name : str
        ``"spectrogram"`` for SpectrogramClassifier or
        ``"codec"`` for CodecArtifactDetector.
    data_root : str
        Path to the ``LA/`` directory of the ASVspoof 2024 corpus.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Peak learning rate for AdamW.
    weight_decay : float
        AdamW weight decay.
    checkpoint_dir : str
        Directory for saving checkpoints.
    resume : str or None
        Path to a checkpoint to resume training from.
    num_workers : int
        DataLoader worker processes.
    patience : int
        Early-stopping patience (epochs without val_loss improvement).

    Returns
    -------
    List of EpochMetrics, one per completed epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Build model ───────────────────────────────────────────────────
    if model_name == "spectrogram":
        model: nn.Module = SpectrogramClassifier()
        prefix = "spec"
    elif model_name == "codec":
        model = CodecArtifactDetector()
        prefix = "codec"
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s — %s trainable parameters", model_name, f"{param_count:,}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    # ── Mixed precision ───────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(Path(resume), model, optimizer)

    # ── Training ──────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    log.info(
        "Starting training: model=%s epochs=%d batch=%d lr=%.1e",
        model_name,
        epochs,
        batch_size,
        lr,
    )

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip_norm=1.0,
        )
        val_loss, val_acc, val_eer = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=round(train_loss, 5),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 5),
            val_acc=round(val_acc, 4),
            val_eer=round(val_eer, 4),
            lr=current_lr,
            elapsed_s=round(elapsed, 1),
        )
        history.append(metrics)

        log.info(
            "Epoch %02d/%02d — train_loss=%.4f train_acc=%.3f "
            "val_loss=%.4f val_acc=%.3f val_eer=%.4f lr=%.2e [%.1fs]",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_eer,
            current_lr,
            elapsed,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch, metrics,
                ckpt_dir / f"{prefix}_best.pt",
            )
        else:
            epochs_no_improve += 1

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                ckpt_dir / f"{prefix}_epoch{epoch + 1:03d}.pt",
            )

        # Early stopping
        if epochs_no_improve >= patience:
            log.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1,
                patience,
            )
            break

    # Save final checkpoint
    if history:
        save_checkpoint(
            model, optimizer, history[-1].epoch, history[-1],
            ckpt_dir / f"{prefix}_final.pt",
        )

    # Write training history
    history_path = ckpt_dir / f"{prefix}_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_acc": m.train_acc,
                    "val_loss": m.val_loss,
                    "val_acc": m.val_acc,
                    "val_eer": m.val_eer,
                    "lr": m.lr,
                    "elapsed_s": m.elapsed_s,
                }
                for m in history
            ],
            fh,
            indent=2,
        )
    log.info("Training history saved → %s", history_path)

    return history


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train VAJRA Voice AI deepfake detection models"
    )
    parser.add_argument(
        "--model",
        choices=["spectrogram", "codec", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the ASVspoof 2024 LA/ directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Max training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint output directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience in epochs (default: 7)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use from-scratch training (legacy mode, no pretrained weights)",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: train → evaluate → export",
    )
    args = parser.parse_args()

    # ── Pretrained + fine-tuning (new default) ────────────────────────
    if not args.no_pretrained and not args.full_pipeline:
        log.info("Using pretrained + fine-tuning training strategy")
        log.info("(Use --no-pretrained for legacy from-scratch mode)")

        if args.model in ("spectrogram", "all"):
            from training.train_spectrogram import train_spectrogram

            log.info("=" * 60)
            log.info("Training: spectrogram (pretrained + fine-tuning)")
            log.info("=" * 60)
            train_spectrogram(
                data_root=args.data_root,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                num_workers=args.num_workers,
                patience=args.patience,
                resume=args.resume if args.model == "spectrogram" else None,
            )

        if args.model in ("codec", "all"):
            from training.train_codec import train_codec

            log.info("=" * 60)
            log.info("Training: codec")
            log.info("=" * 60)
            train_codec(
                data_root=args.data_root,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                num_workers=args.num_workers,
                patience=args.patience,
                resume=args.resume if args.model == "codec" else None,
            )
        return

    # ── Full pipeline mode ────────────────────────────────────────────
    if args.full_pipeline:
        from training.train_all import run_full_pipeline

        run_full_pipeline(
            data_root=args.data_root,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            num_workers=args.num_workers,
        )
        return

    # ── Legacy from-scratch mode ──────────────────────────────────────
    models_to_train = (
        ["spectrogram", "codec"] if args.model == "all" else [args.model]
    )

    for name in models_to_train:
        log.info("=" * 60)
        log.info("Training: %s (from-scratch)", name)
        log.info("=" * 60)
        train_model(
            model_name=name,
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume if len(models_to_train) == 1 else None,
            num_workers=args.num_workers,
            patience=args.patience,
        )


if __name__ == "__main__":
    _main()
