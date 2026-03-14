"""
VAJRA Voice AI — Codec Artifact Detector Training.

Trains the 1-D CNN codec artifact detector to classify:
    - HUMAN (genuine speech)
    - ENCODEC (Meta EnCodec artifacts)
    - SOUNDSTREAM (Google SoundStream artifacts)

Uses cross-entropy loss with mixed precision and gradient clipping.

Usage:
    python -m training.train_codec --data-root /data/ASVspoof2024/LA
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.asvspoof import ASVspoof2024Dataset
from models.codec_detector import CodecDetectorModel
from training.trainer import EpochMetrics, Trainer, TrainingConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-3


def _collate_fn(batch):
    """Stack waveforms and labels, drop metadata."""
    waveforms = torch.stack([item[0].squeeze(0) for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_duration_s: float = 4.0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from ASVspoof 2024."""
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

    log.info("Train set: %d samples — %s", len(train_ds), train_ds.label_distribution())
    log.info("Val set:   %d samples — %s", len(val_ds), val_ds.label_distribution())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def train_codec(
    data_root: str,
    batch_size: int = 32,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 4,
    patience: int = 7,
    resume: Optional[str] = None,
) -> List[EpochMetrics]:
    """
    Train the codec artifact detector.

    Returns per-epoch metrics.
    """
    # ── Build model ───────────────────────────────────────────────────────
    model = CodecDetectorModel(num_classes=3)

    config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = Trainer(model, config, model_name="codec")

    # ── Resume if requested ───────────────────────────────────────────────
    start_epoch = 0
    if resume:
        dummy_opt = torch.optim.AdamW(model.parameters(), lr=lr)
        start_epoch = trainer.load_checkpoint(Path(resume), dummy_opt)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── Train ─────────────────────────────────────────────────────────────
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        stage="codec_training",
        start_epoch=start_epoch,
    )

    trainer.save_history()

    log.info("Codec training complete: %d epochs", len(metrics))
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train codec artifact detector"
    )
    parser.add_argument("--data-root", required=True, help="Path to ASVspoof 2024 LA/ directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train_codec(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
