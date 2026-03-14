"""
VAJRA Voice AI — Spectrogram Model Training (Pretrained + Fine-Tuning).

Implements two-stage transfer learning:

Stage 1 — Feature Extraction:
    - Load pretrained EfficientNet-B0 (ImageNet weights)
    - Freeze backbone feature extractor
    - Train only the custom classifier head
    - Learning rate: 1e-4, epochs: 10

Stage 2 — Fine-Tuning:
    - Unfreeze last layers of EfficientNet-B0
    - Fine-tune entire network with discriminative learning rates
    - Backbone LR: 1e-5, Head LR: 1e-4, epochs: 5

Usage:
    python -m training.train_spectrogram --data-root /data/ASVspoof2024/LA
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.asvspoof import ASVspoof2024Dataset
from models.spectrogram_model import SpectrogramModel
from training.trainer import EpochMetrics, Trainer, TrainingConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# ── Stage configs ─────────────────────────────────────────────────────────

STAGE1_EPOCHS = 10
STAGE1_LR = 1e-4

STAGE2_EPOCHS = 5
STAGE2_LR_BACKBONE = 1e-5
STAGE2_LR_HEAD = 1e-4


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


def train_spectrogram(
    data_root: str,
    batch_size: int = 32,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage1_lr: float = STAGE1_LR,
    stage2_lr_backbone: float = STAGE2_LR_BACKBONE,
    stage2_lr_head: float = STAGE2_LR_HEAD,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 4,
    patience: int = 7,
    resume: Optional[str] = None,
) -> List[EpochMetrics]:
    """
    Train the spectrogram deepfake detector with two-stage transfer learning.

    Stage 1: Freeze backbone, train classifier head only.
    Stage 2: Unfreeze last backbone layers, fine-tune with small LR.

    Returns combined metrics from both stages.
    """
    # ── Build model ───────────────────────────────────────────────────────
    model = SpectrogramModel(num_classes=2, pretrained=True)

    config = TrainingConfig(
        batch_size=batch_size,
        epochs=stage1_epochs + stage2_epochs,
        learning_rate=stage1_lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = Trainer(model, config, model_name="spectrogram")

    # ── Resume if requested ───────────────────────────────────────────────
    start_epoch = 0
    if resume:
        dummy_opt = torch.optim.AdamW(model.parameters(), lr=stage1_lr)
        start_epoch = trainer.load_checkpoint(Path(resume), dummy_opt)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    all_metrics: List[EpochMetrics] = []

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Feature Extraction (frozen backbone)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STAGE 1: Feature extraction — frozen backbone, train classifier head")
    log.info("=" * 70)

    model.freeze_backbone()

    optimizer_s1 = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=stage1_lr,
        weight_decay=weight_decay,
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=stage1_epochs, eta_min=stage1_lr * 0.01
    )

    s1_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s1,
        scheduler=scheduler_s1,
        epochs=stage1_epochs,
        stage="stage1_feature_extraction",
        start_epoch=start_epoch,
    )
    all_metrics.extend(s1_metrics)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Fine-Tuning (unfreeze last layers)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STAGE 2: Fine-tuning — unfreeze last backbone layers")
    log.info("=" * 70)

    model.unfreeze_backbone(unfreeze_from=5)

    # Reset early stopping for stage 2
    trainer.epochs_no_improve = 0

    # Discriminative learning rates: backbone gets smaller LR
    param_groups = model.get_param_groups(
        lr_backbone=stage2_lr_backbone,
        lr_head=stage2_lr_head,
    )
    optimizer_s2 = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=stage2_epochs, eta_min=stage2_lr_backbone * 0.01
    )

    s2_start = (s1_metrics[-1].epoch + 1) if s1_metrics else start_epoch
    s2_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s2,
        scheduler=scheduler_s2,
        epochs=stage2_epochs,
        stage="stage2_fine_tuning",
        start_epoch=s2_start,
    )
    all_metrics.extend(s2_metrics)

    # ── Save history ──────────────────────────────────────────────────────
    trainer.save_history()

    log.info("Spectrogram training complete: %d total epochs", len(all_metrics))
    return all_metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train spectrogram deepfake detector (pretrained + fine-tuning)"
    )
    parser.add_argument("--data-root", required=True, help="Path to ASVspoof 2024 LA/ directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--stage1-lr", type=float, default=STAGE1_LR)
    parser.add_argument("--stage2-lr-backbone", type=float, default=STAGE2_LR_BACKBONE)
    parser.add_argument("--stage2-lr-head", type=float, default=STAGE2_LR_HEAD)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train_spectrogram(
        data_root=args.data_root,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr_backbone=args.stage2_lr_backbone,
        stage2_lr_head=args.stage2_lr_head,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
