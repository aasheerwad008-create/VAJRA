"""
VAJRA Voice AI — Core Training Engine.

Production-grade training loop with:
    - Mixed precision training (torch.cuda.amp)
    - Gradient clipping
    - Cosine annealing learning rate schedule
    - Early stopping
    - Checkpoint save/load/resume
    - Data loader prefetching with pinned memory
    - Automatic GPU detection with CPU fallback
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    patience: int = 7
    checkpoint_dir: str = "checkpoints"
    use_mixed_precision: bool = True
    seed: int = 42


@dataclass
class EpochMetrics:
    """Metrics collected for a single epoch."""

    epoch: int
    stage: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    elapsed_s: float
    extra: Dict[str, float] = field(default_factory=dict)


class Trainer:
    """
    Production-grade model training engine.

    Supports mixed precision, gradient clipping, early stopping,
    checkpoint management, and experiment logging.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : TrainingConfig
        Training hyperparameters.
    model_name : str
        Name prefix for checkpoints (e.g., ``"spectrogram"``).
    experiment_logger : optional
        An ExperimentLogger instance for tracking metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_name: str = "model",
        experiment_logger: Any = None,
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.logger = experiment_logger

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Training device: %s", self.device)

        self.model = model.to(self.device)

        # Mixed precision
        self.use_amp = config.use_mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            log.info("Mixed precision training enabled (torch.cuda.amp)")

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Checkpoint directory
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.history: List[EpochMetrics] = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epochs: Optional[int] = None,
        stage: str = "train",
        start_epoch: int = 0,
    ) -> List[EpochMetrics]:
        """
        Run the training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        criterion : loss function
        optimizer : optimizer
        scheduler : LR scheduler (optional)
        epochs : override for max epochs
        stage : label for this training stage (e.g., "stage1", "stage2")
        start_epoch : epoch number to start from (for resume)

        Returns
        -------
        List[EpochMetrics]
        """
        max_epochs = epochs or self.config.epochs
        stage_history: List[EpochMetrics] = []

        param_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log.info(
            "Starting training: stage=%s epochs=%d batch=%d lr=%.1e "
            "trainable_params=%s",
            stage,
            max_epochs,
            self.config.batch_size,
            optimizer.param_groups[0]["lr"],
            f"{param_count:,}",
        )

        for epoch in range(start_epoch, start_epoch + max_epochs):
            t0 = time.perf_counter()

            train_loss, train_acc = self._train_one_epoch(
                train_loader, criterion, optimizer
            )
            val_loss, val_acc = self._validate(val_loader, criterion)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.perf_counter() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            metrics = EpochMetrics(
                epoch=epoch,
                stage=stage,
                train_loss=round(train_loss, 5),
                train_acc=round(train_acc, 4),
                val_loss=round(val_loss, 5),
                val_acc=round(val_acc, 4),
                lr=current_lr,
                elapsed_s=round(elapsed, 1),
            )
            stage_history.append(metrics)
            self.history.append(metrics)

            log.info(
                "[%s] Epoch %02d — train_loss=%.4f train_acc=%.3f "
                "val_loss=%.4f val_acc=%.3f lr=%.2e [%.1fs]",
                stage,
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                elapsed,
            )

            # Log to experiment tracker
            if self.logger is not None:
                self.logger.log_epoch(metrics)

            # Best model checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(
                    optimizer, epoch, metrics,
                    self.ckpt_dir / f"{self.model_name}_best.pt",
                )
            else:
                self.epochs_no_improve += 1

            # Periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    optimizer, epoch, metrics,
                    self.ckpt_dir / f"{self.model_name}_epoch{epoch + 1:03d}.pt",
                )

            # Early stopping
            if self.epochs_no_improve >= self.config.patience:
                log.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    self.config.patience,
                )
                break

        # Final checkpoint
        if stage_history:
            self.save_checkpoint(
                optimizer,
                stage_history[-1].epoch,
                stage_history[-1],
                self.ckpt_dir / f"{self.model_name}_{stage}_final.pt",
            )

        return stage_history

    def _train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Run one training epoch with mixed precision and gradient clipping."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for waveforms, labels in loader:
            waveforms = waveforms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(waveforms)
                    loss = criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(waveforms)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for waveforms, labels in loader:
            waveforms = waveforms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(waveforms)
                    loss = criterion(logits, labels)
            else:
                logits = self.model(waveforms)
                loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ── Checkpoint management ─────────────────────────────────────────────

    def save_checkpoint(
        self,
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
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": (
                    self.scaler.state_dict() if self.scaler else None
                ),
                "metrics": asdict(metrics),
                "config": asdict(self.config),
                "best_val_loss": self.best_val_loss,
            },
            str(path),
        )
        log.info("Checkpoint saved → %s", path)

    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Load a checkpoint. Returns the epoch to resume from."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scaler and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed from checkpoint %s (epoch %d)", path, start_epoch - 1)
        return start_epoch

    def save_history(self, path: Optional[Path] = None) -> Path:
        """Save training history to JSON."""
        if path is None:
            path = self.ckpt_dir / f"{self.model_name}_history.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump([asdict(m) for m in self.history], fh, indent=2)
        log.info("Training history saved → %s", path)
        return path
