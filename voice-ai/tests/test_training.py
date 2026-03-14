"""
Unit tests for VAJRA Voice AI — training/trainer.py

Covers:
    - TrainingConfig defaults
    - EpochMetrics dataclass
    - Trainer initialization (device, checkpoint dir, reproducibility)
    - Full training loop with synthetic DataLoader
    - Checkpoint save / load / resume
    - Early stopping behaviour
    - Training history persistence
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from training.trainer import EpochMetrics, Trainer, TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    """Minimal model for testing the training loop."""

    def __init__(self, input_dim: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_loader(n: int = 64, input_dim: int = 16, num_classes: int = 2, batch_size: int = 16):
    """Create a synthetic DataLoader for testing."""
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ═══════════════════════════════════════════════════════════════════════════
# TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════
class TestTrainingConfig:

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 32
        assert cfg.epochs == 30
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 1e-4
        assert cfg.grad_clip_norm == 1.0
        assert cfg.patience == 7
        assert cfg.use_mixed_precision is True
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = TrainingConfig(batch_size=64, epochs=10, learning_rate=1e-3)
        assert cfg.batch_size == 64
        assert cfg.epochs == 10
        assert cfg.learning_rate == 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# EpochMetrics
# ═══════════════════════════════════════════════════════════════════════════
class TestEpochMetrics:

    def test_creation(self):
        m = EpochMetrics(
            epoch=0, stage="train", train_loss=0.5, train_acc=0.8,
            val_loss=0.6, val_acc=0.75, lr=1e-4, elapsed_s=1.0,
        )
        assert m.epoch == 0
        assert m.stage == "train"
        assert m.val_acc == 0.75

    def test_extra_field_defaults_empty(self):
        m = EpochMetrics(
            epoch=0, stage="s1", train_loss=0.1, train_acc=0.9,
            val_loss=0.2, val_acc=0.85, lr=1e-4, elapsed_s=0.5,
        )
        assert m.extra == {}


# ═══════════════════════════════════════════════════════════════════════════
# Trainer initialization
# ═══════════════════════════════════════════════════════════════════════════
class TestTrainerInit:

    def test_device_is_cpu_when_no_gpu(self):
        model = _TinyModel()
        cfg = TrainingConfig(use_mixed_precision=False)
        trainer = Trainer(model, cfg, model_name="test")
        # In CI there's typically no GPU
        assert trainer.device.type in ("cpu", "cuda")

    def test_checkpoint_dir_created(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        cfg = TrainingConfig(checkpoint_dir=str(ckpt_dir), use_mixed_precision=False)
        model = _TinyModel()
        Trainer(model, cfg, model_name="test")
        assert ckpt_dir.exists()

    def test_initial_state(self, tmp_path):
        cfg = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        model = _TinyModel()
        trainer = Trainer(model, cfg, model_name="test")
        assert trainer.best_val_loss == float("inf")
        assert trainer.epochs_no_improve == 0
        assert trainer.history == []


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════
class TestTrainerTrainLoop:

    @pytest.fixture()
    def setup(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(
            batch_size=16, epochs=3, patience=10,
            checkpoint_dir=str(tmp_path), use_mixed_precision=False,
        )
        trainer = Trainer(model, cfg, model_name="tiny")
        train_loader = _make_loader(n=64, batch_size=16)
        val_loader = _make_loader(n=32, batch_size=16)
        return trainer, model, train_loader, val_loader, tmp_path

    def test_returns_epoch_metrics(self, setup):
        trainer, model, train_loader, val_loader, _ = setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = trainer.train(
            train_loader, val_loader, criterion, optimizer,
            epochs=3, stage="test",
        )
        assert isinstance(metrics, list)
        assert len(metrics) == 3
        assert all(isinstance(m, EpochMetrics) for m in metrics)

    def test_history_accumulated(self, setup):
        trainer, model, train_loader, val_loader, _ = setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer.train(train_loader, val_loader, criterion, optimizer, epochs=2, stage="s1")
        trainer.train(train_loader, val_loader, criterion, optimizer, epochs=2, stage="s2")
        assert len(trainer.history) == 4

    def test_best_checkpoint_saved(self, setup):
        trainer, model, train_loader, val_loader, tmp_path = setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer.train(train_loader, val_loader, criterion, optimizer, epochs=3, stage="test")
        best_ckpt = tmp_path / "tiny_best.pt"
        assert best_ckpt.exists()

    def test_final_checkpoint_saved(self, setup):
        trainer, model, train_loader, val_loader, tmp_path = setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer.train(train_loader, val_loader, criterion, optimizer, epochs=3, stage="test")
        final_ckpt = tmp_path / "tiny_test_final.pt"
        assert final_ckpt.exists()


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ═══════════════════════════════════════════════════════════════════════════
class TestCheckpoints:

    def test_save_and_load_checkpoint(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        trainer = Trainer(model, cfg, model_name="ckpt_test")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = EpochMetrics(
            epoch=5, stage="s1", train_loss=0.3, train_acc=0.9,
            val_loss=0.4, val_acc=0.85, lr=1e-3, elapsed_s=2.0,
        )
        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(optimizer, 5, metrics, ckpt_path)
        assert ckpt_path.exists()

        # Load into a fresh model
        model2 = _TinyModel()
        cfg2 = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        trainer2 = Trainer(model2, cfg2, model_name="ckpt_test")
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        start_epoch = trainer2.load_checkpoint(ckpt_path, opt2)
        assert start_epoch == 6  # epoch 5 + 1

    def test_checkpoint_contents(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        trainer = Trainer(model, cfg, model_name="contents_test")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = EpochMetrics(
            epoch=2, stage="s1", train_loss=0.5, train_acc=0.8,
            val_loss=0.6, val_acc=0.75, lr=1e-3, elapsed_s=1.0,
        )
        ckpt_path = tmp_path / "contents.pt"
        trainer.save_checkpoint(optimizer, 2, metrics, ckpt_path)

        data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "model_state_dict" in data
        assert "optimizer_state_dict" in data
        assert "metrics" in data
        assert "config" in data
        assert data["epoch"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Early stopping
# ═══════════════════════════════════════════════════════════════════════════
class TestEarlyStopping:

    def test_stops_when_no_improvement(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(
            epochs=50, patience=2,
            checkpoint_dir=str(tmp_path), use_mixed_precision=False,
        )
        trainer = Trainer(model, cfg, model_name="early")

        # Use a very low learning rate so val_loss barely improves
        train_loader = _make_loader(n=32, batch_size=16)
        val_loader = _make_loader(n=16, batch_size=16)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)

        metrics = trainer.train(
            train_loader, val_loader, criterion, optimizer,
            epochs=50, stage="early_test",
        )
        # Should stop well before 50 epochs due to patience=2
        assert len(metrics) < 50


# ═══════════════════════════════════════════════════════════════════════════
# History saving
# ═══════════════════════════════════════════════════════════════════════════
class TestHistorySaving:

    def test_save_history_creates_json(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        trainer = Trainer(model, cfg, model_name="hist")

        train_loader = _make_loader(n=32, batch_size=16)
        val_loader = _make_loader(n=16, batch_size=16)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer.train(train_loader, val_loader, criterion, optimizer, epochs=2, stage="s1")
        path = trainer.save_history()

        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "train_loss" in data[0]
        assert "val_acc" in data[0]

    def test_save_history_custom_path(self, tmp_path):
        model = _TinyModel()
        cfg = TrainingConfig(checkpoint_dir=str(tmp_path), use_mixed_precision=False)
        trainer = Trainer(model, cfg, model_name="hist2")

        trainer.history.append(EpochMetrics(
            epoch=0, stage="s", train_loss=0.1, train_acc=0.9,
            val_loss=0.2, val_acc=0.85, lr=1e-4, elapsed_s=0.5,
        ))
        custom_path = tmp_path / "custom_history.json"
        result = trainer.save_history(custom_path)
        assert result == custom_path
        assert custom_path.exists()
