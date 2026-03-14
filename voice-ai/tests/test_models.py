"""
Unit tests for VAJRA Voice AI — New pretrained models.

Covers ECAPA-TDNN classifier, RawNet2 model, and their components:
    - Forward pass shapes
    - Freeze / unfreeze transfer learning utilities
    - Parameter group splitting for discriminative LR
    - SincConv filter bank
    - ResidualBlock
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from models.ecapa_tdnn import ECAPATDNNClassifier, EMBEDDING_DIM
from models.rawnet2 import RawNet2, SincConv, ResidualBlock

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = SAMPLE_RATE * 2  # 2-second window


# ═══════════════════════════════════════════════════════════════════════════
# ECAPATDNNClassifier
# ═══════════════════════════════════════════════════════════════════════════
class TestECAPATDNNClassifier:
    """ECAPA-TDNN anti-spoofing classifier tests."""

    @pytest.fixture(scope="class")
    def model(self):
        return ECAPATDNNClassifier(num_classes=2, pretrained=False).eval()

    def test_output_shape_single(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_shape_batch(self, model):
        x = torch.randn(4, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_is_finite(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 192

    def test_num_classes_configurable(self):
        model3 = ECAPATDNNClassifier(num_classes=3, pretrained=False).eval()
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model3(x)
        assert out.shape == (1, 3)


class TestECAPATDNNTransferLearning:
    """Freeze / unfreeze transfer learning for ECAPA-TDNN."""

    def test_freeze_reduces_trainable_params(self):
        model = ECAPATDNNClassifier(num_classes=2, pretrained=False)
        total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.freeze_backbone()
        total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_after < total_before

    def test_unfreeze_restores_params(self):
        model = ECAPATDNNClassifier(num_classes=2, pretrained=False)
        total_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.freeze_backbone()
        model.unfreeze_backbone()
        total_restored = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_restored == total_all

    def test_unfreeze_all(self):
        model = ECAPATDNNClassifier(num_classes=2, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_all()
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_get_trainable_params(self):
        model = ECAPATDNNClassifier(num_classes=2, pretrained=False)
        model.freeze_backbone()
        trainable = model.get_trainable_params()
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)

    def test_get_param_groups(self):
        model = ECAPATDNNClassifier(num_classes=2, pretrained=False)
        groups = model.get_param_groups(lr_backbone=1e-5, lr_head=1e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# SincConv
# ═══════════════════════════════════════════════════════════════════════════
class TestSincConv:
    """Sinc convolution layer tests."""

    @pytest.fixture(scope="class")
    def sinc(self):
        return SincConv(out_channels=64, kernel_size=129).eval()

    def test_output_shape(self, sinc):
        x = torch.randn(2, 1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = sinc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64
        assert out.shape[2] == CHUNK_SAMPLES  # same padding

    def test_output_is_finite(self, sinc):
        x = torch.randn(1, 1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = sinc(x)
        assert torch.isfinite(out).all()

    def test_learnable_parameters(self, sinc):
        """Sinc layer has learnable low_hz and band_hz parameters."""
        param_names = {n for n, _ in sinc.named_parameters()}
        assert "low_hz_" in param_names
        assert "band_hz_" in param_names


# ═══════════════════════════════════════════════════════════════════════════
# ResidualBlock
# ═══════════════════════════════════════════════════════════════════════════
class TestResidualBlock:
    """Residual block with FMS tests."""

    def test_output_shape_same_channels(self):
        block = ResidualBlock(128, 128).eval()
        x = torch.randn(2, 128, 100)
        with torch.no_grad():
            out = block(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 128
        # MaxPool1d(3) reduces temporal dim
        assert out.shape[2] == 100 // 3

    def test_output_shape_different_channels(self):
        block = ResidualBlock(128, 256).eval()
        x = torch.randn(2, 128, 99)
        with torch.no_grad():
            out = block(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 256

    def test_output_is_finite(self):
        block = ResidualBlock(64, 128).eval()
        x = torch.randn(1, 64, 50)
        with torch.no_grad():
            out = block(x)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════
# RawNet2
# ═══════════════════════════════════════════════════════════════════════════
class TestRawNet2:
    """RawNet2 model forward pass tests."""

    @pytest.fixture(scope="class")
    def model(self):
        return RawNet2(num_classes=2).eval()

    def test_output_shape_single(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_shape_batch(self, model):
        x = torch.randn(4, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_is_finite(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_num_classes_configurable(self):
        model3 = RawNet2(num_classes=3).eval()
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model3(x)
        assert out.shape == (1, 3)

    def test_short_waveform(self, model):
        """Model should handle waveforms shorter than 2 seconds."""
        x = torch.randn(1, SAMPLE_RATE)  # 1 second
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)


class TestRawNet2TransferLearning:
    """Freeze / unfreeze transfer learning for RawNet2."""

    def test_freeze_reduces_trainable_params(self):
        model = RawNet2(num_classes=2)
        total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.freeze_feature_extractor()
        total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_after < total_before

    def test_unfreeze_restores_params(self):
        model = RawNet2(num_classes=2)
        total_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.freeze_feature_extractor()
        model.unfreeze_all()
        total_restored = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_restored == total_all

    def test_get_trainable_params(self):
        model = RawNet2(num_classes=2)
        model.freeze_feature_extractor()
        trainable = model.get_trainable_params()
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)

    def test_get_param_groups(self):
        model = RawNet2(num_classes=2)
        groups = model.get_param_groups(lr_features=1e-5, lr_head=1e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-4

    def test_frozen_forward_pass_still_works(self):
        model = RawNet2(num_classes=2).eval()
        model.freeze_feature_extractor()
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()
