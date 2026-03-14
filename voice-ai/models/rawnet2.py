"""
VAJRA Voice AI — RawNet2 Anti-Spoofing Model.

End-to-end raw waveform model for deepfake / spoofing detection.
Implements a simplified RawNet2 architecture:

    1. Sinc convolution layer (learnable bandpass filters on raw audio)
    2. Residual blocks with batch normalisation and LeakyReLU
    3. GRU temporal aggregation
    4. Fully connected classifier

Reference:
    Tak, H. et al., "End-to-End Anti-Spoofing with RawNet2",
    Proc. ICASSP, 2021.
"""
from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


# ── Sinc Convolution Layer ─────────────────────────────────────────────────

class SincConv(nn.Module):
    """Learnable sinc-based bandpass filter bank operating on raw waveforms.

    Initialises filters as mel-spaced sinc functions and learns the low/high
    cutoff frequencies during training.

    Parameters
    ----------
    out_channels : int
        Number of filters (default 128).
    kernel_size : int
        Length of each FIR filter (default 129, must be odd).
    sample_rate : int
        Audio sample rate in Hz (default 16000).
    min_low_hz : float
        Minimum lower cutoff frequency.
    min_band_hz : float
        Minimum bandwidth of each filter.
    """

    def __init__(
        self,
        out_channels: int = 128,
        kernel_size: int = 129,
        sample_rate: int = SAMPLE_RATE,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialise filter cutoffs on mel scale
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_low = 2595.0 * math.log10(1.0 + min_low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + (min_low_hz + high_hz) / 700.0)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(
            torch.tensor(hz_points[:-1], dtype=torch.float32).view(-1, 1)
        )
        self.band_hz_ = nn.Parameter(
            torch.tensor(np.diff(hz_points), dtype=torch.float32).view(-1, 1)
        )

        # Hamming window (not learned)
        n = torch.linspace(0, kernel_size - 1, kernel_size)
        self.register_buffer(
            "window",
            0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (kernel_size - 1)),
        )
        n_half = (kernel_size - 1) // 2
        self.register_buffer(
            "n_",
            (2.0 * math.pi * torch.arange(-n_half, 0, dtype=torch.float32).view(1, -1))
            / sample_rate,
        )

    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised sinc: sin(x) / x, with sinc(0) = 1."""
        eps = 1e-8
        return torch.where(
            x.abs() < eps,
            torch.ones_like(x),
            torch.sin(x) / (x + eps),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : Tensor, shape (B, 1, T)

        Returns
        -------
        Tensor, shape (B, out_channels, T')
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2.0,
        )

        # Build bandpass filters via sinc difference
        low_pass1 = 2.0 * low / self.sample_rate * self._sinc(low * self.n_)
        low_pass2 = 2.0 * high / self.sample_rate * self._sinc(high * self.n_)

        band_pass_left = low_pass2 - low_pass1
        band_pass_centre = (2.0 * high - 2.0 * low) / self.sample_rate
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_centre, band_pass_right], dim=1)
        band_pass = band_pass * self.window
        # Normalise energy
        band_pass = band_pass / (band_pass.norm(dim=1, keepdim=True) + 1e-8)

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            waveform,
            filters,
            stride=1,
            padding=self.kernel_size // 2,
        )


# ── Residual Block ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block with FMS (Feature Map Scaling)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.3)
        self.pool = nn.MaxPool1d(3)

        # Skip connection
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Feature Map Scaling (FMS)
        self.fms_fc = nn.Linear(out_channels, out_channels)
        self.fms_sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.bn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        out = out + residual

        # FMS: channel-wise attention
        t = out.mean(dim=-1)  # (B, C)
        t = self.fms_sig(self.fms_fc(t)).unsqueeze(-1)  # (B, C, 1)
        out = out * t + t  # scale + shift

        out = self.pool(out)
        return out


# ── RawNet2 Model ──────────────────────────────────────────────────────────

class RawNet2(nn.Module):
    """
    RawNet2 end-to-end anti-spoofing model.

    Operates directly on raw audio waveforms. Architecture:
        1. SincConv layer (learnable bandpass filters)
        2. Two stacks of residual blocks
        3. GRU temporal aggregation
        4. Fully connected classifier

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: REAL, FAKE).
    sinc_channels : int
        Number of sinc filters in the first layer (default 128).
    sinc_kernel : int
        Sinc filter length (default 129).
    res_channels : tuple
        Channel sizes for the two residual block stacks.
    gru_hidden : int
        GRU hidden dimension.
    gru_layers : int
        Number of GRU layers.
    dropout : float
        Dropout rate for the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        sinc_channels: int = 128,
        sinc_kernel: int = 129,
        res_channels: tuple = (128, 512),
        gru_hidden: int = 256,
        gru_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes

        # ── Layer 1: Sinc convolution ─────────────────────────────────────
        self.sinc_conv = SincConv(
            out_channels=sinc_channels,
            kernel_size=sinc_kernel,
        )
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)
        self.sinc_pool = nn.MaxPool1d(3)
        self.sinc_lrelu = nn.LeakyReLU(0.3)

        # ── Layer 2-3: Residual blocks ────────────────────────────────────
        self.res_block1 = ResidualBlock(sinc_channels, res_channels[0])
        self.res_block2 = ResidualBlock(res_channels[0], res_channels[0])
        self.res_block3 = ResidualBlock(res_channels[0], res_channels[1])
        self.res_block4 = ResidualBlock(res_channels[1], res_channels[1])

        # ── Layer 4: GRU ──────────────────────────────────────────────────
        self.bn_before_gru = nn.BatchNorm1d(res_channels[1])
        self.gru = nn.GRU(
            input_size=res_channels[1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Layer 5: Classifier ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

        log.info(
            "RawNet2: sinc_channels=%d, res_channels=%s, gru_hidden=%d, "
            "num_classes=%d",
            sinc_channels,
            res_channels,
            gru_hidden,
            num_classes,
        )

    def _init_weights(self) -> None:
        """Initialize non-sinc weights using Kaiming initialization."""
        for name, m in self.named_modules():
            if "sinc_conv" in name:
                continue  # SincConv has its own mel-scale init
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        waveform : Tensor, shape (B, T)
            Raw audio waveform at 16 kHz.

        Returns
        -------
        Tensor, shape (B, num_classes) — class logits.
        """
        x = waveform.unsqueeze(1)  # (B, 1, T)

        # Sinc layer
        x = self.sinc_conv(x)      # (B, sinc_channels, T)
        x = self.sinc_bn(x)
        x = self.sinc_lrelu(x)
        x = self.sinc_pool(x)      # (B, sinc_channels, T/3)

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # GRU
        x = self.bn_before_gru(x)
        x = x.permute(0, 2, 1)     # (B, T', C) for GRU
        x, _ = self.gru(x)
        x = x[:, -1, :]            # Last hidden state (B, gru_hidden)

        # Classifier
        return self.classifier(x)  # (B, num_classes)

    # ── Transfer learning utilities ───────────────────────────────────────

    def freeze_feature_extractor(self) -> None:
        """
        Freeze sinc layer and residual blocks (Stage 1).
        Only the GRU and classifier head will be trainable.
        """
        for param in self.sinc_conv.parameters():
            param.requires_grad = False
        for param in self.sinc_bn.parameters():
            param.requires_grad = False
        for block in [self.res_block1, self.res_block2, self.res_block3, self.res_block4]:
            for param in block.parameters():
                param.requires_grad = False
        log.info("RawNet2: feature extractor frozen (GRU + classifier trainable)")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            "RawNet2: all parameters unfrozen — %s / %s params trainable (%.1f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self, lr_features: float, lr_head: float) -> list:
        """
        Get parameter groups with different learning rates for fine-tuning.

        Parameters
        ----------
        lr_features : float
            Learning rate for sinc + residual block parameters.
        lr_head : float
            Learning rate for GRU + classifier parameters.

        Returns
        -------
        List of parameter group dicts for the optimizer.
        """
        feature_params = []
        for module in [
            self.sinc_conv, self.sinc_bn,
            self.res_block1, self.res_block2,
            self.res_block3, self.res_block4,
        ]:
            feature_params.extend(p for p in module.parameters() if p.requires_grad)

        head_params = []
        for module in [self.bn_before_gru, self.gru, self.classifier]:
            head_params.extend(p for p in module.parameters() if p.requires_grad)

        return [
            {"params": feature_params, "lr": lr_features},
            {"params": head_params, "lr": lr_head},
        ]
