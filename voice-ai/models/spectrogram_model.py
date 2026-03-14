"""
VAJRA Voice AI — Pretrained EfficientNet-B0 Spectrogram Model.

Implements a two-stage transfer learning architecture:

Stage 1 — Feature Extraction:
    Load ImageNet-pretrained EfficientNet-B0, freeze the backbone,
    replace the classifier head, and train only the head.

Stage 2 — Fine-Tuning:
    Unfreeze the last N blocks of EfficientNet-B0 and fine-tune
    the entire network with a smaller learning rate.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_MELS = 128


class SpectrogramModel(nn.Module):
    """
    EfficientNet-B0 classifier for mel-spectrogram deepfake detection.

    Supports pretrained initialization and two-stage freeze/unfreeze
    for transfer learning.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: REAL, FAKE).
    pretrained : bool
        If True, load ImageNet-pretrained weights (default True).
    dropout : float
        Dropout rate for the classifier head (default 0.3).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        import timm

        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=N_MELS,
        )
        self.amp_to_db = T.AmplitudeToDB()

        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            in_chans=1,     # Single-channel mel spectrogram
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            feat_dim = self.backbone(dummy).shape[-1]

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._num_classes = num_classes

        if pretrained:
            log.info(
                "SpectrogramModel: loaded pretrained EfficientNet-B0 "
                "(ImageNet weights), feature_dim=%d", feat_dim
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        waveform : Tensor, shape (B, T)
            Raw audio waveform.

        Returns
        -------
        Tensor, shape (B, num_classes) — class logits.
        """
        spec = self.mel(waveform)  # (B, n_mels, T)
        spec = self.amp_to_db(spec)
        spec = spec.unsqueeze(1)  # (B, 1, n_mels, T)
        spec = torch.nn.functional.interpolate(
            spec, size=(128, 128), mode="bilinear", align_corners=False
        )
        features = self.backbone(spec)  # (B, feat_dim)
        return self.classifier(features)  # (B, num_classes)

    # ── Transfer learning utilities ───────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters (Stage 1).
        Only the classifier head will be trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info("SpectrogramModel: backbone frozen (classifier-only training)")

    def unfreeze_backbone(self, unfreeze_from: int = 5) -> None:
        """
        Unfreeze backbone layers starting from a given block (Stage 2).

        EfficientNet-B0 has 7 blocks (0-6). Setting ``unfreeze_from=5``
        unfreezes blocks 5 and 6 (the last two).

        Parameters
        ----------
        unfreeze_from : int
            Block index from which to start unfreezing (default 5).
        """
        # First, ensure everything is frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze blocks from unfreeze_from onwards
        blocks = list(self.backbone.blocks) if hasattr(self.backbone, 'blocks') else []
        for i, block in enumerate(blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze the final norm and head layers
        for name, param in self.backbone.named_parameters():
            if "bn" in name and "blocks" not in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            "SpectrogramModel: unfrozen from block %d — %s / %s params trainable (%.1f%%)",
            unfreeze_from,
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        log.info("SpectrogramModel: all parameters unfrozen")

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """
        Get parameter groups with different learning rates for fine-tuning.

        Parameters
        ----------
        lr_backbone : float
            Learning rate for backbone parameters.
        lr_head : float
            Learning rate for classifier head.

        Returns
        -------
        List of parameter group dicts for the optimizer.
        """
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = list(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
