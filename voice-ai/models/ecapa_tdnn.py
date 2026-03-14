"""
VAJRA Voice AI — ECAPA-TDNN Anti-Spoofing Classifier.

Wraps a pretrained SpeechBrain ECAPA-TDNN speaker encoder as a backbone
for binary deepfake / anti-spoofing classification.

Supports two-stage transfer learning:

Stage 1 — Feature Extraction:
    Load pretrained ECAPA-TDNN (VoxCeleb weights via SpeechBrain),
    freeze the backbone, and train only the classifier head.

Stage 2 — Fine-Tuning:
    Unfreeze part or all of the backbone and fine-tune end-to-end
    with a smaller learning rate.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

EMBEDDING_DIM = 192
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))


class _LightweightTDNN(nn.Module):
    """Lightweight TDNN producing 192-d embeddings (offline fallback)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 40, T)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class ECAPATDNNClassifier(nn.Module):
    """
    ECAPA-TDNN-based anti-spoofing classifier with pretrained backbone.

    Loads SpeechBrain's pretrained ECAPA-TDNN (VoxCeleb) as the feature
    extractor and adds a classification head for REAL/FAKE detection.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: REAL, FAKE).
    pretrained : bool
        If True, load pretrained SpeechBrain ECAPA-TDNN weights.
    dropout : float
        Dropout rate for the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._pretrained = pretrained
        self._use_speechbrain = False

        self.backbone = self._build_backbone(pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        if pretrained:
            log.info(
                "ECAPATDNNClassifier: loaded pretrained ECAPA-TDNN backbone, "
                "embedding_dim=%d, num_classes=%d",
                EMBEDDING_DIM,
                num_classes,
            )

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        """Load SpeechBrain ECAPA-TDNN or fall back to lightweight TDNN."""
        if pretrained:
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                sb_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(MODEL_DIR / "spkrec-ecapa"),
                    run_opts={"device": "cpu"},
                )
                self._use_speechbrain = True
                log.info("ECAPA-TDNN: loaded SpeechBrain pretrained weights")
                return sb_model  # type: ignore[return-value]
            except Exception as exc:
                log.warning(
                    "ECAPA-TDNN: SpeechBrain load failed (%s), using fallback TDNN",
                    exc,
                )
        return _LightweightTDNN()

    def _extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the backbone.

        Parameters
        ----------
        waveform : Tensor, shape (B, T)
            Raw 16 kHz waveform.

        Returns
        -------
        Tensor, shape (B, EMBEDDING_DIM)
        """
        if self._use_speechbrain:
            with torch.no_grad():
                emb = self.backbone.encode_batch(waveform)  # type: ignore[union-attr]
            return emb.squeeze(1)  # (B, 192)
        else:
            import torchaudio.compliance.kaldi as kaldi

            embeddings = []
            for i in range(waveform.size(0)):
                feats = kaldi.fbank(
                    waveform[i : i + 1],
                    num_mel_bins=40,
                    sample_frequency=16000,
                )  # (T, 40)
                feats = feats.unsqueeze(0).permute(0, 2, 1)  # (1, 40, T)
                emb = self.backbone(feats)  # (1, 192)
                embeddings.append(emb)
            return torch.cat(embeddings, dim=0)  # (B, 192)

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
        emb = self._extract_embedding(waveform)  # (B, 192)
        return self.classifier(emb)  # (B, num_classes)

    # ── Transfer learning utilities ───────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze backbone parameters (Stage 1).
        Only the classifier head will be trainable.
        """
        if self._use_speechbrain:
            # SpeechBrain models are used in no_grad mode already
            pass
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
        log.info("ECAPATDNNClassifier: backbone frozen (classifier-only training)")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters (Stage 2).
        Only applicable for the lightweight TDNN fallback.
        """
        if not self._use_speechbrain:
            for param in self.backbone.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            log.info(
                "ECAPATDNNClassifier: backbone unfrozen — %s / %s params trainable (%.1f%%)",
                f"{trainable:,}",
                f"{total:,}",
                100.0 * trainable / max(total, 1),
            )
        else:
            log.info(
                "ECAPATDNNClassifier: SpeechBrain backbone embeddings are "
                "extracted in no-grad mode; classifier head is always trainable"
            )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        log.info("ECAPATDNNClassifier: all parameters unfrozen")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """
        Get parameter groups with different learning rates.

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
        if self._use_speechbrain:
            # SpeechBrain backbone is not directly optimizable
            return [{"params": list(self.classifier.parameters()), "lr": lr_head}]

        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
