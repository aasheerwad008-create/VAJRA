"""
VAJRA Voice AI — Ensemble classifier.

Combines three models into a weighted trust score:
  - Model 1: Deepfake Spectrogram Classifier (EfficientNet-B0 on mel-spec)
  - Model 2: Neural Codec Artifact Detector  (1-D CNN on raw waveform)
  - Model 3: Speaker Verification            (ECAPA-TDNN cosine similarity)

Plus an rPPG liveness placeholder (returns fixed 0.8 in audio-only mode).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from schemas import TrustScore

SAMPLE_RATE = 16_000
N_MELS = 128
CHUNK_SAMPLES = SAMPLE_RATE * 2  # 2-second window


# ── Model 1: Deepfake Spectrogram Classifier ───────────────────────────────
class SpectrogramClassifier(nn.Module):
    """EfficientNet-B0 on mel-spectrograms for REAL/FAKE detection.

    Supports pretrained ImageNet initialization for transfer learning.
    See ``models.spectrogram_model.SpectrogramModel`` for the full
    two-stage freeze/unfreeze training wrapper.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        import timm

        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=N_MELS,
        )
        self.amp_to_db = T.AmplitudeToDB()

        backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=2,
        )
        # Adapt first conv for single-channel input
        old_conv = backbone.conv_stem
        backbone.conv_stem = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.backbone = backbone

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        spec = self.mel(waveform)  # (B, n_mels, T)
        spec = self.amp_to_db(spec)
        spec = spec.unsqueeze(1)  # (B, 1, n_mels, T)
        # Resize to fixed size expected by EfficientNet-B0
        spec = torch.nn.functional.interpolate(spec, size=(128, 128), mode="bilinear", align_corners=False)
        return self.backbone(spec)  # (B, 2)


# ── Model 2: Neural Codec Artifact Detector ────────────────────────────────
class CodecArtifactDetector(nn.Module):
    """Lightweight 1-D CNN classifying HUMAN / ENCODEC / SOUNDSTREAM."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 3),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        x = waveform.unsqueeze(1)  # (B, 1, T)
        return self.net(x)  # (B, 3)


# ── Ensemble ───────────────────────────────────────────────────────────────
class EnsembleClassifier:
    """Weighted ensemble of the three voice AI models + rPPG liveness."""

    WEIGHTS = {
        "deepfake": 0.35,
        "codec": 0.25,
        "speaker": 0.20,
        "liveness": 0.20,
    }

    THRESHOLDS = {"deepfake": 40.0, "suspicious": 70.0}

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spec_model = SpectrogramClassifier().to(self.device).eval()
        self.codec_model = CodecArtifactDetector().to(self.device).eval()

    # ------------------------------------------------------------------
    def score(
        self,
        waveform: np.ndarray,
        enrolled_embedding: Optional[np.ndarray],
    ) -> TrustScore:
        t0 = time.perf_counter()
        waveform = _pad_or_trim(waveform, CHUNK_SAMPLES)
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model 1 — deepfake score (probability of REAL class)
            spec_logits = self.spec_model(tensor)
            p_real_deepfake = float(torch.softmax(spec_logits, dim=-1)[0, 0])

            # Model 2 — codec detector (probability of HUMAN class)
            codec_logits = self.codec_model(tensor)
            p_human = float(torch.softmax(codec_logits, dim=-1)[0, 0])

        # Model 3 — speaker similarity (0-1)
        p_speaker = 0.5  # default when no enrollment
        if enrolled_embedding is not None:
            from models.speaker import SpeakerEmbedder

            embedder = SpeakerEmbedder()
            live_emb = embedder.embed(waveform)
            p_speaker = float(_cosine_sim(live_emb, enrolled_embedding))
            p_speaker = max(0.0, min(1.0, p_speaker))

        # rPPG liveness (audio-only placeholder — 0.8)
        p_liveness = 0.8

        # Weighted trust score 0-100
        raw = (
            self.WEIGHTS["deepfake"] * p_real_deepfake
            + self.WEIGHTS["codec"] * p_human
            + self.WEIGHTS["speaker"] * p_speaker
            + self.WEIGHTS["liveness"] * p_liveness
        )
        trust_score = round(raw * 100, 2)

        if trust_score < self.THRESHOLDS["deepfake"]:
            verdict = "DEEPFAKE"
        elif trust_score < self.THRESHOLDS["suspicious"]:
            verdict = "SUSPICIOUS"
        else:
            verdict = "VERIFIED"

        latency_ms = (time.perf_counter() - t0) * 1000

        return TrustScore(
            score=trust_score,
            verdict=verdict,
            components={
                "deepfake_model": round(p_real_deepfake * 100, 2),
                "codec_detector": round(p_human * 100, 2),
                "speaker_match": round(p_speaker * 100, 2),
                "rppg_liveness": round(p_liveness * 100, 2),
            },
            latency_ms=round(latency_ms, 2),
        )


# ── Utilities ──────────────────────────────────────────────────────────────
def _pad_or_trim(wav: np.ndarray, length: int) -> np.ndarray:
    if len(wav) >= length:
        return wav[:length]
    return np.pad(wav, (0, length - len(wav)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
