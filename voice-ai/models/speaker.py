"""
VAJRA Voice AI — Speaker Embedder (ECAPA-TDNN via SpeechBrain).
Falls back to a lightweight TDNN if SpeechBrain model download fails.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EMBEDDING_DIM = 192
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))


class _LightweightTDNN(nn.Module):
    """Lightweight TDNN producing 192-d speaker embeddings (offline fallback)."""

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


class SpeakerEmbedder:
    """Produces 192-d speaker embeddings from raw 16-kHz mono audio."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    def _load_model(self) -> nn.Module:
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(MODEL_DIR / "spkrec-ecapa"),
                run_opts={"device": str(self.device)},
            )
            return model  # type: ignore[return-value]
        except Exception:
            # Offline fallback
            m = _LightweightTDNN().to(self.device).eval()
            return m

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """Return a (192,) L2-normalised embedding."""
        if isinstance(self._model, _LightweightTDNN):
            return self._embed_tdnn(waveform)
        return self._embed_speechbrain(waveform)

    def _embed_speechbrain(self, waveform: np.ndarray) -> np.ndarray:
        import torchaudio.compliance.kaldi as kaldi

        wav = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_batch(wav)  # type: ignore[union-attr]
        emb_np = emb.squeeze().cpu().numpy()
        return (emb_np / (np.linalg.norm(emb_np) + 1e-8)).astype(np.float32)

    def _embed_tdnn(self, waveform: np.ndarray) -> np.ndarray:
        import torchaudio.compliance.kaldi as kaldi

        wav = torch.from_numpy(waveform).unsqueeze(0)
        feats = kaldi.fbank(wav, num_mel_bins=40, sample_frequency=16000)  # (T, 40)
        feats = feats.unsqueeze(0).permute(0, 2, 1).to(self.device)  # (1, 40, T)
        with torch.no_grad():
            emb = self._model(feats).squeeze().cpu().numpy()
        return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)
