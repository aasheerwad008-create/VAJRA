"""
Unit tests for VAJRA Voice AI — models/ensemble.py

Covers the three utility helpers (_pad_or_trim, _cosine_sim, _audio_liveness)
and the forward-pass shapes of SpectrogramClassifier, CodecArtifactDetector,
and the high-level EnsembleClassifier.score() method.
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

from models.ensemble import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    CodecArtifactDetector,
    EnsembleClassifier,
    SpectrogramClassifier,
    _audio_liveness,
    _cosine_sim,
    _pad_or_trim,
)
from schemas import TrustScore


# ═══════════════════════════════════════════════════════════════════════════
# _pad_or_trim
# ═══════════════════════════════════════════════════════════════════════════
class TestPadOrTrim:
    """Ensure waveforms are padded or trimmed to exactly the target length."""

    def test_short_array_is_padded(self):
        wav = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _pad_or_trim(wav, 6)
        assert len(result) == 6
        np.testing.assert_array_equal(result[:3], wav)
        np.testing.assert_array_equal(result[3:], [0.0, 0.0, 0.0])

    def test_long_array_is_trimmed(self):
        wav = np.arange(10, dtype=np.float32)
        result = _pad_or_trim(wav, 5)
        assert len(result) == 5
        np.testing.assert_array_equal(result, wav[:5])

    def test_exact_length_unchanged(self):
        wav = np.ones(8, dtype=np.float32)
        result = _pad_or_trim(wav, 8)
        assert len(result) == 8
        np.testing.assert_array_equal(result, wav)

    def test_empty_array_padded(self):
        wav = np.array([], dtype=np.float32)
        result = _pad_or_trim(wav, 4)
        assert len(result) == 4
        np.testing.assert_array_equal(result, np.zeros(4))


# ═══════════════════════════════════════════════════════════════════════════
# _cosine_sim
# ═══════════════════════════════════════════════════════════════════════════
class TestCosineSim:
    """Cosine similarity between embedding vectors."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert _cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_scaled_vectors_same_direction(self):
        a = np.array([2.0, 4.0])
        b = np.array([1.0, 2.0])
        assert _cosine_sim(a, b) == pytest.approx(1.0, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# _audio_liveness
# ═══════════════════════════════════════════════════════════════════════════
class TestAudioLiveness:
    """Liveness scoring based on RMS energy, ZCR, and spectral flatness."""

    def test_silence_returns_low_score(self):
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float64)
        score = _audio_liveness(silence)
        # Energy=0 and ZCR≈0 pull score down, but spectral flatness of
        # an all-zero signal (after eps padding) sits near 0.5, which the
        # implementation maps to sf_score=1.0.  Combined average ≈ 0.37.
        assert 0.0 <= score < 0.4, f"Silence liveness {score} should be < 0.4"

    def test_pure_sine_returns_moderate_score(self):
        t = np.linspace(0, 2.0, CHUNK_SAMPLES, endpoint=False)
        sine_440 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
        score = _audio_liveness(sine_440)
        assert 0.3 <= score <= 0.8, f"Sine liveness {score} should be 0.3–0.8"

    def test_white_noise_returns_moderate_score(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(CHUNK_SAMPLES).astype(np.float64) * 0.1
        score = _audio_liveness(noise)
        assert 0.3 <= score <= 0.8, f"Noise liveness {score} should be 0.3–0.8"

    def test_speech_like_signal_returns_high_score(self):
        """Amplitude-modulated noise mimics voiced speech characteristics."""
        rng = np.random.default_rng(123)
        t = np.linspace(0, 2.0, CHUNK_SAMPLES, endpoint=False)
        # Voiced-speech-like: formant-ish filtered noise with AM envelope
        carrier = np.sin(2 * np.pi * 150 * t)
        modulator = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # ~3 Hz syllable rate
        noise_component = rng.standard_normal(CHUNK_SAMPLES) * 0.05
        speech_like = ((carrier * modulator) * 0.3 + noise_component).astype(np.float64)
        score = _audio_liveness(speech_like)
        assert score > 0.5, f"Speech-like liveness {score} should be > 0.5"

    def test_very_short_array_returns_zero(self):
        assert _audio_liveness(np.array([0.1])) == 0.0
        assert _audio_liveness(np.array([])) == 0.0

    def test_return_value_clamped_0_1(self):
        rng = np.random.default_rng(7)
        wav = rng.standard_normal(CHUNK_SAMPLES).astype(np.float64)
        score = _audio_liveness(wav)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SpectrogramClassifier
# ═══════════════════════════════════════════════════════════════════════════
class TestSpectrogramClassifier:
    """Forward pass produces the correct (B, 2) output shape."""

    @pytest.fixture(scope="class")
    def model(self):
        return SpectrogramClassifier(pretrained=False).eval()

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


# ═══════════════════════════════════════════════════════════════════════════
# CodecArtifactDetector
# ═══════════════════════════════════════════════════════════════════════════
class TestCodecArtifactDetector:
    """Forward pass produces the correct (B, 3) output shape."""

    @pytest.fixture(scope="class")
    def model(self):
        return CodecArtifactDetector().eval()

    def test_output_shape_single(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_output_shape_batch(self, model):
        x = torch.randn(4, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_finite(self, model):
        x = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════
# EnsembleClassifier.score
# ═══════════════════════════════════════════════════════════════════════════
class TestEnsembleScore:
    """Integration-level test: score() returns a valid TrustScore."""

    @pytest.fixture(scope="class")
    def classifier(self):
        return EnsembleClassifier()

    def test_returns_trust_score(self, classifier):
        wav = np.random.default_rng(0).standard_normal(CHUNK_SAMPLES).astype(np.float32)
        result = classifier.score(wav, enrolled_embedding=None)
        assert isinstance(result, TrustScore)

    def test_score_in_range(self, classifier):
        wav = np.random.default_rng(1).standard_normal(CHUNK_SAMPLES).astype(np.float32)
        result = classifier.score(wav, enrolled_embedding=None)
        assert 0.0 <= result.score <= 100.0

    def test_verdict_is_valid(self, classifier):
        wav = np.random.default_rng(2).standard_normal(CHUNK_SAMPLES).astype(np.float32)
        result = classifier.score(wav, enrolled_embedding=None)
        assert result.verdict in {"VERIFIED", "SUSPICIOUS", "DEEPFAKE"}

    def test_components_present(self, classifier):
        wav = np.random.default_rng(3).standard_normal(CHUNK_SAMPLES).astype(np.float32)
        result = classifier.score(wav, enrolled_embedding=None)
        for key in ("deepfake_model", "codec_detector", "speaker_match", "rppg_liveness"):
            assert key in result.components
            assert 0.0 <= result.components[key] <= 100.0

    def test_latency_positive(self, classifier):
        wav = np.random.default_rng(4).standard_normal(CHUNK_SAMPLES).astype(np.float32)
        result = classifier.score(wav, enrolled_embedding=None)
        assert result.latency_ms > 0.0

    def test_silence_scores_low(self, classifier):
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        result = classifier.score(silence, enrolled_embedding=None)
        # Liveness component is low-ish for silence (~36.67) due to zero
        # energy and ZCR, though spectral flatness edge-case boosts it.
        assert result.components["rppg_liveness"] < 40.0
