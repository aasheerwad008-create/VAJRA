"""
Tests for the VAJRA adversarial-engine module.

Covers:
  - FGSM perturbation (shape, magnitude bound)
  - PGD perturbation (shape, L∞ bound)
  - Adversarial illumination (shape)
  - rPPG signal estimation
  - ScreenPulseGenerator (generate, apply_to_frame, HMAC, validation)
  - generate_pulse_signal helper
  - FastAPI /health and /api/adversarial/perturb-frame endpoints
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure the adversarial-engine root is importable.
ENGINE_ROOT = str(Path(__file__).resolve().parent.parent)
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from main import (
    _adversarial_illumination,
    _decode_image_sync,
    _encode_image,
    _estimate_rppg,
    _fgsm,
    _pgd,
    app,
)
from screen_pulse import (
    PulseConfig,
    PulseFrame,
    ScreenPulseGenerator,
    generate_pulse_signal,
)

# ── Helpers ────────────────────────────────────────────────────────────────

def _make_rgb_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 RGB image (H, W, 3)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _encode_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    """Encode an RGB array to PNG bytes (suitable for upload)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return bytes(buf)


# ── FGSM Tests ─────────────────────────────────────────────────────────────

class TestFGSM:
    def test_output_shape_matches_input(self):
        img = _make_rgb_image(48, 64)
        perturbed, _ = _fgsm(img, epsilon=0.03)
        assert perturbed.shape == img.shape
        assert perturbed.dtype == np.uint8

    def test_perturbation_magnitude_bounded(self):
        img = _make_rgb_image(32, 32)
        eps = 0.05
        perturbed, noise_norm = _fgsm(img, eps)
        diff = np.abs(perturbed.astype(np.float32) / 255.0
                      - img.astype(np.float32) / 255.0)
        # Per-pixel diff should be <= epsilon (with small float tolerance)
        assert diff.max() <= eps + 1e-3

    def test_noise_norm_is_positive(self):
        img = _make_rgb_image()
        _, noise_norm = _fgsm(img, epsilon=0.03)
        assert noise_norm > 0.0

    def test_different_epsilon_different_result(self):
        img = _make_rgb_image()
        _, norm_small = _fgsm(img, epsilon=0.01)
        _, norm_large = _fgsm(img, epsilon=0.1)
        assert norm_large > norm_small


# ── PGD Tests ──────────────────────────────────────────────────────────────

class TestPGD:
    def test_output_shape_matches_input(self):
        img = _make_rgb_image(48, 64)
        perturbed, _ = _pgd(img, epsilon=0.03, steps=5)
        assert perturbed.shape == img.shape
        assert perturbed.dtype == np.uint8

    def test_linf_bound(self):
        """PGD output must stay within the L∞ epsilon-ball of the input."""
        img = _make_rgb_image(32, 32)
        eps = 0.06
        perturbed, _ = _pgd(img, eps, steps=10)
        diff = np.abs(perturbed.astype(np.float32) / 255.0
                      - img.astype(np.float32) / 255.0)
        # Allow a tiny tolerance for uint8 rounding
        assert diff.max() <= eps + 2.0 / 255.0

    def test_noise_norm_is_positive(self):
        img = _make_rgb_image()
        _, noise_norm = _pgd(img, epsilon=0.03, steps=5)
        assert noise_norm > 0.0

    def test_multiple_steps_produces_nonzero_norm(self):
        img = _make_rgb_image()
        _, norm = _pgd(img, 0.05, steps=20)
        assert norm > 0.0


# ── Illumination Tests ─────────────────────────────────────────────────────

class TestAdversarialIllumination:
    def test_output_shape_matches_input(self):
        img = _make_rgb_image(48, 64)
        perturbed, _ = _adversarial_illumination(img, epsilon=0.03)
        assert perturbed.shape == img.shape
        assert perturbed.dtype == np.uint8

    def test_noise_norm_positive(self):
        img = _make_rgb_image()
        _, noise_norm = _adversarial_illumination(img, 0.03)
        assert noise_norm > 0.0

    def test_perturbation_bounded(self):
        img = _make_rgb_image(32, 32)
        eps = 0.05
        perturbed, _ = _adversarial_illumination(img, eps)
        diff = np.abs(perturbed.astype(np.float32) / 255.0
                      - img.astype(np.float32) / 255.0)
        assert diff.max() <= eps + 1e-3


# ── rPPG Estimation Tests ──────────────────────────────────────────────────

class TestEstimateRPPG:
    def _synthetic_signal(self, bpm: float = 72.0, fps: float = 30.0,
                          duration_s: float = 10.0) -> np.ndarray:
        """Create a synthetic RGB signal with a known heart-rate frequency."""
        n = int(fps * duration_s)
        t = np.arange(n) / fps
        freq = bpm / 60.0
        green = np.sin(2 * np.pi * freq * t)
        red = green * 0.4 + np.random.default_rng(0).normal(0, 0.1, n)
        blue = green * 0.2 + np.random.default_rng(1).normal(0, 0.1, n)
        return np.stack([red, green, blue], axis=1)

    def test_heart_rate_within_range(self):
        signal = self._synthetic_signal(bpm=72.0)
        hr, _ = _estimate_rppg(signal, fps=30.0)
        assert 40.0 <= hr <= 240.0

    def test_liveness_score_between_0_and_1(self):
        signal = self._synthetic_signal(bpm=80.0)
        _, score = _estimate_rppg(signal, fps=30.0)
        assert 0.0 <= score <= 1.0

    def test_returns_tuple(self):
        signal = self._synthetic_signal()
        result = _estimate_rppg(signal, fps=30.0)
        assert isinstance(result, tuple) and len(result) == 2


# ── ScreenPulseGenerator Tests ─────────────────────────────────────────────

class TestScreenPulseGenerator:
    def test_generate_correct_count(self):
        gen = ScreenPulseGenerator(heart_rate_bpm=72.0)
        frames = gen.generate(100)
        assert len(frames) == 100

    def test_luminance_delta_bounded(self):
        amp = 0.004
        gen = ScreenPulseGenerator(
            config=PulseConfig(amplitude=amp, harmonics=3)
        )
        frames = gen.generate(300)
        deltas = [f.luminance_delta for f in frames]
        # Sum of harmonic amps = amp*(1 + 1/2 + 1/4) = amp*1.75
        bound = amp * sum(1 / (2 ** h) for h in range(3)) + 1e-9
        assert all(-bound <= d <= bound for d in deltas)

    def test_phase_wraps_to_2pi(self):
        gen = ScreenPulseGenerator(heart_rate_bpm=72.0)
        frames = gen.generate(500)
        for f in frames:
            assert 0 <= f.phase_rad < 2 * math.pi + 1e-9

    def test_timestamp_increases(self):
        gen = ScreenPulseGenerator(heart_rate_bpm=72.0, fps=30.0)
        frames = gen.generate(50)
        for i in range(1, len(frames)):
            assert frames[i].timestamp_s > frames[i - 1].timestamp_s

    def test_hmac_stream_when_nonce_provided(self):
        gen = ScreenPulseGenerator(
            config=PulseConfig(
                nonce=b"session-123",
                secret_key=b"top-secret",
            )
        )
        frames = gen.generate(10)
        for f in frames:
            assert f.hmac_byte is not None
            assert 0 <= f.hmac_byte <= 255

    def test_hmac_none_without_nonce(self):
        gen = ScreenPulseGenerator(heart_rate_bpm=72.0)
        frames = gen.generate(5)
        for f in frames:
            assert f.hmac_byte is None

    def test_verify_frame_hmac(self):
        gen = ScreenPulseGenerator(
            config=PulseConfig(
                nonce=b"nonce-abc",
                secret_key=b"key-xyz",
            )
        )
        frames = gen.generate(5)
        for f in frames:
            assert gen.verify_frame_hmac(f) is True

    def test_apply_to_frame_shape(self):
        gen = ScreenPulseGenerator(heart_rate_bpm=72.0, amplitude=0.004)
        pulse_frames = gen.generate(1)
        img = _make_rgb_image(16, 16)
        out = gen.apply_to_frame(img, pulse_frames[0])
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_invalid_bpm_raises(self):
        with pytest.raises(ValueError, match="heart_rate_bpm"):
            ScreenPulseGenerator(heart_rate_bpm=10.0)

    def test_invalid_amplitude_raises(self):
        with pytest.raises(ValueError, match="amplitude"):
            ScreenPulseGenerator(
                config=PulseConfig(amplitude=0.5)
            )


# ── generate_pulse_signal Tests ────────────────────────────────────────────

class TestGeneratePulseSignal:
    def test_output_length(self):
        sig = generate_pulse_signal(duration_s=5.0, fps=30.0)
        assert len(sig) == 150

    def test_dtype_float32(self):
        sig = generate_pulse_signal()
        assert sig.dtype == np.float32

    def test_values_bounded(self):
        amp = 0.004
        sig = generate_pulse_signal(amplitude=amp, harmonics=3)
        bound = amp * sum(1 / (2 ** h) for h in range(3)) + 1e-6
        assert sig.max() <= bound
        assert sig.min() >= -bound


# ── Image Codec Helpers ────────────────────────────────────────────────────

class TestImageHelpers:
    def test_decode_image_sync_roundtrip(self):
        img = _make_rgb_image(32, 32)
        png_bytes = _encode_to_png_bytes(img)
        decoded = _decode_image_sync(png_bytes)
        assert decoded.shape == img.shape
        # PNG is lossless so values should match
        np.testing.assert_array_equal(decoded, img)

    def test_encode_image_returns_base64(self):
        img = _make_rgb_image(16, 16)
        b64 = _encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 0
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


# ── FastAPI Endpoint Tests ─────────────────────────────────────────────────

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

# Patch Redis to avoid real connections during endpoint tests.
@pytest.fixture()
def client():
    with patch("main._redis", None), \
         patch("main._shutdown_event", None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "adversarial-engine"


class TestPerturbFrameEndpoint:
    def _png_file(self, h: int = 32, w: int = 32) -> tuple[str, bytes, str]:
        """Return a (filename, bytes, content-type) tuple for upload."""
        img = _make_rgb_image(h, w)
        return ("frame.png", _encode_to_png_bytes(img), "image/png")

    def test_fgsm_returns_200(self, client):
        resp = client.post(
            "/api/adversarial/perturb-frame",
            data={"algorithm": "fgsm", "epsilon": "0.03"},
            files={"frame": self._png_file()},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["algorithm"] == "fgsm"
        assert "perturbed_image_b64" in body
        assert body["noise_norm"] >= 0

    def test_pgd_returns_200(self, client):
        resp = client.post(
            "/api/adversarial/perturb-frame",
            data={"algorithm": "pgd", "epsilon": "0.03", "pgd_steps": "5"},
            files={"frame": self._png_file()},
        )
        assert resp.status_code == 200
        assert resp.json()["algorithm"] == "pgd"

    def test_illumination_returns_200(self, client):
        resp = client.post(
            "/api/adversarial/perturb-frame",
            data={"algorithm": "illumination", "epsilon": "0.03"},
            files={"frame": self._png_file()},
        )
        assert resp.status_code == 200
        assert resp.json()["algorithm"] == "illumination"

    def test_unknown_algorithm_returns_400(self, client):
        resp = client.post(
            "/api/adversarial/perturb-frame",
            data={"algorithm": "unknown", "epsilon": "0.03"},
            files={"frame": self._png_file()},
        )
        assert resp.status_code == 400

    def test_response_contains_latency(self, client):
        resp = client.post(
            "/api/adversarial/perturb-frame",
            data={"algorithm": "fgsm", "epsilon": "0.03"},
            files={"frame": self._png_file()},
        )
        body = resp.json()
        assert "latency_ms" in body
        assert body["latency_ms"] > 0
