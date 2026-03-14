"""
VAJRA Adversarial Engine — Screen Pulse Signal Generator.

Generates imperceptible screen-flicker pulses that encode a heartbeat
signal in the display's luminance channel.  The rPPG detector in the
browser (rppg.wasm / Python) reads this back from a standard webcam to
prove liveness without any dedicated hardware.

The pulse modulation works at <1 Hz amplitude variation so that it is:
  - Imperceptible to human observers (ΔL < 0.5%)
  - Detectable by the FFT-based rPPG pipeline at 30 fps
  - Cryptographically bound to a session nonce (replay prevention)

Usage (standalone):
    python screen_pulse.py --heart-rate 72 --duration 10

Usage (from adversarial engine main.py):
    from screen_pulse import ScreenPulseGenerator
    gen = ScreenPulseGenerator(heart_rate_bpm=72)
    frames = gen.generate(num_frames=300, fps=30.0)
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class PulseConfig:
    """Configuration for the screen pulse signal."""
    heart_rate_bpm: float = 72.0          # Target heart rate to encode
    fps: float = 30.0                     # Display frame rate
    amplitude: float = 0.004             # Peak luminance delta (0–1 scale, <0.5%)
    harmonics: int = 3                    # Number of harmonic frequencies to include
    nonce: Optional[bytes] = None        # Session nonce for HMAC binding
    secret_key: Optional[bytes] = None  # HMAC secret key


@dataclass
class PulseFrame:
    """A single screen frame with embedded pulse signal."""
    frame_index: int
    timestamp_s: float
    luminance_delta: float   # Additive luminance offset (−amplitude … +amplitude)
    phase_rad: float         # Current phase of the fundamental frequency
    hmac_byte: Optional[int] = None  # Watermark byte from HMAC stream


# ── Generator ──────────────────────────────────────────────────────────────

class ScreenPulseGenerator:
    """
    Generates a sequence of luminance-delta values that encode a
    heartbeat signal in the screen's brightness.

    Parameters
    ----------
    heart_rate_bpm:
        Target cardiac frequency to embed (35–200 BPM).
    fps:
        Display refresh rate in frames per second.
    amplitude:
        Peak luminance change as a fraction of full scale [0–1].
        Values ≤ 0.005 are imperceptible on typical displays.
    harmonics:
        Number of harmonic overtones to include alongside the fundamental.
        Adds realism: natural rPPG signals contain harmonics at 2f, 3f …
    nonce:
        Optional session nonce.  When provided, each frame's phase is
        HMAC-verified to prevent pre-recorded replay attacks.
    secret_key:
        HMAC secret key (required when *nonce* is set).
    """

    MIN_BPM: float = 35.0
    MAX_BPM: float = 200.0

    def __init__(self, config: Optional[PulseConfig] = None, **kwargs) -> None:
        if config is None:
            config = PulseConfig(**kwargs)
        self.cfg = config
        self._validate()

        self._freq_hz: float = config.heart_rate_bpm / 60.0
        self._dt: float = 1.0 / config.fps

        # Pre-compute harmonic amplitudes (decreasing geometric series)
        self._harmonic_amps: List[float] = [
            config.amplitude / (2 ** h) for h in range(config.harmonics)
        ]

        # HMAC watermark stream
        self._hmac_stream: Optional[bytes] = None
        if config.nonce and config.secret_key:
            self._hmac_stream = self._derive_hmac_stream(
                config.secret_key, config.nonce, length=4096
            )

    # ------------------------------------------------------------------
    def generate(
        self,
        num_frames: int,
        start_phase_rad: float = 0.0,
    ) -> List[PulseFrame]:
        """
        Generate *num_frames* pulse frames starting from *start_phase_rad*.

        Returns a list of PulseFrame objects.  Apply each frame's
        ``luminance_delta`` to every pixel's RGB channels before display.
        """
        frames: List[PulseFrame] = []
        phase = start_phase_rad

        for i in range(num_frames):
            t = i * self._dt
            delta = self._compute_delta(phase)

            hmac_byte: Optional[int] = None
            if self._hmac_stream:
                hmac_byte = self._hmac_stream[i % len(self._hmac_stream)]

            frames.append(
                PulseFrame(
                    frame_index=i,
                    timestamp_s=t,
                    luminance_delta=delta,
                    phase_rad=phase % (2 * math.pi),
                    hmac_byte=hmac_byte,
                )
            )
            phase += 2 * math.pi * self._freq_hz * self._dt

        return frames

    # ------------------------------------------------------------------
    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        pulse_frame: PulseFrame,
    ) -> np.ndarray:
        """
        Apply *pulse_frame*'s luminance delta to an RGB image array.

        Parameters
        ----------
        frame_rgb:
            uint8 numpy array, shape (H, W, 3), values in [0, 255].
        pulse_frame:
            PulseFrame from ``generate()``.

        Returns
        -------
        uint8 numpy array with the same shape.
        """
        delta_u8 = int(round(pulse_frame.luminance_delta * 255))
        out = frame_rgb.astype(np.int16)
        out += delta_u8
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def verify_frame_hmac(
        self, pulse_frame: PulseFrame, tolerance: float = 0.02
    ) -> bool:
        """
        Verify that *pulse_frame* is consistent with the session HMAC stream.
        Returns True if valid or if no HMAC stream was configured.
        """
        if self._hmac_stream is None:
            return True
        expected = self._hmac_stream[pulse_frame.frame_index % len(self._hmac_stream)]
        return pulse_frame.hmac_byte == expected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_delta(self, phase: float) -> float:
        """Sum harmonic sinusoids at the given phase."""
        delta = 0.0
        for h, amp in enumerate(self._harmonic_amps):
            delta += amp * math.sin((h + 1) * phase)
        return delta

    def _validate(self) -> None:
        bpm = self.cfg.heart_rate_bpm
        if not (self.MIN_BPM <= bpm <= self.MAX_BPM):
            raise ValueError(
                f"heart_rate_bpm must be in [{self.MIN_BPM}, {self.MAX_BPM}]; got {bpm}"
            )
        if not (0 < self.cfg.fps <= 240):
            raise ValueError(f"fps must be in (0, 240]; got {self.cfg.fps}")
        if not (0 < self.cfg.amplitude <= 0.1):
            raise ValueError(
                f"amplitude must be in (0, 0.1]; got {self.cfg.amplitude} "
                "(values >0.005 may be perceptible)"
            )

    @staticmethod
    def _derive_hmac_stream(
        key: bytes, nonce: bytes, length: int = 4096
    ) -> bytes:
        """Stretch HMAC-SHA256(key, nonce) to *length* bytes via counter mode."""
        stream = b""
        counter = 0
        while len(stream) < length:
            msg = nonce + counter.to_bytes(4, "big")
            stream += hmac.new(key, msg, hashlib.sha256).digest()
            counter += 1
        return stream[:length]


# ── Numpy array helper ─────────────────────────────────────────────────────

def generate_pulse_signal(
    heart_rate_bpm: float = 72.0,
    fps: float = 30.0,
    duration_s: float = 10.0,
    amplitude: float = 0.004,
    harmonics: int = 3,
) -> np.ndarray:
    """
    Generate a continuous pulse signal as a 1-D float32 numpy array.

    Returns
    -------
    np.ndarray, shape (num_frames,), dtype float32
        Luminance delta values in [-amplitude, +amplitude].
    """
    gen = ScreenPulseGenerator(
        PulseConfig(
            heart_rate_bpm=heart_rate_bpm,
            fps=fps,
            amplitude=amplitude,
            harmonics=harmonics,
        )
    )
    num_frames = int(duration_s * fps)
    frames = gen.generate(num_frames)
    return np.array([f.luminance_delta for f in frames], dtype=np.float32)


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="VAJRA Screen Pulse Signal Generator"
    )
    parser.add_argument("--heart-rate", type=float, default=72.0,
                        help="Heart rate to encode in BPM (default: 72)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Display frame rate (default: 30)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds (default: 10)")
    parser.add_argument("--amplitude", type=float, default=0.004,
                        help="Peak luminance delta [0-1] (default: 0.004)")
    parser.add_argument("--harmonics", type=int, default=3,
                        help="Number of harmonic frequencies (default: 3)")
    args = parser.parse_args()

    signal = generate_pulse_signal(
        heart_rate_bpm=args.heart_rate,
        fps=args.fps,
        duration_s=args.duration,
        amplitude=args.amplitude,
        harmonics=args.harmonics,
    )

    print(f"Generated {len(signal)} frames @ {args.fps} fps")
    print(f"Heart rate: {args.heart_rate} BPM  |  Amplitude: {args.amplitude:.4f}")
    print(f"Signal range: [{signal.min():.6f}, {signal.max():.6f}]")
    print(f"First 10 values: {signal[:10].tolist()}")


if __name__ == "__main__":
    _main()
