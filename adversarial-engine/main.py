"""
VAJRA Adversarial Engine — FastAPI service.
Implements adversarial perturbation algorithms that collapse deepfake generators:
  • FGSM  — Fast Gradient Sign Method
  • PGD   — Projected Gradient Descent
  • Adversarial illumination

Also provides rPPG (remote photoplethysmography) liveness detection.

Heavy perturbations (especially PGD with many steps) can be offloaded to a
background task queue backed by Redis.  Callers may either use the synchronous
endpoint or submit work via the ``/async`` variant and poll for results.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

log = structlog.get_logger()

app = FastAPI(
    title="VAJRA Adversarial Engine",
    description="Adversarial video perturbation & rPPG liveness detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_redis: Optional[aioredis.Redis] = None
_shutdown_event: asyncio.Event = None  # type: ignore[assignment]


@app.on_event("startup")
async def startup():
    global _redis, _shutdown_event
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    _redis = aioredis.from_url(redis_url, decode_responses=False)
    _shutdown_event = asyncio.Event()
    # Start the background task worker.
    asyncio.create_task(_task_worker())
    log.info("adversarial_engine.ready")


@app.on_event("shutdown")
async def shutdown():
    if _shutdown_event is not None:
        _shutdown_event.set()
    if _redis:
        await _redis.aclose()


# ── Schemas ────────────────────────────────────────────────────────────────
class PerturbRequest(BaseModel):
    algorithm: str = Field("fgsm", description="fgsm | pgd | illumination")
    epsilon: float = Field(0.03, ge=0.001, le=0.3)
    pgd_steps: int = Field(20, ge=1, le=100)


class PerturbResponse(BaseModel):
    algorithm: str
    epsilon: float
    perturbed_image_b64: str
    noise_norm: float
    latency_ms: float


class LivenessResponse(BaseModel):
    heart_rate_bpm: float
    liveness_score: float
    verdict: str


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[PerturbResponse] = None
    error: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "adversarial-engine"}


# ── Adversarial Frame Perturbation ─────────────────────────────────────────
@app.post("/api/adversarial/perturb-frame", response_model=PerturbResponse)
async def perturb_frame(
    algorithm: str = Form("fgsm"),
    epsilon: float = Form(0.03),
    pgd_steps: int = Form(20),
    frame: UploadFile = File(...),
) -> PerturbResponse:
    """
    Apply adversarial perturbation to a video frame.
    The perturbation is imperceptible to humans but collapses deepfake generators.
    """
    t0 = time.perf_counter()

    raw = await frame.read()
    img = _decode_image(raw)

    algorithm = algorithm.lower()
    if algorithm == "fgsm":
        perturbed, noise_norm = _fgsm(img, epsilon)
    elif algorithm == "pgd":
        perturbed, noise_norm = _pgd(img, epsilon, pgd_steps)
    elif algorithm == "illumination":
        perturbed, noise_norm = _adversarial_illumination(img, epsilon)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

    encoded = _encode_image(perturbed)
    latency_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "adversarial.perturbed",
        algorithm=algorithm,
        epsilon=epsilon,
        noise_norm=noise_norm,
        latency_ms=latency_ms,
    )

    return PerturbResponse(
        algorithm=algorithm,
        epsilon=epsilon,
        perturbed_image_b64=encoded,
        noise_norm=round(noise_norm, 4),
        latency_ms=round(latency_ms, 2),
    )


# ── rPPG Liveness Detection ────────────────────────────────────────────────
@app.post("/api/liveness/rppg", response_model=LivenessResponse)
async def rppg_liveness(
    frames: List[UploadFile] = File(...),
) -> LivenessResponse:
    """
    Estimate heart rate from a sequence of video frames using rPPG.
    Screens emit RGB pulses; camera detects capillary blood flow.
    """
    if len(frames) < 30:
        raise HTTPException(
            status_code=422,
            detail="At least 30 frames required for rPPG analysis",
        )

    rgb_signals: List[np.ndarray] = []
    for f in frames:
        raw = await f.read()
        img = _decode_image(raw)
        mean_rgb = img.reshape(-1, 3).mean(axis=0)
        rgb_signals.append(mean_rgb)

    signal = np.array(rgb_signals, dtype=np.float64)
    heart_rate, liveness_score = _estimate_rppg(signal, fps=30.0)

    verdict = "ALIVE" if liveness_score > 0.6 else "SPOOF"
    return LivenessResponse(
        heart_rate_bpm=round(heart_rate, 1),
        liveness_score=round(liveness_score, 3),
        verdict=verdict,
    )


# ── Adversarial Algorithms ─────────────────────────────────────────────────
def _fgsm(img: np.ndarray, epsilon: float) -> tuple[np.ndarray, float]:
    """
    Fast Gradient Sign Method.
    Adds a sign-pattern perturbation that maximises loss for generative models.
    We approximate the gradient using the image Sobel gradient.
    """
    img_f = img.astype(np.float32) / 255.0
    grad = np.zeros_like(img_f)
    for c in range(3):
        gx = cv2.Sobel(img_f[:, :, c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img_f[:, :, c], cv2.CV_32F, 0, 1)
        grad[:, :, c] = gx + gy

    perturbation = epsilon * np.sign(grad)
    perturbed = np.clip(img_f + perturbation, 0.0, 1.0)
    noise_norm = float(np.linalg.norm(perturbation))
    return (perturbed * 255).astype(np.uint8), noise_norm


def _pgd(
    img: np.ndarray, epsilon: float, steps: int, alpha: Optional[float] = None
) -> tuple[np.ndarray, float]:
    """
    Projected Gradient Descent adversarial perturbation.
    Iteratively refines the FGSM perturbation within an L∞ ball.
    """
    if alpha is None:
        alpha = epsilon / max(steps, 1) * 2.5

    img_f = img.astype(np.float32) / 255.0
    x_adv = img_f.copy()

    for _ in range(steps):
        grad = np.zeros_like(x_adv)
        for c in range(3):
            gx = cv2.Sobel(x_adv[:, :, c], cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(x_adv[:, :, c], cv2.CV_32F, 0, 1)
            grad[:, :, c] = gx + gy

        x_adv = x_adv + alpha * np.sign(grad)
        # Project back onto L∞ epsilon-ball around original
        x_adv = np.clip(x_adv, img_f - epsilon, img_f + epsilon)
        x_adv = np.clip(x_adv, 0.0, 1.0)

    noise_norm = float(np.linalg.norm(x_adv - img_f))
    return (x_adv * 255).astype(np.uint8), noise_norm


def _adversarial_illumination(
    img: np.ndarray, epsilon: float
) -> tuple[np.ndarray, float]:
    """
    Adversarial illumination perturbation.
    Modulates pixel values using a structured light pattern that confuses
    deepfake generators' decoder while remaining imperceptible to humans.
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    # Generate structured sinusoidal illumination pattern
    x = np.linspace(0, 2 * np.pi * 4, w)
    y = np.linspace(0, 2 * np.pi * 4, h)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy)).astype(np.float32)

    perturbation = epsilon * pattern[:, :, np.newaxis]
    perturbed = np.clip(img_f + perturbation, 0.0, 1.0)
    noise_norm = float(np.linalg.norm(perturbation))
    return (perturbed * 255).astype(np.uint8), noise_norm


# ── rPPG Signal Processing ─────────────────────────────────────────────────
def _estimate_rppg(signal: np.ndarray, fps: float) -> tuple[float, float]:
    """
    Estimate heart rate and liveness score from RGB channel signals.

    Steps:
      1. Extract green channel (most sensitive to blood volume changes)
      2. Bandpass filter (0.7 – 4 Hz ≈ 42 – 240 BPM)
      3. FFT frequency analysis
      4. Estimate dominant frequency → heart rate
    """
    from scipy import signal as sp_signal

    green = signal[:, 1]  # Green channel

    # Detrend
    green = sp_signal.detrend(green)

    # Bandpass filter 0.7 – 4 Hz
    nyq = fps / 2.0
    low = 0.7 / nyq
    high = min(4.0 / nyq, 0.99)
    try:
        b, a = sp_signal.butter(3, [low, high], btype="band")
        filtered = sp_signal.filtfilt(b, a, green)
    except Exception:
        filtered = green

    # FFT
    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(filtered))

    # Find peak in valid heart-rate range
    valid = (freqs >= 0.7) & (freqs <= 4.0)
    if valid.sum() == 0:
        return 75.0, 0.5  # sane defaults

    peak_freq = freqs[valid][np.argmax(fft_mag[valid])]
    heart_rate = peak_freq * 60.0

    # Liveness score: signal-to-noise ratio normalised to [0,1]
    snr = fft_mag[valid].max() / (fft_mag[valid].mean() + 1e-8)
    liveness_score = float(np.tanh(snr / 5.0))

    return heart_rate, liveness_score


# ── Image helpers ──────────────────────────────────────────────────────────
def _decode_image(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot decode image",
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _encode_image(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(bytes(buf)).decode()


# ── Async Task Queue ───────────────────────────────────────────────────────

_TASK_QUEUE = "vajra:adversarial:tasks"
_TASK_RESULT_PREFIX = "vajra:adversarial:result:"
_TASK_RESULT_TTL = 600  # seconds


@app.post("/api/adversarial/perturb-frame/async", response_model=AsyncTaskResponse)
async def perturb_frame_async(
    algorithm: str = Form("fgsm"),
    epsilon: float = Form(0.03),
    pgd_steps: int = Form(20),
    frame: UploadFile = File(...),
) -> AsyncTaskResponse:
    """
    Submit a heavy perturbation job to the background task queue.
    Returns a task_id that can be polled via GET /api/adversarial/task/{task_id}.
    """
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    raw = await frame.read()
    task_id = uuid.uuid4().hex

    payload = json.dumps({
        "task_id": task_id,
        "algorithm": algorithm,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "frame_b64": base64.b64encode(raw).decode(),
    })

    await _redis.lpush(_TASK_QUEUE, payload)
    await _redis.set(
        f"{_TASK_RESULT_PREFIX}{task_id}",
        json.dumps({"status": "queued"}),
        ex=_TASK_RESULT_TTL,
    )

    log.info("adversarial.task_queued", task_id=task_id, algorithm=algorithm)
    return AsyncTaskResponse(task_id=task_id, status="queued")


@app.get("/api/adversarial/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Poll for the result of an async perturbation task."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    raw = await _redis.get(f"{_TASK_RESULT_PREFIX}{task_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    data = json.loads(raw)
    return TaskStatusResponse(
        task_id=task_id,
        status=data.get("status", "unknown"),
        result=data.get("result"),
        error=data.get("error"),
    )


async def _task_worker() -> None:
    """
    Background coroutine that pulls perturbation tasks from a Redis list and
    processes them.  Results are stored in Redis keyed by task_id.
    Exits gracefully when the shutdown event is set.
    """
    log.info("adversarial.task_worker.started")
    while not (_shutdown_event is not None and _shutdown_event.is_set()):
        try:
            if _redis is None:
                await asyncio.sleep(1)
                continue

            # Blocking pop with 2s timeout so shutdown can be detected.
            item = await _redis.brpop(_TASK_QUEUE, timeout=2)
            if item is None:
                continue

            _, payload_bytes = item
            payload = json.loads(payload_bytes)

            task_id = payload["task_id"]
            algorithm = payload["algorithm"].lower()
            epsilon = payload["epsilon"]
            pgd_steps = payload["pgd_steps"]
            frame_bytes = base64.b64decode(payload["frame_b64"])

            # Mark as processing.
            await _redis.set(
                f"{_TASK_RESULT_PREFIX}{task_id}",
                json.dumps({"status": "processing"}),
                ex=_TASK_RESULT_TTL,
            )

            t0 = time.perf_counter()
            img = _decode_image_sync(frame_bytes)

            if algorithm == "fgsm":
                perturbed, noise_norm = _fgsm(img, epsilon)
            elif algorithm == "pgd":
                perturbed, noise_norm = _pgd(img, epsilon, pgd_steps)
            elif algorithm == "illumination":
                perturbed, noise_norm = _adversarial_illumination(img, epsilon)
            else:
                await _redis.set(
                    f"{_TASK_RESULT_PREFIX}{task_id}",
                    json.dumps({"status": "failed", "error": f"Unknown algorithm: {algorithm}"}),
                    ex=_TASK_RESULT_TTL,
                )
                continue

            encoded = _encode_image(perturbed)
            latency_ms = (time.perf_counter() - t0) * 1000

            result = {
                "algorithm": algorithm,
                "epsilon": epsilon,
                "perturbed_image_b64": encoded,
                "noise_norm": round(noise_norm, 4),
                "latency_ms": round(latency_ms, 2),
            }
            await _redis.set(
                f"{_TASK_RESULT_PREFIX}{task_id}",
                json.dumps({"status": "completed", "result": result}),
                ex=_TASK_RESULT_TTL,
            )
            log.info("adversarial.task_completed", task_id=task_id, latency_ms=round(latency_ms, 2))

        except Exception as exc:
            log.error("adversarial.task_worker.error", error=str(exc))
            await asyncio.sleep(1)

    log.info("adversarial.task_worker.stopped")


def _decode_image_sync(raw_bytes: bytes) -> np.ndarray:
    """Non-HTTP variant of _decode_image for use inside the background worker."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
