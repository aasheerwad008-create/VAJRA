"""
VAJRA Voice AI Service — FastAPI entry-point.
Provides REST + WebSocket endpoints for voice enrollment, verification,
and real-time streaming analysis.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import Any

import numpy as np
import redis.asyncio as aioredis
import soundfile as sf
import structlog
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.ensemble import EnsembleClassifier
from models.speaker import SpeakerEmbedder
from schemas import (
    EnrollRequest,
    EnrollResponse,
    HealthResponse,
    TrustScore,
    VerifyResponse,
)
from storage import EmbeddingStore

# ── Logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VAJRA Voice AI",
    description="Real-time AI voice liveness & deepfake detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialised in lifespan) ─────────────────────────────────────
_ensemble: EnsembleClassifier | None = None
_embedder: SpeakerEmbedder | None = None
_store: EmbeddingStore | None = None
_redis: aioredis.Redis | None = None

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 2


@app.on_event("startup")
async def startup():
    global _ensemble, _embedder, _store, _redis

    log.info("voice_ai.startup")
    _ensemble = EnsembleClassifier()
    _embedder = SpeakerEmbedder()

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    _redis = aioredis.from_url(redis_url, decode_responses=False)

    db_url = os.getenv("DATABASE_URL", "")
    _store = EmbeddingStore(db_url)
    await _store.init()

    log.info("voice_ai.ready")


@app.on_event("shutdown")
async def shutdown():
    if _redis:
        await _redis.aclose()


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="voice-ai")


# ── Enroll ─────────────────────────────────────────────────────────────────
@app.post("/api/voice/enroll", response_model=EnrollResponse)
async def enroll(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
) -> EnrollResponse:
    """Enrol a user by extracting and persisting their speaker embedding."""
    raw = await audio.read()
    waveform = _load_audio(raw)

    embedding = _embedder.embed(waveform)  # type: ignore[union-attr]
    await _store.save_embedding(user_id, embedding)  # type: ignore[union-attr]

    log.info("voice_ai.enrolled", user_id=user_id)
    return EnrollResponse(user_id=user_id, status="enrolled")


# ── Verify ─────────────────────────────────────────────────────────────────
@app.post("/api/voice/verify", response_model=VerifyResponse)
async def verify(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
) -> VerifyResponse:
    """Verify a voice sample against the enrolled embedding."""
    raw = await audio.read()
    waveform = _load_audio(raw)

    stored_embedding = await _store.get_embedding(user_id)  # type: ignore[union-attr]
    if stored_embedding is None:
        raise HTTPException(status_code=404, detail="User not enrolled")

    trust = _ensemble.score(waveform, stored_embedding)  # type: ignore[union-attr]

    log.info(
        "voice_ai.verified",
        user_id=user_id,
        trust_score=trust.score,
        verdict=trust.verdict,
    )
    return VerifyResponse(user_id=user_id, trust=trust, latency_ms=trust.latency_ms)


# ── WebSocket streaming ────────────────────────────────────────────────────
@app.websocket("/ws/voice/stream/{session_id}")
async def voice_stream(websocket: WebSocket, session_id: str):
    """
    Bidirectional streaming endpoint.
    Client sends binary PCM16 audio chunks; server responds with JSON trust scores.
    """
    await websocket.accept()
    log.info("ws.connected", session_id=session_id)

    buffer = np.array([], dtype=np.float32)
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS

    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)

            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm])

            if len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                trust = _ensemble.score(chunk, enrolled_embedding=None)  # type: ignore[union-attr]
                await websocket.send_text(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "trust_score": trust.score,
                            "verdict": trust.verdict,
                            "components": trust.components,
                            "timestamp": time.time(),
                        }
                    )
                )

                # Publish to Redis stream for downstream consumers
                if _redis:
                    await _redis.xadd(
                        f"vajra:voice:{session_id}",
                        {
                            "trust_score": str(trust.score),
                            "verdict": trust.verdict,
                        },
                        maxlen=100,
                    )

    except (WebSocketDisconnect, asyncio.TimeoutError):
        log.info("ws.disconnected", session_id=session_id)
    except Exception as exc:
        log.error("ws.error", session_id=session_id, error=str(exc))
        await websocket.close(code=1011)


# ── Helpers ────────────────────────────────────────────────────────────────
def _load_audio(raw_bytes: bytes) -> np.ndarray:
    """Load any audio format and resample to 16 kHz mono float32."""
    import librosa

    buf = io.BytesIO(raw_bytes)
    try:
        waveform, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot decode audio: {exc}",
        ) from exc
    return waveform.astype(np.float32)
