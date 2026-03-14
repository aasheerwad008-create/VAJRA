"""
VAJRA Voice AI — Pydantic schemas.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class TrustScore(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0, description="0-100 trust score")
    verdict: str = Field(..., description="VERIFIED | SUSPICIOUS | DEEPFAKE")
    components: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0


class EnrollRequest(BaseModel):
    user_id: str


class EnrollResponse(BaseModel):
    user_id: str
    status: str


class VerifyResponse(BaseModel):
    user_id: str
    trust: TrustScore
    latency_ms: float
