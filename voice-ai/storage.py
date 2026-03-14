"""
VAJRA Voice AI — Persistent storage for speaker embeddings.
Uses PostgreSQL via asyncpg with a simple table.
"""
from __future__ import annotations

from typing import Optional

import asyncpg
import numpy as np


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS voice_embeddings (
    user_id TEXT PRIMARY KEY,
    embedding BYTEA NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""


class EmbeddingStore:
    def __init__(self, db_url: str) -> None:
        # Convert SQLAlchemy-style URL to asyncpg URL
        self._dsn = db_url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: Optional[asyncpg.Pool] = None

    async def init(self) -> None:
        if not self._dsn:
            return
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as conn:
                await conn.execute(CREATE_TABLE)
        except Exception as exc:
            import structlog

            structlog.get_logger().warning("storage.init_failed", error=str(exc))
            self._pool = None

    async def save_embedding(self, user_id: str, embedding: np.ndarray) -> None:
        if self._pool is None:
            return
        blob = embedding.astype(np.float32).tobytes()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO voice_embeddings (user_id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """,
                user_id,
                blob,
            )

    async def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        if self._pool is None:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT embedding FROM voice_embeddings WHERE user_id = $1", user_id
            )
        if row is None:
            return None
        return np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
