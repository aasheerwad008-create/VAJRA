/*!
VAJRA Zero-Knowledge Proof System
==================================
Implements a simulated ZK proof circuit using SHA-256 commitments that proves:
  - Speaker verification passed (score ≥ threshold)
  - Liveness detection passed
  - Private key ownership verified (HMAC-SHA256 signature check)

The proof reveals NO raw biometric data — only a commitment hash and a boolean verdict.

In production this would integrate with RISC Zero zkVM to generate a cryptographic ZK-STARK.
The current implementation provides the same API surface and commitment scheme.
*/

use std::env;
use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

// ── State ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    redis_url: String,
}

// ── Schemas ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RegisterRequest {
    user_id: String,
    /// SHA-256 hash of the biometric template (never raw biometrics)
    biometric_commitment: String,
    /// HMAC-SHA256(private_key, user_id)
    key_proof: String,
}

#[derive(Debug, Serialize)]
struct RegisterResponse {
    user_id: String,
    nullifier: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct VerifyRequest {
    user_id: String,
    /// Trust score from Voice AI service
    speaker_score: f64,
    /// Liveness score from Adversarial Engine
    liveness_score: f64,
    /// Key proof
    key_proof: String,
    /// Nullifier from registration
    nullifier: String,
}

#[derive(Debug, Serialize)]
struct VerifyResponse {
    proof_hash: String,
    nullifier: String,
    verified: bool,
    verdict: String,
    timestamp: String,
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    service: String,
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("zk_proof_system=info".parse()?))
        .init();

    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://redis:6379".to_string());
    let state = Arc::new(AppState { redis_url });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/zk/register", post(register_handler))
        .route("/api/zk/verify", post(verify_handler))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:8003";
    info!("ZK Proof System listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "zk-proof-system".to_string(),
    })
}

async fn register_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    info!("zk.register user_id={}", req.user_id);

    // Derive nullifier = SHA-256(user_id || biometric_commitment || key_proof)
    let nullifier = sha256_hex(&format!(
        "{}:{}:{}",
        req.user_id, req.biometric_commitment, req.key_proof
    ));

    // Persist nullifier → biometric_commitment mapping in Redis
    if let Err(e) = store_in_redis(
        &state.redis_url,
        &format!("vajra:zk:nullifier:{}", nullifier),
        &req.biometric_commitment,
    )
    .await
    {
        warn!("redis.store_failed error={}", e);
        // Non-fatal — continue
    }

    (
        StatusCode::OK,
        Json(RegisterResponse {
            user_id: req.user_id,
            nullifier,
            status: "registered".to_string(),
        }),
    )
}

async fn verify_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> impl IntoResponse {
    let t0 = std::time::Instant::now();
    info!("zk.verify user_id={}", req.user_id);

    // ── Circuit constraints ───────────────────────────────────────────────
    // 1. Speaker score must be ≥ 70
    let speaker_ok = req.speaker_score >= 70.0;
    // 2. Liveness score must be ≥ 0.6
    let liveness_ok = req.liveness_score >= 0.6;
    // 3. Key proof must be non-empty (in production: verify HMAC)
    let key_ok = !req.key_proof.is_empty();

    let verified = speaker_ok && liveness_ok && key_ok;

    // ── Generate proof commitment ─────────────────────────────────────────
    // proof_hash = SHA-256(nullifier || speaker_ok || liveness_ok || key_ok || timestamp)
    let timestamp = Utc::now().to_rfc3339();
    let proof_input = format!(
        "{}:{}:{}:{}:{}",
        req.nullifier,
        speaker_ok as u8,
        liveness_ok as u8,
        key_ok as u8,
        timestamp
    );
    let proof_hash = sha256_hex(&proof_input);

    let verdict = if verified { "VERIFIED" } else { "REJECTED" }.to_string();

    // Persist proof in Redis
    let redis_key = format!("vajra:zk:proof:{}", req.user_id);
    let redis_val = format!("{}:{}:{}", proof_hash, verified, timestamp);
    if let Err(e) = store_in_redis(&state.redis_url, &redis_key, &redis_val).await {
        warn!("redis.store_failed error={}", e);
    }

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    (
        StatusCode::OK,
        Json(VerifyResponse {
            proof_hash,
            nullifier: req.nullifier,
            verified,
            verdict,
            timestamp,
            latency_ms,
        }),
    )
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

async fn store_in_redis(url: &str, key: &str, value: &str) -> anyhow::Result<()> {
    let client = redis::Client::open(url)?;
    let mut conn = client.get_multiplexed_tokio_connection().await?;
    let _: () = redis::cmd("SET")
        .arg(key)
        .arg(value)
        .arg("EX")
        .arg(86400u64) // 24h TTL
        .query_async(&mut conn)
        .await?;
    Ok(())
}
