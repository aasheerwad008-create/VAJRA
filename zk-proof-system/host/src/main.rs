/*!
VAJRA Zero-Knowledge Proof System — Host
==========================================
HTTP service that generates and verifies Fiat-Shamir ZK proofs for
biometric identity verification.

The host:
  1. Receives private biometric inputs from the Go backend.
  2. Executes the ZK circuit (via the `proof` module) to generate a
     binding commitment and Fiat-Shamir proof.
  3. Returns the proof (commitment + challenge + response + proof_hash)
     — **no raw biometric data leaves this service**.
  4. Provides a /api/zk/verify-proof endpoint for independent proof
     verification without access to the private witness.

Architecture:
  - `proof.rs` — Fiat-Shamir STARK-like proof engine (prover + verifier)
  - `main.rs` — Axum HTTP service with Redis-backed session storage

In production, the `proof::prove()` call would be replaced with a
RISC Zero zkVM execution that produces a STARK receipt.  The API
surface and proof schema remain identical.
*/

mod proof;

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
    /// Fiat-Shamir proof components — available for independent verification.
    proof: ProofComponents,
}

/// Proof components exposed to the verifier.
#[derive(Debug, Serialize)]
struct ProofComponents {
    commitment: String,
    challenge: String,
    response: String,
    proof_type: String,
}

/// Request body for the /api/zk/verify-proof endpoint.
#[derive(Debug, Deserialize)]
struct VerifyProofRequest {
    commitment: String,
    challenge: String,
    response: String,
    proof_hash: String,
    nullifier: String,
    timestamp: String,
    verified: bool,
    verdict: String,
}

/// Response from the /api/zk/verify-proof endpoint.
#[derive(Debug, Serialize)]
struct VerifyProofResponse {
    valid: bool,
    reason: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    service: String,
    proof_system: String,
    version: String,
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
        .route("/api/zk/verify-proof", post(verify_proof_handler))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:8003";
    info!("ZK Proof System v2.0 (Fiat-Shamir) listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "zk-proof-system".to_string(),
        proof_system: "fiat-shamir-stark".to_string(),
        version: "2.0.0".to_string(),
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
    info!("zk.verify user_id={} (Fiat-Shamir proof generation)", req.user_id);

    let timestamp = Utc::now().to_rfc3339();

    // ── Build the circuit witness (private inputs) ────────────────────────
    let witness = proof::CircuitWitness {
        user_id: req.user_id.clone(),
        speaker_score: req.speaker_score,
        liveness_score: req.liveness_score,
        key_proof: req.key_proof,
        nullifier: req.nullifier.clone(),
        timestamp: timestamp.clone(),
    };

    // ── Generate Fiat-Shamir proof ────────────────────────────────────────
    let fs_proof = proof::prove(&witness);

    info!(
        "zk.proof_generated user_id={} verified={} proof_hash={} latency_ms={:.2}",
        req.user_id, fs_proof.statement.verified, fs_proof.proof_hash, fs_proof.latency_ms
    );

    // ── Self-verify the proof to ensure correctness ───────────────────────
    let self_check = proof::verify(&fs_proof);
    if !self_check.valid {
        warn!(
            "zk.self_verify_failed user_id={} reason={}",
            req.user_id, self_check.reason
        );
    }

    // Persist proof in Redis
    let redis_key = format!("vajra:zk:proof:{}", req.user_id);
    let redis_val = serde_json::to_string(&fs_proof).unwrap_or_default();
    if let Err(e) = store_in_redis(&state.redis_url, &redis_key, &redis_val).await {
        warn!("redis.store_failed error={}", e);
    }

    (
        StatusCode::OK,
        Json(VerifyResponse {
            proof_hash: fs_proof.proof_hash.clone(),
            nullifier: req.nullifier,
            verified: fs_proof.statement.verified,
            verdict: fs_proof.statement.verdict.clone(),
            timestamp,
            latency_ms: fs_proof.latency_ms,
            proof: ProofComponents {
                commitment: fs_proof.commitment,
                challenge: fs_proof.challenge,
                response: fs_proof.response,
                proof_type: "fiat-shamir-stark-v2".to_string(),
            },
        }),
    )
}

/// Independent proof verification endpoint.
/// Verifiers can submit proof components and receive a validity check
/// without needing access to the original private witness.
async fn verify_proof_handler(
    Json(req): Json<VerifyProofRequest>,
) -> impl IntoResponse {
    info!("zk.verify_proof nullifier={}", req.nullifier);

    let fs_proof = proof::FiatShamirProof {
        commitment: req.commitment,
        challenge: req.challenge,
        response: req.response,
        statement: proof::CircuitStatement {
            nullifier: req.nullifier,
            timestamp: req.timestamp,
            verified: req.verified,
            verdict: req.verdict,
        },
        proof_hash: req.proof_hash,
        latency_ms: 0.0,
    };

    let result = proof::verify(&fs_proof);

    (
        StatusCode::OK,
        Json(VerifyProofResponse {
            valid: result.valid,
            reason: result.reason,
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
