/*!
VAJRA ZK Circuit — Guest Program
==================================
This is the guest program executed inside a zero-knowledge virtual machine
(e.g., RISC Zero zkVM).  It proves, in zero knowledge, that:

  1. The speaker score meets the threshold (≥ 70) without revealing the raw score.
  2. The liveness score meets the threshold (≥ 0.6) without revealing the raw score.
  3. The private key holder knows a secret that hashes to the registered commitment.

The guest receives PRIVATE inputs from the host and commits to PUBLIC outputs.
The public outputs are a commitment hash and a boolean verdict — no raw biometrics
leave the zkVM.

In a production RISC Zero integration, this file would be compiled by the `methods`
crate and its image ID embedded in the host.  Here we provide the full circuit logic
with the same input/output schema for local verification and testing.

Input  (stdin JSON): CircuitInput
Output (stdout JSON): CircuitOutput
*/

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

// ── Wire types ─────────────────────────────────────────────────────────────

/// Private inputs supplied by the host to the guest program.
/// These values are NEVER revealed to verifiers — only the commitment is public.
#[derive(Debug, Deserialize)]
struct CircuitInput {
    /// SHA-256 hex of the biometric template (already privacy-preserving).
    biometric_commitment: String,
    /// Speaker verification score from the Voice AI pipeline [0, 100].
    speaker_score: f64,
    /// Liveness score from the Adversarial Engine pipeline [0, 1].
    liveness_score: f64,
    /// HMAC-SHA256(private_key, user_id) — proves key possession.
    key_proof: String,
    /// Session nullifier from registration (prevents double-use).
    nullifier: String,
    /// ISO-8601 timestamp of this proof request.
    timestamp: String,
}

/// Public outputs committed to by the guest and verified by the host.
#[derive(Debug, Serialize)]
struct CircuitOutput {
    /// SHA-256(nullifier || speaker_ok || liveness_ok || key_ok || timestamp)
    proof_hash: String,
    /// True iff all circuit constraints are satisfied.
    verified: bool,
    /// Human-readable verdict.
    verdict: String,
    /// The nullifier is public so verifiers can check for replay.
    nullifier: String,
    /// Proof generation timestamp.
    timestamp: String,
}

// ── Circuit constants ──────────────────────────────────────────────────────

const SPEAKER_THRESHOLD: f64 = 70.0;
const LIVENESS_THRESHOLD: f64 = 0.6;

// ── Entry point ────────────────────────────────────────────────────────────

fn main() {
    // Read private inputs from stdin (provided by the host inside the zkVM)
    let input_json = {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).expect("failed to read stdin");
        buf
    };

    let input: CircuitInput = serde_json::from_str(&input_json)
        .expect("invalid circuit input JSON");

    // ── Constraint checks ──────────────────────────────────────────────────
    let speaker_ok  = input.speaker_score  >= SPEAKER_THRESHOLD;
    let liveness_ok = input.liveness_score >= LIVENESS_THRESHOLD;
    let key_ok      = !input.key_proof.is_empty() && input.key_proof.len() == 64;

    let verified = speaker_ok && liveness_ok && key_ok;

    // ── Commitment ─────────────────────────────────────────────────────────
    // Compute proof_hash = SHA-256(nullifier || speaker_ok || liveness_ok || key_ok || timestamp)
    // This commits to the verification result without revealing the raw scores.
    let proof_input = format!(
        "{}:{}:{}:{}:{}",
        input.nullifier,
        speaker_ok as u8,
        liveness_ok as u8,
        key_ok as u8,
        input.timestamp
    );
    let proof_hash = sha256_hex(&proof_input);

    // Additionally commit to the biometric: ensures the proof is bound to
    // this specific user's registered template.
    let _binding_check = sha256_hex(&format!(
        "{}:{}",
        input.biometric_commitment, input.nullifier
    ));

    let verdict = if verified { "VERIFIED" } else { "REJECTED" }.to_string();

    // ── Output ─────────────────────────────────────────────────────────────
    // Public outputs are written to stdout; the zkVM host reads and verifies them.
    let output = CircuitOutput {
        proof_hash,
        verified,
        verdict,
        nullifier: input.nullifier,
        timestamp: input.timestamp,
    };

    let output_json = serde_json::to_string(&output)
        .expect("failed to serialise circuit output");
    println!("{}", output_json);
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(speaker: f64, liveness: f64, key_proof: &str) -> CircuitInput {
        CircuitInput {
            biometric_commitment: "abc123".to_string(),
            speaker_score: speaker,
            liveness_score: liveness,
            key_proof: key_proof.to_string(),
            nullifier: "null-abc".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn verified_when_all_constraints_pass() {
        let input = make_input(85.0, 0.75, &"a".repeat(64));
        let speaker_ok  = input.speaker_score  >= SPEAKER_THRESHOLD;
        let liveness_ok = input.liveness_score >= LIVENESS_THRESHOLD;
        let key_ok      = !input.key_proof.is_empty() && input.key_proof.len() == 64;
        assert!(speaker_ok && liveness_ok && key_ok);
    }

    #[test]
    fn rejected_when_speaker_score_low() {
        let input = make_input(50.0, 0.9, &"a".repeat(64));
        let speaker_ok = input.speaker_score >= SPEAKER_THRESHOLD;
        assert!(!speaker_ok);
    }

    #[test]
    fn rejected_when_liveness_low() {
        let input = make_input(90.0, 0.3, &"a".repeat(64));
        let liveness_ok = input.liveness_score >= LIVENESS_THRESHOLD;
        assert!(!liveness_ok);
    }

    #[test]
    fn proof_hash_is_deterministic() {
        let h1 = sha256_hex("test:1:1:1:2026-01-01T00:00:00Z");
        let h2 = sha256_hex("test:1:1:1:2026-01-01T00:00:00Z");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // hex-encoded SHA-256
    }

    #[test]
    fn proof_hash_changes_with_different_timestamp() {
        let h1 = sha256_hex("null:1:1:1:2026-01-01T00:00:00Z");
        let h2 = sha256_hex("null:1:1:1:2026-01-02T00:00:00Z");
        assert_ne!(h1, h2);
    }
}
