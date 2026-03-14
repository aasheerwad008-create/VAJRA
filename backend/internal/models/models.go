// VAJRA Backend — data models for verification sessions and API wire types.
package models

import "time"

// ── Verification pipeline ──────────────────────────────────────────────────

// VerifyRequest is the JSON body sent by clients to the /api/verify endpoint.
type VerifyRequest struct {
	UserID        string  `json:"user_id"        binding:"required"`
	TrustScore    float64 `json:"trust_score"    binding:"required"`
	Verdict       string  `json:"verdict"        binding:"required"`
	ProofHash     string  `json:"proof_hash"`
	LivenessScore float64 `json:"liveness_score"`
	KeyProof      string  `json:"key_proof"`
}

// VerifyResponse is the JSON body returned by /api/verify.
type VerifyResponse struct {
	SessionID  string  `json:"session_id"`
	TrustScore float64 `json:"trust_score"`
	Verdict    string  `json:"verdict"`
	ProofHash  string  `json:"proof_hash"`
	TxHash     string  `json:"tx_hash,omitempty"`
	Timestamp  string  `json:"timestamp"`
}

// ── Session records ────────────────────────────────────────────────────────

// SessionStatus represents the lifecycle state of a verification session.
type SessionStatus string

const (
	SessionPending   SessionStatus = "pending"
	SessionCompleted SessionStatus = "completed"
	SessionFailed    SessionStatus = "failed"
)

// SessionRecord represents a row in the verification_sessions table,
// as returned by the /api/history/:user endpoint.
type SessionRecord struct {
	SessionID   string        `json:"session_id"`
	TrustScore  float64       `json:"trust_score"`
	Verdict     string        `json:"verdict"`
	ProofHash   string        `json:"proof_hash"`
	TxHash      string        `json:"tx_hash"`
	Status      SessionStatus `json:"status"`
	CreatedAt   time.Time     `json:"created_at"`
	CompletedAt *time.Time    `json:"completed_at,omitempty"`
}

// ── Enrollment ─────────────────────────────────────────────────────────────

// EnrollRequest is sent to the /api/enroll endpoint to register a voice print.
type EnrollRequest struct {
	UserID string `json:"user_id" binding:"required"`
}

// EnrollResponse is returned after successful enrollment.
type EnrollResponse struct {
	UserID    string `json:"user_id"`
	Status    string `json:"status"`
	Timestamp string `json:"timestamp"`
}

// ── ZK proof wire types ────────────────────────────────────────────────────

// ZKRegisterRequest is the payload forwarded to the ZK proof service for enrollment.
type ZKRegisterRequest struct {
	UserID              string `json:"user_id"`
	BiometricCommitment string `json:"biometric_commitment"`
	KeyProof            string `json:"key_proof"`
}

// ZKVerifyRequest is the payload sent to the ZK proof service for verification.
type ZKVerifyRequest struct {
	UserID        string  `json:"user_id"`
	SpeakerScore  float64 `json:"speaker_score"`
	LivenessScore float64 `json:"liveness_score"`
	KeyProof      string  `json:"key_proof"`
	Nullifier     string  `json:"nullifier"`
}

// ZKVerifyResponse is the response from the ZK proof service.
type ZKVerifyResponse struct {
	ProofHash string `json:"proof_hash"`
	Nullifier string `json:"nullifier"`
	Verified  bool   `json:"verified"`
	Verdict   string `json:"verdict"`
	Timestamp string `json:"timestamp"`
}

// ── Blockchain anchor ──────────────────────────────────────────────────────

// AnchorRequest is sent to the blockchain anchoring layer.
type AnchorRequest struct {
	ProofHash string `json:"proof_hash"`
	UserID    string `json:"user_id"`
	Verdict   string `json:"verdict"`
}

// AnchorResponse is returned after a proof hash is anchored on-chain.
type AnchorResponse struct {
	TxHash      string `json:"tx_hash"`
	BlockNumber uint64 `json:"block_number,omitempty"`
	Status      string `json:"status"`
}
