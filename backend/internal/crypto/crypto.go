// VAJRA Backend — Cryptographic utilities.
// Provides SHA-256 hashing and HMAC-SHA256 signing helpers used throughout
// the backend for proof commitment generation and request authentication.
package crypto

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// SHA256Hex returns the lowercase hex-encoded SHA-256 hash of the input string.
func SHA256Hex(input string) string {
	h := sha256.Sum256([]byte(input))
	return hex.EncodeToString(h[:])
}

// HMACSHA256Hex returns the lowercase hex-encoded HMAC-SHA256 of msg using key.
// Used to generate and verify key-possession proofs.
func HMACSHA256Hex(key, msg string) string {
	mac := hmac.New(sha256.New, []byte(key))
	mac.Write([]byte(msg))
	return hex.EncodeToString(mac.Sum(nil))
}

// VerifyHMAC performs a constant-time comparison between the expected HMAC
// of msg under key and the provided signature.
// Returns true if and only if the signatures match.
func VerifyHMAC(key, msg, signature string) bool {
	expected := HMACSHA256Hex(key, msg)
	return hmac.Equal([]byte(expected), []byte(signature))
}

// GenerateNullifier produces a cryptographically random 32-byte nullifier
// encoded as a hex string.  Used to prevent proof replay attacks.
func GenerateNullifier() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("crypto.GenerateNullifier: %w", err)
	}
	return hex.EncodeToString(b), nil
}

// DeriveIdentityCommitment computes:
//
//	keccak256-equivalent = SHA-256(userID || ":" || nullifier)
//
// This is the public identity commitment stored on-chain.
// It binds a user to their nullifier without revealing the user ID.
func DeriveIdentityCommitment(userID, nullifier string) string {
	return SHA256Hex(userID + ":" + nullifier)
}

// ProofHash computes the ZK proof commitment hash from its components.
// Matches the formula used in the ZK guest circuit.
func ProofHash(nullifier string, speakerOK, livenessOK, keyOK bool, timestamp string) string {
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return 0
	}
	input := fmt.Sprintf(
		"%s:%d:%d:%d:%s",
		nullifier, b2i(speakerOK), b2i(livenessOK), b2i(keyOK), timestamp,
	)
	return SHA256Hex(input)
}
