package crypto

import (
	"testing"
)

// ── GeneratePQCKeyPair ─────────────────────────────────────────────────────

func TestGeneratePQCKeyPair_ValidSeed(t *testing.T) {
	kp, err := GeneratePQCKeyPair("this-is-a-valid-seed-with-enough-length")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if kp.Algorithm != "dilithium-sha512-v1" {
		t.Errorf("algorithm = %q, want %q", kp.Algorithm, "dilithium-sha512-v1")
	}
	// SHA-512 → 64 bytes → hex: public 64 chars, secret 64 chars
	if len(kp.PublicKey) != 64 {
		t.Errorf("public key length = %d, want 64", len(kp.PublicKey))
	}
	if len(kp.SecretKey) != 64 {
		t.Errorf("secret key length = %d, want 64", len(kp.SecretKey))
	}
	if kp.PublicKey == kp.SecretKey {
		t.Error("public and secret keys must differ")
	}
}

func TestGeneratePQCKeyPair_ShortSeed(t *testing.T) {
	_, err := GeneratePQCKeyPair("short")
	if err == nil {
		t.Fatal("expected error for short seed, got nil")
	}
}

func TestGeneratePQCKeyPair_Determinism(t *testing.T) {
	seed := "deterministic-test-seed-32bytes!"
	kp1, err := GeneratePQCKeyPair(seed)
	if err != nil {
		t.Fatalf("first keygen: %v", err)
	}
	kp2, err := GeneratePQCKeyPair(seed)
	if err != nil {
		t.Fatalf("second keygen: %v", err)
	}
	if kp1.PublicKey != kp2.PublicKey {
		t.Error("same seed must produce same public key")
	}
	if kp1.SecretKey != kp2.SecretKey {
		t.Error("same seed must produce same secret key")
	}
}

func TestGeneratePQCKeyPair_DifferentSeeds(t *testing.T) {
	kp1, _ := GeneratePQCKeyPair("seed-alpha-1234567890")
	kp2, _ := GeneratePQCKeyPair("seed-bravo-0987654321")
	if kp1.PublicKey == kp2.PublicKey {
		t.Error("different seeds must produce different public keys")
	}
}

// ── PQCSign ────────────────────────────────────────────────────────────────

func TestPQCSign_ValidKeyPair(t *testing.T) {
	kp, _ := GeneratePQCKeyPair("test-signing-seed-abcdefgh")
	sig, err := PQCSign(kp, "hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sig.Signature == "" {
		t.Error("signature must not be empty")
	}
	if sig.PublicKey != kp.PublicKey {
		t.Error("signature public key must match key pair")
	}
	if sig.Algorithm != kp.Algorithm {
		t.Error("signature algorithm must match key pair")
	}
	// SHA-256 → 32 bytes → 64 hex chars
	if len(sig.Signature) != 64 {
		t.Errorf("signature length = %d, want 64", len(sig.Signature))
	}
}

func TestPQCSign_NoSecretKey(t *testing.T) {
	kp := &PQCKeyPair{PublicKey: "abcd1234", SecretKey: "", Algorithm: "dilithium-sha512-v1"}
	_, err := PQCSign(kp, "message")
	if err == nil {
		t.Fatal("expected error when signing without secret key")
	}
}

func TestPQCSign_Determinism(t *testing.T) {
	kp, _ := GeneratePQCKeyPair("deterministic-signing-seed-xyz")
	msg := "test message for determinism"
	sig1, _ := PQCSign(kp, msg)
	sig2, _ := PQCSign(kp, msg)
	if sig1.Signature != sig2.Signature {
		t.Error("same key+message must produce identical signatures")
	}
}

func TestPQCSign_DifferentMessages(t *testing.T) {
	kp, _ := GeneratePQCKeyPair("signing-different-messages-seed")
	sig1, _ := PQCSign(kp, "message A")
	sig2, _ := PQCSign(kp, "message B")
	if sig1.Signature == sig2.Signature {
		t.Error("different messages must produce different signatures")
	}
}

// ── PQCVerify ──────────────────────────────────────────────────────────────

func TestPQCVerify_ValidSignature(t *testing.T) {
	kp, _ := GeneratePQCKeyPair("verify-test-seed-0123456789")
	sig, _ := PQCSign(kp, "verify me")
	if !PQCVerify(sig, "verify me") {
		t.Error("valid signature should verify")
	}
}

func TestPQCVerify_InvalidSignatureLength(t *testing.T) {
	sig := &PQCSignature{
		Signature: "tooshort",
		PublicKey: "a]b]c]d",
		Algorithm: "dilithium-sha512-v1",
	}
	if PQCVerify(sig, "anything") {
		t.Error("short signature should not verify")
	}
}

func TestPQCVerify_EmptyFields(t *testing.T) {
	if PQCVerify(&PQCSignature{}, "msg") {
		t.Error("empty signature should not verify")
	}
	if PQCVerify(&PQCSignature{Signature: "", PublicKey: "abc"}, "msg") {
		t.Error("empty signature field should not verify")
	}
	if PQCVerify(&PQCSignature{Signature: "abc", PublicKey: ""}, "msg") {
		t.Error("empty public key should not verify")
	}
}

func TestPQCVerify_InvalidHex(t *testing.T) {
	sig := &PQCSignature{
		// 64 chars but not valid hex
		Signature: "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
		PublicKey: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Algorithm: "dilithium-sha512-v1",
	}
	if PQCVerify(sig, "msg") {
		t.Error("invalid hex signature should not verify")
	}
}

// ── SignAttestation ────────────────────────────────────────────────────────

func TestSignAttestation_EndToEnd(t *testing.T) {
	kp, err := GeneratePQCKeyPair("attestation-signing-seed-12345")
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}

	att, err := SignAttestation(kp, "user-42", "deadbeef1234", "VERIFIED", "2025-01-01T00:00:00Z")
	if err != nil {
		t.Fatalf("sign attestation: %v", err)
	}

	if att.UserID != "user-42" {
		t.Errorf("UserID = %q, want %q", att.UserID, "user-42")
	}
	if att.ProofHash != "deadbeef1234" {
		t.Errorf("ProofHash = %q, want %q", att.ProofHash, "deadbeef1234")
	}
	if att.Verdict != "VERIFIED" {
		t.Errorf("Verdict = %q, want %q", att.Verdict, "VERIFIED")
	}
	if att.Timestamp != "2025-01-01T00:00:00Z" {
		t.Errorf("Timestamp mismatch")
	}
	if att.Signature.Signature == "" {
		t.Error("attestation signature must not be empty")
	}
	if att.Signature.PublicKey != kp.PublicKey {
		t.Error("attestation public key must match key pair")
	}

	// Verify the signature structurally
	if !PQCVerify(&att.Signature, "user-42:deadbeef1234:VERIFIED:2025-01-01T00:00:00Z") {
		t.Error("attestation signature should verify")
	}
}

func TestSignAttestation_NoSecretKey(t *testing.T) {
	kp := &PQCKeyPair{PublicKey: "pub", SecretKey: "", Algorithm: "dilithium-sha512-v1"}
	_, err := SignAttestation(kp, "u", "p", "v", "t")
	if err == nil {
		t.Fatal("expected error when signing attestation without secret key")
	}
}
