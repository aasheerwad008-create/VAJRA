// VAJRA Backend — ZK proof verification handler.
// Extracted from router.go to provide a dedicated module for ZK-proof logic.
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// zkVerifyResp is the JSON response from the ZK proof system service.
type zkVerifyResp struct {
	ProofHash string `json:"proof_hash"`
	Verified  bool   `json:"verified"`
	Verdict   string `json:"verdict"`
}

// callZKVerify proxies a verification request to the ZK proof system service
// and returns the proof hash and verification verdict.
func callZKVerify(zkURL string, req VerifyRequest) (*zkVerifyResp, error) {
	payload := map[string]interface{}{
		"user_id":        req.UserID,
		"speaker_score":  req.TrustScore,
		"liveness_score": req.LivenessScore,
		"key_proof":      req.KeyProof,
		"nullifier":      req.ProofHash,
	}
	body, _ := json.Marshal(payload)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost, zkURL+"/api/zk/verify", bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	var result zkVerifyResp
	if err := json.NewDecoder(httpResp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}
