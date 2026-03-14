// VAJRA Backend — Polygon Amoy blockchain anchoring handler.
// Extracted from router.go to provide a dedicated module for blockchain logic.
package api

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/vajra/backend/internal/polygon"
	"go.uber.org/zap"
)

const (
	// maxRetries is the number of attempts before giving up on a blockchain call.
	maxRetries = 3
	// baseDelay is the initial backoff delay between retries.
	baseDelay = 500 * time.Millisecond
	// maxDelay caps the exponential backoff to prevent unbounded wait times.
	maxDelay = 5 * time.Second
)

// anchorToBlockchain anchors a ZK proof hash to the Polygon Amoy trust registry.
//
// It tries the primary RPC URL first.  On failure it retries with exponential
// backoff and, when configured, falls back to the secondary RPC URL.
//
// Returns the transaction hash string, or an empty string when anchoring is skipped.
func anchorToBlockchain(cfg *Config, proofHash, userID, verdict string) (string, error) {
	if cfg.TrustRegistryAddress == "0x0000000000000000000000000000000000000000" {
		// No contract deployed — skip anchoring
		return "", nil
	}

	// Try with primary RPC URL first.
	txHash, err := anchorWithRetry(cfg.PolygonRPCURL, cfg.TrustRegistryAddress, proofHash, userID, verdict)
	if err == nil {
		return txHash, nil
	}

	// If a fallback RPC URL is configured, try it.
	if cfg.PolygonFallbackRPCURL != "" {
		txHash, fallbackErr := anchorWithRetry(cfg.PolygonFallbackRPCURL, cfg.TrustRegistryAddress, proofHash, userID, verdict)
		if fallbackErr == nil {
			return txHash, nil
		}
		// Return the original error, but log the fallback failure too.
		return "", fmt.Errorf("primary rpc failed: %w; fallback rpc also failed: %v", err, fallbackErr)
	}

	return "", err
}

// anchorWithRetry attempts to anchor a proof with exponential back-off retries.
func anchorWithRetry(rpcURL, contractAddress, proofHash, userID, verdict string) (string, error) {
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		client := polygon.NewClient(rpcURL, contractAddress)
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		txHash, err := client.AnchorProof(ctx, proofHash, userID, verdict)
		cancel()
		if err == nil {
			return txHash, nil
		}
		lastErr = err

		// Exponential backoff: 500ms, 1s, 2s, capped at maxDelay.
		delay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt)))
		if delay > maxDelay {
			delay = maxDelay
		}
		time.Sleep(delay)
	}
	return "", fmt.Errorf("all %d attempts failed: %w", maxRetries, lastErr)
}

// logBlockchainFailure is a helper used by the verify handler to log
// blockchain anchoring failures at the appropriate severity level.
func logBlockchainFailure(log *zap.Logger, err error) {
	log.Warn("blockchain.anchor_failed",
		zap.Error(err),
		zap.String("hint", "primary and/or fallback RPC may be down"),
	)
}
