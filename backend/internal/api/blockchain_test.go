package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// ── anchorToBlockchain ─────────────────────────────────────────────────────

func TestAnchorToBlockchain_SkipsZeroAddress(t *testing.T) {
	cfg := &Config{
		TrustRegistryAddress: "0x0000000000000000000000000000000000000000",
		PolygonRPCURL:        "http://unused",
	}
	txHash, err := anchorToBlockchain(cfg, "proof", "user", "VERIFIED")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if txHash != "" {
		t.Errorf("txHash should be empty for zero address, got %q", txHash)
	}
}

func TestAnchorToBlockchain_WithMockRPC(t *testing.T) {
	srv := newMockAnchorRPC(t, nil)
	defer srv.Close()

	cfg := &Config{
		TrustRegistryAddress: "0x1234567890abcdef1234567890abcdef12345678",
		PolygonRPCURL:        srv.URL,
	}
	txHash, err := anchorToBlockchain(cfg, "proofhash", "user1", "VERIFIED")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if txHash == "" {
		t.Error("txHash should not be empty")
	}
	if !strings.HasPrefix(txHash, "0x") {
		t.Errorf("txHash should have 0x prefix, got %q", txHash)
	}
}

func TestAnchorToBlockchain_FallbackRPC(t *testing.T) {
	// Primary always errors
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer primarySrv.Close()

	// Fallback succeeds
	fallbackSrv := newMockAnchorRPC(t, nil)
	defer fallbackSrv.Close()

	cfg := &Config{
		TrustRegistryAddress:  "0x1234567890abcdef1234567890abcdef12345678",
		PolygonRPCURL:         primarySrv.URL,
		PolygonFallbackRPCURL: fallbackSrv.URL,
	}
	txHash, err := anchorToBlockchain(cfg, "proofhash", "user1", "VERIFIED")
	if err != nil {
		t.Fatalf("expected fallback to succeed, got error: %v", err)
	}
	if txHash == "" {
		t.Error("txHash should not be empty from fallback")
	}
}

// ── anchorWithRetry ────────────────────────────────────────────────────────

func TestAnchorWithRetry_SucceedsOnFirstAttempt(t *testing.T) {
	srv := newMockAnchorRPC(t, nil)
	defer srv.Close()

	txHash, err := anchorWithRetry(srv.URL, "0xContract", "proof", "user", "VERIFIED")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if txHash == "" {
		t.Error("txHash should not be empty")
	}
}

func TestAnchorWithRetry_RetriesOnFailure(t *testing.T) {
	var calls atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n < 3 {
			// Fail the first 2 attempts
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// Succeed on 3rd attempt
		type rpcResp struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      int             `json:"id"`
			Result  json.RawMessage `json:"result"`
		}
		resp := rpcResp{JSONRPC: "2.0", ID: 1, Result: json.RawMessage(`"0x"`)}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	txHash, err := anchorWithRetry(srv.URL, "0xContract", "proof", "user", "VERIFIED")
	if err != nil {
		t.Fatalf("expected retry to succeed, got: %v", err)
	}
	if txHash == "" {
		t.Error("txHash should not be empty after successful retry")
	}
	if c := calls.Load(); c < 3 {
		t.Errorf("expected at least 3 calls (2 failures + 1 success), got %d", c)
	}
}

func TestAnchorWithRetry_AllAttemptsFail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := anchorWithRetry(srv.URL, "0xContract", "proof", "user", "VERIFIED")
	if err == nil {
		t.Fatal("expected error when all attempts fail")
	}
	if !strings.Contains(err.Error(), "all") {
		t.Errorf("error should mention all attempts failed, got: %v", err)
	}
}

// ── helpers ────────────────────────────────────────────────────────────────

// newMockAnchorRPC creates a mock JSON-RPC server that returns a successful
// eth_call response. The optional override func can customise behaviour.
func newMockAnchorRPC(t *testing.T, override func(method string) (json.RawMessage, error)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		type rpcReq struct {
			Method string `json:"method"`
		}
		var req rpcReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", 400)
			return
		}

		type rpcResp struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      int             `json:"id"`
			Result  json.RawMessage `json:"result"`
		}

		result := json.RawMessage(`"0x"`)
		if override != nil {
			if res, err := override(req.Method); err == nil {
				result = res
			}
		}

		resp := rpcResp{JSONRPC: "2.0", ID: 1, Result: result}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}
