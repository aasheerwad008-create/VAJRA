package polygon

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ── NewClient ──────────────────────────────────────────────────────────────

func TestNewClient(t *testing.T) {
	c := NewClient("https://rpc.example.com", "0xABCD")
	if c.rpcURL != "https://rpc.example.com" {
		t.Errorf("rpcURL = %q, want %q", c.rpcURL, "https://rpc.example.com")
	}
	if c.contractAddress != "0xABCD" {
		t.Errorf("contractAddress = %q, want %q", c.contractAddress, "0xABCD")
	}
	if c.httpClient == nil {
		t.Error("httpClient must not be nil")
	}
}

// ── deterministicTxHash ────────────────────────────────────────────────────

func TestDeterministicTxHash_Determinism(t *testing.T) {
	h1 := deterministicTxHash("proof1", "user1", "VERIFIED")
	h2 := deterministicTxHash("proof1", "user1", "VERIFIED")
	if h1 != h2 {
		t.Errorf("same inputs must produce same hash: %q vs %q", h1, h2)
	}
}

func TestDeterministicTxHash_DifferentInputs(t *testing.T) {
	h1 := deterministicTxHash("proof1", "user1", "VERIFIED")
	h2 := deterministicTxHash("proof2", "user1", "VERIFIED")
	h3 := deterministicTxHash("proof1", "user2", "VERIFIED")
	h4 := deterministicTxHash("proof1", "user1", "REJECTED")

	if h1 == h2 {
		t.Error("different proofHash must produce different hashes")
	}
	if h1 == h3 {
		t.Error("different userID must produce different hashes")
	}
	if h1 == h4 {
		t.Error("different verdict must produce different hashes")
	}
}

func TestDeterministicTxHash_HexPrefix(t *testing.T) {
	h := deterministicTxHash("a", "b", "c")
	if !strings.HasPrefix(h, "0x") {
		t.Errorf("hash must start with 0x, got %q", h)
	}
	// "0x" + 64 hex chars = 66 total
	if len(h) != 66 {
		t.Errorf("hash length = %d, want 66", len(h))
	}
}

// ── buildAnchorCallData ────────────────────────────────────────────────────

func TestBuildAnchorCallData_HexPrefix(t *testing.T) {
	data := buildAnchorCallData("proofhash", "userid", "VERIFIED")
	if !strings.HasPrefix(data, "0x") {
		t.Errorf("calldata must start with 0x, got prefix: %q", data[:10])
	}
}

func TestBuildAnchorCallData_Selector(t *testing.T) {
	data := buildAnchorCallData("p", "u", "v")
	// Selector is "0xa1b2c3d4" → after "0x" the next 8 chars are the selector.
	selector := data[2:10]
	if selector != "a1b2c3d4" {
		t.Errorf("selector = %q, want %q", selector, "a1b2c3d4")
	}
}

func TestBuildAnchorCallData_ContainsPaddedArgs(t *testing.T) {
	data := buildAnchorCallData("abc", "def", "ghi")
	// After selector (10 chars), expect 3 × 64-char padded fields = 192 chars
	payload := data[10:]
	if len(payload) != 192 {
		t.Errorf("payload length = %d, want 192 (3×64)", len(payload))
	}
}

// ── TransactionReceipt.IsSuccess ──────────────────────────────────────────

func TestTransactionReceipt_IsSuccess(t *testing.T) {
	tests := []struct {
		status string
		want   bool
	}{
		{"0x1", true},
		{"0x0", false},
		{"", false},
		{"0x2", false},
	}
	for _, tt := range tests {
		r := &TransactionReceipt{Status: tt.status}
		if got := r.IsSuccess(); got != tt.want {
			t.Errorf("IsSuccess(%q) = %v, want %v", tt.status, got, tt.want)
		}
	}
}

// ── Mock RPC server helpers ────────────────────────────────────────────────

// mockRPCServer creates an httptest.Server that responds to JSON-RPC calls.
// The handler func receives the method and params and returns the result to embed.
func mockRPCServer(t *testing.T, handler func(method string, params json.RawMessage) (json.RawMessage, *rpcError)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Logf("mock: decode error: %v", err)
			http.Error(w, "bad request", 400)
			return
		}
		paramsRaw, _ := json.Marshal(req.Params)
		result, rpcErr := handler(req.Method, paramsRaw)

		resp := rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: result, Error: rpcErr}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}

// ── BlockNumber via mock ───────────────────────────────────────────────────

func TestBlockNumber_Mock(t *testing.T) {
	srv := mockRPCServer(t, func(method string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		if method != "eth_blockNumber" {
			t.Errorf("unexpected method: %s", method)
		}
		return json.RawMessage(`"0x1a4"`), nil // 420
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0x0")
	bn, err := c.BlockNumber(context.Background())
	if err != nil {
		t.Fatalf("BlockNumber: %v", err)
	}
	if bn != 420 {
		t.Errorf("block number = %d, want 420", bn)
	}
}

// ── GetTransactionReceipt via mock ─────────────────────────────────────────

func TestGetTransactionReceipt_Success(t *testing.T) {
	srv := mockRPCServer(t, func(method string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		receipt := TransactionReceipt{
			TransactionHash: "0xabc",
			BlockNumber:     "0x10",
			Status:          "0x1",
			GasUsed:         "0x5208",
		}
		b, _ := json.Marshal(receipt)
		return b, nil
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0x0")
	r, err := c.GetTransactionReceipt(context.Background(), "0xabc")
	if err != nil {
		t.Fatalf("GetTransactionReceipt: %v", err)
	}
	if r == nil {
		t.Fatal("receipt should not be nil")
	}
	if !r.IsSuccess() {
		t.Error("receipt should be successful")
	}
	if r.TransactionHash != "0xabc" {
		t.Errorf("txHash = %q, want %q", r.TransactionHash, "0xabc")
	}
}

func TestGetTransactionReceipt_NotMined(t *testing.T) {
	srv := mockRPCServer(t, func(_ string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		return json.RawMessage(`null`), nil
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0x0")
	r, err := c.GetTransactionReceipt(context.Background(), "0xpending")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r != nil {
		t.Error("receipt should be nil for unmined tx")
	}
}

// ── IsContractDeployed via mock ────────────────────────────────────────────

func TestIsContractDeployed_True(t *testing.T) {
	srv := mockRPCServer(t, func(method string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		if method != "eth_getCode" {
			t.Errorf("unexpected method: %s", method)
		}
		return json.RawMessage(`"0x6080604052"`), nil
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0xContract")
	deployed, err := c.IsContractDeployed(context.Background())
	if err != nil {
		t.Fatalf("IsContractDeployed: %v", err)
	}
	if !deployed {
		t.Error("expected contract to be deployed")
	}
}

func TestIsContractDeployed_False(t *testing.T) {
	srv := mockRPCServer(t, func(_ string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		return json.RawMessage(`"0x"`), nil
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0xEmpty")
	deployed, err := c.IsContractDeployed(context.Background())
	if err != nil {
		t.Fatalf("IsContractDeployed: %v", err)
	}
	if deployed {
		t.Error("expected contract NOT to be deployed")
	}
}

// ── RPC error handling ─────────────────────────────────────────────────────

func TestBlockNumber_RPCError(t *testing.T) {
	srv := mockRPCServer(t, func(_ string, _ json.RawMessage) (json.RawMessage, *rpcError) {
		return nil, &rpcError{Code: -32600, Message: "invalid request"}
	})
	defer srv.Close()

	c := NewClient(srv.URL, "0x0")
	_, err := c.BlockNumber(context.Background())
	if err == nil {
		t.Fatal("expected error for RPC error response")
	}
	if !strings.Contains(err.Error(), "invalid request") {
		t.Errorf("error should contain RPC message, got: %v", err)
	}
}
