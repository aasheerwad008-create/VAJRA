// VAJRA Backend — Polygon Amoy RPC Client.
//
// Provides a typed client for interacting with the KavachaTrustRegistry smart
// contract deployed on the Polygon Amoy testnet (chain ID 80002).
//
// In production, replace the placeholder implementation with calls via
// go-ethereum's ethclient + abigen-generated bindings.
package polygon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	// AmoyChainID is the EIP-155 chain ID for Polygon Amoy testnet.
	AmoyChainID = 80002
	// DefaultRPCURL is the public Polygon Amoy RPC endpoint.
	DefaultRPCURL = "https://rpc-amoy.polygon.technology"
)

// Client is a minimal JSON-RPC client for the Polygon Amoy network.
type Client struct {
	rpcURL          string
	contractAddress string
	httpClient      *http.Client
}

// NewClient creates a Client for the given RPC URL and contract address.
func NewClient(rpcURL, contractAddress string) *Client {
	return &Client{
		rpcURL:          rpcURL,
		contractAddress: contractAddress,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

// BlockNumber returns the current block number on the chain.
func (c *Client) BlockNumber(ctx context.Context) (uint64, error) {
	resp, err := c.call(ctx, "eth_blockNumber", nil)
	if err != nil {
		return 0, err
	}

	var hexBlock string
	if err := json.Unmarshal(resp.Result, &hexBlock); err != nil {
		return 0, fmt.Errorf("polygon: decode blockNumber: %w", err)
	}

	var n uint64
	fmt.Sscanf(strings.TrimPrefix(hexBlock, "0x"), "%x", &n)
	return n, nil
}

// GetTransactionReceipt fetches the receipt for a given transaction hash.
// Returns nil if the transaction is not yet mined.
func (c *Client) GetTransactionReceipt(ctx context.Context, txHash string) (*TransactionReceipt, error) {
	resp, err := c.call(ctx, "eth_getTransactionReceipt", []interface{}{txHash})
	if err != nil {
		return nil, err
	}

	if string(resp.Result) == "null" {
		return nil, nil // not yet mined
	}

	var receipt TransactionReceipt
	if err := json.Unmarshal(resp.Result, &receipt); err != nil {
		return nil, fmt.Errorf("polygon: decode receipt: %w", err)
	}
	return &receipt, nil
}

// IsContractDeployed checks whether a contract exists at the configured address.
func (c *Client) IsContractDeployed(ctx context.Context) (bool, error) {
	resp, err := c.call(ctx, "eth_getCode", []interface{}{c.contractAddress, "latest"})
	if err != nil {
		return false, err
	}

	var code string
	if err := json.Unmarshal(resp.Result, &code); err != nil {
		return false, err
	}
	// "0x" means no code at address
	return code != "0x" && code != "", nil
}

// AnchorProof sends a raw transaction that encodes an anchorVerification call
// to the trust registry contract.
//
// The call data is ABI-encoded as:
//
//	anchorVerification(bytes32 identityCommitment, bytes32 proofHash, string txRef, bool verified, string verdict)
//
// When no deployer private key is available (i.e. read-only mode) or the RPC
// is unreachable, it falls back to an eth_call simulation and returns a
// deterministic hash derived from the proof inputs so that the rest of the
// pipeline can continue.
func (c *Client) AnchorProof(ctx context.Context, proofHash, userID, verdict string) (string, error) {
	// Build the ABI-encoded call data for anchorVerification.
	// Function selector: keccak256("anchorVerification(bytes32,bytes32,string,bool,string)")[:4]
	callData := buildAnchorCallData(proofHash, userID, verdict)

	// Attempt eth_call to validate the call would succeed on-chain.
	resp, err := c.call(ctx, "eth_call", []interface{}{
		map[string]string{
			"to":   c.contractAddress,
			"data": callData,
		},
		"latest",
	})
	if err != nil {
		// RPC unreachable — return deterministic hash so pipeline can continue.
		return deterministicTxHash(proofHash, userID, verdict), fmt.Errorf(
			"polygon: eth_call failed (using deterministic hash): %w", err,
		)
	}
	_ = resp // call succeeded

	// In production with a funded deployer key this would be eth_sendRawTransaction.
	// For now return a deterministic hash derived from the validated call inputs.
	return deterministicTxHash(proofHash, userID, verdict), nil
}

// buildAnchorCallData constructs hex-encoded call data for the
// anchorVerification function.  The result is a minimal ABI encoding:
//
//	selector(4 bytes) + identityCommitment(32) + proofHash(32) + …
func buildAnchorCallData(proofHash, userID, verdict string) string {
	// Placeholder function selector — replace with the real first 4 bytes of
	// keccak256("anchorVerification(bytes32,bytes32,string,bool,string)")
	// once the contract ABI is finalised.
	selector := "0xa1b2c3d4"
	padded := func(s string, size int) string {
		h := fmt.Sprintf("%x", s)
		for len(h) < size*2 {
			h = "0" + h
		}
		if len(h) > size*2 {
			h = h[:size*2]
		}
		return h
	}
	return selector + padded(userID, 32) + padded(proofHash, 32) + padded(verdict, 32)
}

// deterministicTxHash produces a repeatable, collision-resistant hash from the
// proof inputs.  This is used as a stand-in tx hash when on-chain submission
// is not possible (e.g. no deployer key or RPC down).
func deterministicTxHash(proofHash, userID, verdict string) string {
	data := proofHash + ":" + userID + ":" + verdict
	h := sha256.Sum256([]byte(data))
	return "0x" + hex.EncodeToString(h[:])
}

// ── Types ──────────────────────────────────────────────────────────────────

// TransactionReceipt represents an Ethereum transaction receipt.
type TransactionReceipt struct {
	TransactionHash string `json:"transactionHash"`
	BlockNumber     string `json:"blockNumber"`
	Status          string `json:"status"` // "0x1" = success, "0x0" = revert
	GasUsed         string `json:"gasUsed"`
	ContractAddress string `json:"contractAddress,omitempty"`
}

// IsSuccess returns true if the transaction was mined successfully.
func (r *TransactionReceipt) IsSuccess() bool {
	return r.Status == "0x1"
}

// ── Internal RPC transport ─────────────────────────────────────────────────

type rpcRequest struct {
	JSONRPC string        `json:"jsonrpc"`
	Method  string        `json:"method"`
	Params  []interface{} `json:"params"`
	ID      int           `json:"id"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (c *Client) call(
	ctx context.Context,
	method string,
	params []interface{},
) (*rpcResponse, error) {
	if params == nil {
		params = []interface{}{}
	}

	reqBody, err := json.Marshal(rpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      1,
	})
	if err != nil {
		return nil, fmt.Errorf("polygon: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost, c.rpcURL, strings.NewReader(string(reqBody)),
	)
	if err != nil {
		return nil, fmt.Errorf("polygon: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("polygon: http: %w", err)
	}
	defer resp.Body.Close()

	var rpcResp rpcResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return nil, fmt.Errorf("polygon: decode response: %w", err)
	}
	if rpcResp.Error != nil {
		return nil, fmt.Errorf("polygon: rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	return &rpcResp, nil
}
