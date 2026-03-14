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
