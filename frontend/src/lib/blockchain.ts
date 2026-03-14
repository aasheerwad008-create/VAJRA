/**
 * VAJRA Frontend — Ethers.js integration + KavachaTrustRegistry ABI.
 *
 * Provides:
 *   - getProvider()   — read-only JSON-RPC provider for Polygon Amoy
 *   - getSigner()     — wraps the injected wallet (MetaMask / WalletConnect)
 *   - getContract()   — typed contract instance for KavachaTrustRegistry
 *
 * Usage:
 *   import { getContract } from "@/lib/blockchain";
 *   const contract = await getContract();
 *   const tx = await contract.anchorVerification(...);
 */

import { ethers } from "ethers";

// ── Network constants ──────────────────────────────────────────────────────

export const AMOY_CHAIN_ID = 80002;
export const AMOY_RPC_URL =
  process.env.NEXT_PUBLIC_POLYGON_RPC_URL ??
  "https://rpc-amoy.polygon.technology";
export const CONTRACT_ADDRESS =
  process.env.NEXT_PUBLIC_TRUST_REGISTRY_ADDRESS ?? "";

// ── ABI (minimal — only the functions used by the frontend) ───────────────

export const KAVACH_ABI = [
  "function anchorVerification(bytes32 identityCommitment, bytes32 proofHash, bytes32 txRefHash, bool verified, string calldata verdict) external",
  "function getLatestRecord(bytes32 identityCommitment) external view returns (tuple(bytes32 proofHash, bytes32 txHash, uint256 timestamp, bool verified, string verdict))",
  "function getAllRecords(bytes32 identityCommitment) external view returns (tuple(bytes32 proofHash, bytes32 txHash, uint256 timestamp, bool verified, string verdict)[])",
  "function isVerified(bytes32 identityCommitment) external view returns (bool)",
  "function recordCount(bytes32 identityCommitment) external view returns (uint256)",
  "function totalVerifications() external view returns (uint256)",
  "function totalFraudAttempts() external view returns (uint256)",
  "function VERSION() external view returns (string)",
  "event IdentityVerified(bytes32 indexed identityCommitment, bytes32 proofHash, uint256 timestamp)",
  "event FraudAttemptDetected(bytes32 indexed identityCommitment, bytes32 proofHash, string verdict, uint256 timestamp)",
] as const;

// ── Types ──────────────────────────────────────────────────────────────────

export interface RegistryRecord {
  proofHash:  string;   // bytes32 hex
  txHash:     string;   // bytes32 hex
  timestamp:  number;   // Unix timestamp
  verified:   boolean;
  verdict:    string;
}

export interface KavachaAnchorArgs {
  identityCommitment: string;   // bytes32 hex — SHA-256(userId:nullifier)
  proofHash:          string;   // bytes32 hex — ZK proof commitment
  txRefHash:          string;   // bytes32 hex — off-chain reference
  verified:           boolean;
  verdict:            string;   // "VERIFIED" | "SUSPICIOUS" | "DEEPFAKE"
}

// ── Provider / signer ──────────────────────────────────────────────────────

let _provider: ethers.JsonRpcProvider | null = null;

/**
 * Return a read-only JSON-RPC provider for Polygon Amoy.
 * Returns null if running in a non-browser environment (SSR).
 */
export function getProvider(): ethers.JsonRpcProvider | null {
  if (typeof window === "undefined") return null;

  if (!_provider) {
    _provider = new ethers.JsonRpcProvider(AMOY_RPC_URL, {
      chainId: AMOY_CHAIN_ID,
      name: "polygon-amoy",
    });
  }
  return _provider;
}

/**
 * Return an ethers.js Signer from the injected wallet (e.g., MetaMask).
 * Prompts the user to connect their wallet if not already connected.
 *
 * Returns null if no wallet is available.
 */
export async function getSigner(): Promise<ethers.Signer | null> {
  if (typeof window === "undefined") return null;

  const win = window as typeof window & { ethereum?: ethers.Eip1193Provider };
  if (!win.ethereum) return null;

  const provider = new ethers.BrowserProvider(win.ethereum);
  await provider.send("eth_requestAccounts", []);

  const signer = await provider.getSigner();

  // Ensure we're on Polygon Amoy
  const network = await provider.getNetwork();
  if (Number(network.chainId) !== AMOY_CHAIN_ID) {
    await win.ethereum.request?.({
      method: "wallet_switchEthereumChain",
      params: [{ chainId: `0x${AMOY_CHAIN_ID.toString(16)}` }],
    });
  }

  return signer;
}

/**
 * Return a typed KavachaTrustRegistry contract instance.
 *
 * Uses a read-only provider when no wallet is connected.
 * Uses the injected wallet signer when available (for write operations).
 *
 * Returns null when no contract address is configured.
 */
export async function getContract(): Promise<ethers.Contract | null> {
  if (!CONTRACT_ADDRESS) return null;

  const signer = await getSigner().catch(() => null);
  const runner = signer ?? getProvider();
  if (!runner) return null;

  return new ethers.Contract(CONTRACT_ADDRESS, KAVACH_ABI, runner);
}

// ── Helpers ────────────────────────────────────────────────────────────────

/** Compute the on-chain identity commitment bytes32 from a hex string. */
export function toBytes32(hex: string): string {
  return ethers.zeroPadValue(hex.startsWith("0x") ? hex : `0x${hex}`, 32);
}

/** Format a proof hash bytes32 for display (0x prefix, 8…8). */
export function formatHash(hash: string): string {
  const h = hash.startsWith("0x") ? hash : `0x${hash}`;
  return `${h.slice(0, 10)}…${h.slice(-8)}`;
}
