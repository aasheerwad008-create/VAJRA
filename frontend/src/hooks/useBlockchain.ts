"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getContract,
  getProvider,
  type KavachaAnchorArgs,
  type RegistryRecord,
} from "@/lib/blockchain";

interface UseBlockchainReturn {
  /** True when the Ethereum provider is connected. */
  isConnected: boolean;
  /** Current block number on Polygon Amoy (polled every 12 s). */
  blockNumber: number | null;
  /** Most recently confirmed transaction hash. */
  txHash: string | null;
  /**
   * Anchor a ZK proof commitment to the KavachaTrustRegistry contract.
   * Returns the transaction hash on success.
   */
  anchorProof: (args: KavachaAnchorArgs) => Promise<string | null>;
  /** Fetch the latest on-chain record for the given identity commitment. */
  getRecord: (commitment: string) => Promise<RegistryRecord | null>;
  error: string | null;
}

/**
 * useBlockchain — Ethers.js hook for KavachaTrustRegistry on Polygon Amoy.
 *
 * Provides reactive block-height polling, proof anchoring, and record
 * retrieval via the KavachaTrustRegistry smart contract.
 */
export function useBlockchain(): UseBlockchainReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [blockNumber, setBlockNumber] = useState<number | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Block-height polling ─────────────────────────────────────────────
  const pollBlockNumber = useCallback(async () => {
    try {
      const provider = getProvider();
      if (!provider) return;
      const num = await provider.getBlockNumber();
      setBlockNumber(num);
      setIsConnected(true);
      setError(null);
    } catch {
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    pollBlockNumber();
    intervalRef.current = setInterval(pollBlockNumber, 12_000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [pollBlockNumber]);

  // ── Proof anchoring ──────────────────────────────────────────────────
  const anchorProof = useCallback(
    async (args: KavachaAnchorArgs): Promise<string | null> => {
      try {
        const contract = await getContract();
        if (!contract) {
          throw new Error("Contract not available — check TRUST_REGISTRY_ADDRESS");
        }

        const tx = await contract.anchorVerification(
          args.identityCommitment,
          args.proofHash,
          args.txRefHash,
          args.verified,
          args.verdict
        );
        const receipt = await tx.wait();
        const hash: string = receipt?.hash ?? tx.hash;
        setTxHash(hash);
        setError(null);
        return hash;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Blockchain error";
        setError(msg);
        return null;
      }
    },
    []
  );

  // ── Record retrieval ─────────────────────────────────────────────────
  const getRecord = useCallback(
    async (commitment: string): Promise<RegistryRecord | null> => {
      try {
        const contract = await getContract();
        if (!contract) return null;

        const record = await contract.getLatestRecord(commitment);
        return {
          proofHash:  record.proofHash,
          txHash:     record.txHash,
          timestamp:  Number(record.timestamp),
          verified:   record.verified,
          verdict:    record.verdict,
        };
      } catch {
        return null;
      }
    },
    []
  );

  return { isConnected, blockNumber, txHash, anchorProof, getRecord, error };
}
