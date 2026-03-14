"use client";

import { motion } from "framer-motion";
import { useBlockchain } from "@/hooks/useBlockchain";

interface BlockchainLedgerProps {
  txHash: string | null;
}

/**
 * BlockchainLedger — Polygon Amoy transaction explorer widget.
 *
 * Replaces the legacy BlockchainExplorer component with:
 *   - Live block height fetched via useBlockchain hook
 *   - Animated block-confirmation ticker
 *   - Contract name correctly shown as KavachaTrustRegistry
 */
export default function BlockchainLedger({ txHash }: BlockchainLedgerProps) {
  const { blockNumber, isConnected } = useBlockchain();
  const explorerUrl = txHash ? `https://amoy.polygonscan.com/tx/${txHash}` : null;

  return (
    <div className="space-y-3">
      {/* Chain status */}
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${isConnected ? "bg-purple-500 animate-pulse" : "bg-gray-600"}`} />
        <span className="text-xs text-gray-400">Polygon Amoy Testnet</span>
        {blockNumber !== null && (
          <motion.span
            key={blockNumber}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="ml-auto text-xs text-purple-400 font-mono"
          >
            #{blockNumber.toLocaleString()}
          </motion.span>
        )}
      </div>

      {/* Chain metadata */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Chain ID</span>
          <span className="text-purple-400">80002</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Contract</span>
          <span className="text-vajra-accent">KavachaTrustRegistry</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Status</span>
          <span className={txHash ? "text-vajra-green" : "text-gray-500"}>
            {txHash ? "Anchored" : "Pending"}
          </span>
        </div>
      </div>

      {/* TX hash panel */}
      {txHash ? (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 p-3 bg-vajra-700/30 rounded-lg border border-purple-500/20"
        >
          <p className="text-xs text-gray-400 mb-1">Transaction Hash</p>
          <p className="text-xs text-purple-400 font-mono break-all">
            {txHash.slice(0, 18)}…{txHash.slice(-8)}
          </p>
          {explorerUrl && (
            <a
              href={explorerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-flex items-center gap-1 text-xs text-vajra-accent hover:underline"
            >
              View on Polygonscan ↗
            </a>
          )}
        </motion.div>
      ) : (
        <div className="mt-3 p-3 bg-vajra-700/20 rounded-lg border border-vajra-600/20">
          <p className="text-xs text-gray-500">
            Proof will be anchored to KavachaTrustRegistry after verification…
          </p>
        </div>
      )}

      {/* Block animation */}
      <div className="flex gap-1 mt-2">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="flex-1 h-6 rounded-sm bg-vajra-700/40 border border-vajra-600/20"
            animate={txHash ? { backgroundColor: ["#1a1a4a", "#6b21a8", "#1a1a4a"] } : {}}
            transition={{ delay: i * 0.1, duration: 1, repeat: txHash ? 3 : 0 }}
          />
        ))}
      </div>
    </div>
  );
}
