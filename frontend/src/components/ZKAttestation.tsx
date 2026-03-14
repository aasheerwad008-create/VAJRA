"use client";

import { motion } from "framer-motion";

interface ZKAttestationProps {
  proofHash: string | null;
  isVerified: boolean;
}

/**
 * ZKAttestation — ZK proof status panel with animated step indicators
 * and a live proof hash matrix display.
 *
 * Replaces the legacy ZKProofStatus component with:
 *   - A 4-step attestation pipeline with animated transitions
 *   - A "hash matrix" byte grid when the proof is available
 *   - Clear RISC Zero / STARK proof branding
 */
export default function ZKAttestation({ proofHash, isVerified }: ZKAttestationProps) {
  const steps = [
    { id: "biometric",  label: "Biometric Commitment",  done: !!proofHash },
    { id: "speaker",    label: "Speaker Verification",   done: isVerified },
    { id: "liveness",   label: "Liveness Check",         done: isVerified },
    { id: "zk",         label: "STARK Proof Generated",  done: !!proofHash },
  ];

  // Derive a display-friendly byte matrix from the proof hash
  const hashBytes: string[] = proofHash
    ? Array.from({ length: 32 }, (_, i) =>
        proofHash.slice(i * 2, i * 2 + 2).toUpperCase()
      )
    : [];

  return (
    <div className="space-y-4">
      {/* Pipeline steps */}
      {steps.map((step, i) => (
        <motion.div
          key={step.id}
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1 }}
        >
          <motion.div
            className={`w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
              step.done
                ? "border-vajra-green bg-vajra-green/20"
                : "border-vajra-600"
            }`}
            animate={step.done ? { scale: [1, 1.2, 1] } : {}}
            transition={{ duration: 0.3 }}
          >
            {step.done && (
              <svg className="w-2 h-2 text-vajra-green" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </motion.div>
          <span className={`text-xs ${step.done ? "text-vajra-green" : "text-gray-500"}`}>
            {step.label}
          </span>
        </motion.div>
      ))}

      {/* Proof hash matrix */}
      {proofHash && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 p-3 bg-vajra-700/30 rounded-lg border border-vajra-green/20"
        >
          <p className="text-xs text-gray-400 mb-2">STARK Proof Hash — 256-bit commitment</p>
          <div className="grid grid-cols-8 gap-0.5 font-mono">
            {hashBytes.map((byte, i) => (
              <motion.span
                key={i}
                className="text-[9px] text-vajra-green/80 bg-vajra-800/60 rounded px-0.5 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.01 }}
              >
                {byte}
              </motion.span>
            ))}
          </div>
          <div className="mt-2 flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-vajra-green animate-pulse" />
            <span className="text-xs text-vajra-green">ZK Attestation Valid</span>
          </div>
        </motion.div>
      )}

      {!proofHash && (
        <div className="mt-2 p-3 bg-vajra-700/20 rounded-lg border border-vajra-600/20">
          <p className="text-xs text-gray-500">Awaiting verification to generate STARK proof…</p>
        </div>
      )}
    </div>
  );
}
