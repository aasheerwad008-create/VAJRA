"use client";

import { motion } from "framer-motion";

interface ZKProofStatusProps {
  proofHash: string | null;
  isVerified: boolean;
}

export default function ZKProofStatus({ proofHash, isVerified }: ZKProofStatusProps) {
  const steps = [
    { id: "biometric", label: "Biometric Commitment", done: !!proofHash },
    { id: "speaker", label: "Speaker Verification", done: isVerified },
    { id: "liveness", label: "Liveness Check", done: isVerified },
    { id: "zk", label: "ZK Proof Generated", done: !!proofHash },
  ];

  return (
    <div className="space-y-3">
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

      {proofHash && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 p-3 bg-vajra-700/30 rounded-lg border border-vajra-green/20"
        >
          <p className="text-xs text-gray-400 mb-1">Proof Hash</p>
          <p className="text-xs text-vajra-green font-mono break-all">
            {proofHash.slice(0, 32)}…
          </p>
          <div className="mt-2 flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-vajra-green animate-pulse" />
            <span className="text-xs text-vajra-green">ZK Proof Valid</span>
          </div>
        </motion.div>
      )}

      {!proofHash && (
        <div className="mt-4 p-3 bg-vajra-700/20 rounded-lg border border-vajra-600/20">
          <p className="text-xs text-gray-500">Awaiting verification to generate ZK proof…</p>
        </div>
      )}
    </div>
  );
}
