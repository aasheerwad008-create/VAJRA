"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ThreatGauge from "@/components/ThreatGauge";
import SpectrogramView from "@/components/SpectrogramView";
import ZKProofStatus from "@/components/ZKProofStatus";
import BlockchainExplorer from "@/components/BlockchainExplorer";
import QRCertificate from "@/components/QRCertificate";
import VideoFeed from "@/components/VideoFeed";
import StatusBar from "@/components/StatusBar";
import { useVoiceStream } from "@/hooks/useVoiceStream";
import type { VerificationState } from "@/types";

export default function Dashboard() {
  const [verificationState, setVerificationState] = useState<VerificationState>({
    trustScore: 0,
    verdict: "IDLE",
    components: {},
    proofHash: null,
    txHash: null,
    sessionId: null,
    isVerifying: false,
  });

  const { startStream, stopStream, isStreaming, audioLevel } = useVoiceStream({
    onScore: (score) => {
      setVerificationState((prev) => ({
        ...prev,
        trustScore: score.trust_score,
        verdict: score.verdict,
        components: score.components,
        isVerifying: true,
      }));
    },
  });

  const handleVerify = async () => {
    if (isStreaming) {
      stopStream();
      setVerificationState((prev) => ({ ...prev, isVerifying: false }));
    } else {
      await startStream();
    }
  };

  const verdictColor =
    verificationState.verdict === "VERIFIED"
      ? "text-vajra-green"
      : verificationState.verdict === "DEEPFAKE"
      ? "text-vajra-red"
      : verificationState.verdict === "SUSPICIOUS"
      ? "text-vajra-yellow"
      : "text-gray-400";

  return (
    <main className="min-h-screen grid-bg">
      {/* Header */}
      <header className="border-b border-vajra-600/30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-vajra-accent/20 border border-vajra-accent/40 flex items-center justify-center">
            <span className="text-vajra-accent text-xs font-bold">V</span>
          </div>
          <div>
            <h1 className="text-vajra-accent font-bold text-lg tracking-widest">VAJRA</h1>
            <p className="text-xs text-gray-500 tracking-wider">
              ZERO-TRUST AI IDENTITY DEFENSE
            </p>
          </div>
        </div>
        <StatusBar isStreaming={isStreaming} audioLevel={audioLevel} />
      </header>

      {/* Main grid */}
      <div className="p-6 grid grid-cols-12 gap-4">
        {/* Threat Score Gauge */}
        <motion.div
          className="col-span-12 md:col-span-4 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            Trust Score
          </h2>
          <ThreatGauge score={verificationState.trustScore} verdict={verificationState.verdict} />
          <div className="mt-4 text-center">
            <AnimatePresence mode="wait">
              <motion.div
                key={verificationState.verdict}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className={`text-2xl font-bold tracking-widest ${verdictColor}`}
              >
                {verificationState.verdict}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Component scores */}
          <div className="mt-6 space-y-2">
            {Object.entries(verificationState.components).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="text-gray-400 capitalize">{key.replace(/_/g, " ")}</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-1 bg-vajra-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-vajra-accent rounded-full"
                      animate={{ width: `${val}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <span className="text-vajra-accent w-8 text-right">{val.toFixed(0)}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Live Spectrogram */}
        <motion.div
          className="col-span-12 md:col-span-8 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            Live Spectrogram
          </h2>
          <SpectrogramView isActive={isStreaming} audioLevel={audioLevel} />
        </motion.div>

        {/* Video Feed */}
        <motion.div
          className="col-span-12 md:col-span-6 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            Video Feed — Adversarial Shield
          </h2>
          <VideoFeed />
        </motion.div>

        {/* ZK Proof Status */}
        <motion.div
          className="col-span-12 md:col-span-3 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            ZK Proof
          </h2>
          <ZKProofStatus
            proofHash={verificationState.proofHash}
            isVerified={verificationState.verdict === "VERIFIED"}
          />
        </motion.div>

        {/* Blockchain Explorer */}
        <motion.div
          className="col-span-12 md:col-span-3 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            Chain Anchor
          </h2>
          <BlockchainExplorer txHash={verificationState.txHash} />
        </motion.div>

        {/* QR Certificate */}
        <motion.div
          className="col-span-12 md:col-span-6 bg-vajra-800/50 border border-vajra-600/30 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <h2 className="text-xs tracking-widest text-gray-400 mb-4 uppercase">
            Verification Certificate
          </h2>
          <QRCertificate
            sessionId={verificationState.sessionId}
            proofHash={verificationState.proofHash}
            verdict={verificationState.verdict}
            trustScore={verificationState.trustScore}
          />
        </motion.div>

        {/* Control panel */}
        <motion.div
          className="col-span-12 flex items-center justify-center pt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <motion.button
            onClick={handleVerify}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            className={`px-12 py-4 rounded-xl font-bold tracking-widest text-sm uppercase transition-all ${
              isStreaming
                ? "bg-vajra-red/20 border border-vajra-red/50 text-vajra-red hover:bg-vajra-red/30 glow-red"
                : "bg-vajra-accent/20 border border-vajra-accent/50 text-vajra-accent hover:bg-vajra-accent/30 glow-accent"
            }`}
          >
            {isStreaming ? "⬛ STOP VERIFICATION" : "▶ START LIVE VERIFICATION"}
          </motion.button>
        </motion.div>
      </div>
    </main>
  );
}
