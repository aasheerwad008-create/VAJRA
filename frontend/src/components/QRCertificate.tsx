"use client";

import { useEffect, useRef, useState } from "react";
import QRCode from "qrcode";
import { motion } from "framer-motion";

interface QRCertificateProps {
  sessionId: string | null;
  proofHash: string | null;
  verdict: string;
  trustScore: number;
}

export default function QRCertificate({
  sessionId,
  proofHash,
  verdict,
  trustScore,
}: QRCertificateProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [generated, setGenerated] = useState(false);

  useEffect(() => {
    if (!sessionId || !proofHash || verdict !== "VERIFIED") {
      setGenerated(false);
      return;
    }

    const certData = JSON.stringify({
      v: "1",
      session: sessionId,
      proof: proofHash.slice(0, 32),
      score: trustScore.toFixed(0),
      verdict,
      ts: new Date().toISOString(),
      issuer: "VAJRA-ZTI-Defense",
    });

    if (canvasRef.current) {
      QRCode.toCanvas(canvasRef.current, certData, {
        width: 160,
        margin: 1,
        color: {
          dark: "#00ff88",
          light: "#0d0d2b",
        },
      })
        .then(() => setGenerated(true))
        .catch(() => setGenerated(false));
    }
  }, [sessionId, proofHash, verdict, trustScore]);

  if (!generated) {
    return (
      <div className="flex flex-col items-center justify-center h-48 gap-4">
        <div className="w-32 h-32 border-2 border-dashed border-vajra-600/40 rounded-lg flex items-center justify-center">
          <span className="text-4xl">🔒</span>
        </div>
        <p className="text-xs text-gray-500 text-center">
          Certificate generated after successful <br />
          VERIFIED verdict
        </p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center gap-4"
    >
      <div className="p-3 bg-vajra-800 rounded-xl border border-vajra-green/30 glow-green">
        <canvas ref={canvasRef} className="rounded-lg" />
      </div>

      <div className="text-center space-y-1">
        <div className="flex items-center gap-2 justify-center">
          <div className="w-2 h-2 rounded-full bg-vajra-green animate-pulse" />
          <span className="text-xs text-vajra-green font-bold tracking-widest">
            IDENTITY VERIFIED
          </span>
        </div>
        <p className="text-xs text-gray-400">
          Session: {sessionId?.slice(0, 12)}…
        </p>
        <p className="text-xs text-gray-400">
          Trust Score: <span className="text-vajra-green">{trustScore.toFixed(0)}/100</span>
        </p>
      </div>

      <button
        onClick={() => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          const link = document.createElement("a");
          link.download = `vajra-cert-${sessionId?.slice(0, 8)}.png`;
          link.href = canvas.toDataURL();
          link.click();
        }}
        className="text-xs text-vajra-accent hover:text-white transition-colors border border-vajra-accent/30 hover:border-vajra-accent/60 px-4 py-2 rounded-lg"
      >
        ↓ Download Certificate
      </button>
    </motion.div>
  );
}
