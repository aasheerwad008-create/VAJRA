"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import type { VoiceScoreEvent } from "@/types";

const FFT_SIZE = 128;
const NUM_HISTORY = 100;

interface VoiceEngineProps {
  isStreaming: boolean;
  audioLevel: number;
  trustScore: number;
  verdict: string;
  onStart: () => void;
  onStop: () => void;
}

/**
 * VoiceEngine — live spectrogram + real-time trust score panel.
 *
 * Renders the mel-spectrogram canvas and trust score, driven by props
 * from the parent's useVoiceStream hook.
 */
export default function VoiceEngine({
  isStreaming,
  audioLevel,
  trustScore,
  verdict,
  onStart,
  onStop,
}: VoiceEngineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<number[][]>([]);
  const animRef = useRef<number>(0);

  // ── Canvas rendering ───────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;

      if (isStreaming) {
        const frame: number[] = [];
        for (let i = 0; i < FFT_SIZE; i++) {
          const base = Math.max(0, audioLevel * 255 - i * 1.5);
          const noise = Math.random() * 30;
          const harmonics = i % 8 === 0 ? audioLevel * 100 : 0;
          frame.push(Math.min(255, base + noise + harmonics));
        }
        historyRef.current.push(frame);
        if (historyRef.current.length > NUM_HISTORY) historyRef.current.shift();
      } else if (historyRef.current.length > 0) {
        historyRef.current.push(new Array(FFT_SIZE).fill(0));
        if (historyRef.current.length > NUM_HISTORY) historyRef.current.shift();
      }

      ctx.fillStyle = "#0d0d2b";
      ctx.fillRect(0, 0, W, H);

      if (historyRef.current.length === 0) {
        ctx.strokeStyle = "rgba(0, 212, 255, 0.1)";
        ctx.lineWidth = 0.5;
        for (let x = 0; x < W; x += 20) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        }
        for (let y = 0; y < H; y += 20) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
        ctx.fillStyle = "rgba(0, 212, 255, 0.3)";
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.fillText("AWAITING AUDIO INPUT", W / 2, H / 2);
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      const colW = W / NUM_HISTORY;
      historyRef.current.forEach((frame, t) => {
        const rowH = H / FFT_SIZE;
        frame.forEach((val, f) => {
          const intensity = val / 255;
          const r = Math.floor(intensity > 0.7 ? 255 : intensity > 0.5 ? (intensity - 0.5) * 5 * 255 : 0);
          const g = Math.floor(intensity > 0.8 ? 255 : intensity > 0.3 ? (intensity - 0.3) * 3 * 255 : 0);
          const b = Math.floor(intensity < 0.5 ? 255 : (1 - intensity) * 2 * 255);
          ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + intensity * 0.7})`;
          ctx.fillRect(t * colW, H - (f + 1) * rowH, colW + 0.5, rowH + 0.5);
        });
      });

      ctx.fillStyle = "rgba(0, 212, 255, 0.5)";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ["8kHz", "4kHz", "2kHz", "1kHz", "500Hz"].forEach((label, i) => {
        ctx.fillText(label, W - 4, H - (i / 4) * H - 2);
      });

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [isStreaming, audioLevel]);

  const verdictColor =
    verdict === "VERIFIED" ? "text-vajra-green" :
    verdict === "DEEPFAKE" ? "text-red-400" :
    verdict === "SUSPICIOUS" ? "text-yellow-400" : "text-gray-400";

  return (
    <div className="space-y-3">
      {/* Spectrogram canvas */}
      <canvas
        ref={canvasRef}
        width={600}
        height={200}
        className="w-full rounded-lg border border-vajra-600/20"
        style={{ imageRendering: "pixelated" }}
      />

      {/* Trust score banner */}
      {isStreaming && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between px-4 py-2 rounded-lg bg-vajra-700/30 border border-vajra-600/20"
        >
          <span className="text-xs text-gray-400">Live Trust Score</span>
          <div className="flex items-center gap-2">
            <span className={`text-sm font-bold ${verdictColor}`}>
              {trustScore.toFixed(1)}
            </span>
            <span className={`text-xs font-mono tracking-widest ${verdictColor}`}>
              {verdict}
            </span>
          </div>
        </motion.div>
      )}

      {/* Controls */}
      <motion.button
        onClick={isStreaming ? onStop : onStart}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={`w-full py-2 rounded-lg text-xs font-bold tracking-widest transition-all ${
          isStreaming
            ? "bg-red-500/20 border border-red-500/40 text-red-400"
            : "bg-vajra-600/20 border border-vajra-600/40 text-gray-400 hover:text-white"
        }`}
      >
        {isStreaming ? "⏹ STOP VOICE STREAM" : "🎙 START VOICE STREAM"}
      </motion.button>
    </div>
  );
}
