"use client";

import { useEffect, useRef } from "react";

interface SpectrogramViewProps {
  isActive: boolean;
  audioLevel: number;
}

const FFT_SIZE = 128;
const NUM_HISTORY = 100;

export default function SpectrogramView({ isActive, audioLevel }: SpectrogramViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<number[][]>([]);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;

      // Generate synthetic spectrogram data when streaming
      if (isActive) {
        const frame: number[] = [];
        for (let i = 0; i < FFT_SIZE; i++) {
          const base = Math.max(0, audioLevel * 255 - i * 1.5);
          const noise = Math.random() * 30;
          const harmonics =
            i % 8 === 0 ? audioLevel * 100 : 0; // harmonic peaks
          frame.push(Math.min(255, base + noise + harmonics));
        }
        historyRef.current.push(frame);
        if (historyRef.current.length > NUM_HISTORY) {
          historyRef.current.shift();
        }
      } else if (historyRef.current.length > 0) {
        // Fade out
        historyRef.current.push(new Array(FFT_SIZE).fill(0));
        if (historyRef.current.length > NUM_HISTORY) {
          historyRef.current.shift();
        }
      }

      // Clear
      ctx.fillStyle = "#0d0d2b";
      ctx.fillRect(0, 0, W, H);

      if (historyRef.current.length === 0) {
        // Draw idle grid
        ctx.strokeStyle = "rgba(0, 212, 255, 0.1)";
        ctx.lineWidth = 0.5;
        for (let x = 0; x < W; x += 20) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, H);
          ctx.stroke();
        }
        for (let y = 0; y < H; y += 20) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(W, y);
          ctx.stroke();
        }
        ctx.fillStyle = "rgba(0, 212, 255, 0.3)";
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.fillText("AWAITING AUDIO INPUT", W / 2, H / 2);
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      // Draw spectrogram columns
      const colW = W / NUM_HISTORY;
      historyRef.current.forEach((frame, t) => {
        const rowH = H / FFT_SIZE;
        frame.forEach((val, f) => {
          const intensity = val / 255;
          // Color map: blue → cyan → green → yellow → red
          const r = Math.floor(intensity > 0.7 ? 255 : intensity > 0.5 ? (intensity - 0.5) * 5 * 255 : 0);
          const g = Math.floor(
            intensity > 0.8
              ? 255
              : intensity > 0.3
              ? (intensity - 0.3) * 3 * 255
              : 0
          );
          const b = Math.floor(
            intensity < 0.5 ? 255 : (1 - intensity) * 2 * 255
          );
          ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + intensity * 0.7})`;
          ctx.fillRect(
            t * colW,
            H - (f + 1) * rowH,
            colW + 0.5,
            rowH + 0.5
          );
        });
      });

      // Frequency axis labels
      ctx.fillStyle = "rgba(0, 212, 255, 0.5)";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ["8kHz", "4kHz", "2kHz", "1kHz", "500Hz"].forEach((label, i) => {
        const y = H - (i / 4) * H;
        ctx.fillText(label, W - 4, y - 2);
      });

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [isActive, audioLevel]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={200}
      className="w-full rounded-lg border border-vajra-600/20"
      style={{ imageRendering: "pixelated" }}
    />
  );
}
