"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

export default function VideoFeed() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shieldActive, setShieldActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraError(null);
    } catch (err) {
      setCameraError("Camera access denied or unavailable");
    }
  };

  const toggleShield = () => {
    setShieldActive((prev) => {
      const next = !prev;
      if (next) {
        intervalRef.current = setInterval(applyAdversarialShield, 1000 / 15); // 15 fps
      } else {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
      return next;
    });
  };

  const applyAdversarialShield = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Apply subtle adversarial perturbation (sinusoidal pattern)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const t = Date.now() / 1000;
    const epsilon = 8; // max perturbation per channel

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const i = (y * canvas.width + x) * 4;
        const perturbation = epsilon * Math.sin(x * 0.1 + t) * Math.cos(y * 0.1 + t);
        data[i] = Math.min(255, Math.max(0, data[i] + perturbation));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + perturbation * 0.7));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + perturbation * 0.3));
      }
    }
    ctx.putImageData(imageData, 0, 0);
  };

  useEffect(() => {
    startCamera();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return (
    <div className="space-y-3">
      <div className="relative rounded-lg overflow-hidden bg-vajra-900 aspect-video">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={`w-full h-full object-cover ${shieldActive ? "hidden" : ""}`}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className={`w-full h-full object-cover ${!shieldActive ? "hidden" : ""}`}
        />
        {cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-vajra-900">
            <div className="text-center">
              <span className="text-4xl">📷</span>
              <p className="text-xs text-gray-500 mt-2">{cameraError}</p>
            </div>
          </div>
        )}

        {/* Shield overlay indicator */}
        {shieldActive && (
          <div className="absolute top-2 right-2 flex items-center gap-1 bg-vajra-green/20 border border-vajra-green/40 rounded-full px-2 py-1">
            <div className="w-1.5 h-1.5 rounded-full bg-vajra-green animate-pulse" />
            <span className="text-xs text-vajra-green">SHIELD ACTIVE</span>
          </div>
        )}

        {/* Scan line effect */}
        <motion.div
          className="absolute inset-x-0 h-px bg-vajra-accent/30 pointer-events-none"
          animate={{ top: ["0%", "100%"] }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />
      </div>

      <div className="flex gap-2">
        <motion.button
          onClick={toggleShield}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className={`flex-1 py-2 rounded-lg text-xs font-bold tracking-widest transition-all ${
            shieldActive
              ? "bg-vajra-green/20 border border-vajra-green/40 text-vajra-green"
              : "bg-vajra-600/20 border border-vajra-600/40 text-gray-400 hover:text-white"
          }`}
        >
          {shieldActive ? "⚡ ADVERSARIAL SHIELD ON" : "🛡 ENABLE SHIELD"}
        </motion.button>
      </div>
    </div>
  );
}
