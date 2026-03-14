"use client";

import { motion } from "framer-motion";

interface StatusBarProps {
  isStreaming: boolean;
  audioLevel: number;
}

export default function StatusBar({ isStreaming, audioLevel }: StatusBarProps) {
  const bars = 8;

  return (
    <div className="flex items-center gap-4">
      {/* Audio level visualizer */}
      <div className="flex items-end gap-0.5 h-6">
        {[...Array(bars)].map((_, i) => {
          const threshold = (i + 1) / bars;
          const active = isStreaming && audioLevel >= threshold;
          return (
            <motion.div
              key={i}
              className={`w-1.5 rounded-sm transition-colors ${
                active ? "bg-vajra-accent" : "bg-vajra-700"
              }`}
              animate={{
                height: active
                  ? `${50 + i * 6}%`
                  : "25%",
              }}
              transition={{ duration: 0.1 }}
            />
          );
        })}
      </div>

      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <motion.div
          className={`w-2 h-2 rounded-full ${isStreaming ? "bg-vajra-accent" : "bg-gray-600"}`}
          animate={isStreaming ? { scale: [1, 1.3, 1], opacity: [1, 0.7, 1] } : {}}
          transition={{ duration: 0.8, repeat: Infinity }}
        />
        <span className={`text-xs tracking-widest ${isStreaming ? "text-vajra-accent" : "text-gray-500"}`}>
          {isStreaming ? "LIVE" : "STANDBY"}
        </span>
      </div>

      {/* Layer status */}
      <div className="hidden md:flex items-center gap-2 text-xs">
        {["L1", "L2", "L3"].map((label) => (
          <div
            key={label}
            className={`px-2 py-0.5 rounded border text-xs ${
              isStreaming
                ? "border-vajra-accent/40 text-vajra-accent bg-vajra-accent/10"
                : "border-vajra-700 text-gray-600"
            }`}
          >
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
