"use client";

import { motion } from "framer-motion";

interface ThreatScoreProps {
  score: number;
  verdict: string;
}

/**
 * ThreatScore — animated threat gauge component.
 *
 * Enhanced version of the legacy ThreatGauge with:
 *   - Glowing colour-coded arc (green / yellow / red)
 *   - Score label and verdict badge
 *   - Smooth animated needle transition
 */
export default function ThreatScore({ score, verdict }: ThreatScoreProps) {
  const clamped = Math.max(0, Math.min(100, score));
  const angle   = (clamped / 100) * 180 - 90;

  const color =
    verdict === "VERIFIED"   ? "#00ff88" :
    verdict === "DEEPFAKE"   ? "#ff3366" :
    verdict === "SUSPICIOUS" ? "#ffcc00" : "#00d4ff";

  const cx = 100, cy = 100, r = 70;

  const polar = (deg: number) => {
    const rad = ((deg - 90) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const start     = polar(-180);
  const end       = polar(0);
  const scoreEnd  = polar(-180 + clamped * 1.8);

  const verdictLabel =
    verdict === "VERIFIED"   ? "✓ VERIFIED" :
    verdict === "DEEPFAKE"   ? "✗ DEEPFAKE" :
    verdict === "SUSPICIOUS" ? "⚠ SUSPICIOUS" : "– IDLE";

  return (
    <div className="space-y-2 flex flex-col items-center">
      <svg viewBox="0 0 200 115" className="w-full max-w-xs">
        {/* Background arc */}
        <path
          d={`M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`}
          fill="none" stroke="#1a1a4a" strokeWidth="12" strokeLinecap="round"
        />
        {/* Score arc */}
        <motion.path
          d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${clamped > 50 ? 1 : 0} 1 ${scoreEnd.x} ${scoreEnd.y}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: clamped / 100 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
        {/* Needle */}
        <motion.line
          x1={cx} y1={cy}
          x2={cx + r * 0.7 * Math.cos(((angle - 90) * Math.PI) / 180)}
          y2={cy + r * 0.7 * Math.sin(((angle - 90) * Math.PI) / 180)}
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${color})` }}
        />
        {/* Centre dot */}
        <circle cx={cx} cy={cy} r="4" fill={color} />
        {/* Score text */}
        <text x={cx} y={cy + 20} textAnchor="middle" fontSize="24" fontWeight="bold" fill={color} fontFamily="monospace">
          {clamped.toFixed(0)}
        </text>
        {/* Axis labels */}
        <text x="22"  y="105" fontSize="9" fill="#4a5568">0</text>
        <text x="178" y="105" fontSize="9" fill="#4a5568">100</text>
        <text x="95"  y="18"  fontSize="9" fill="#4a5568">50</text>
      </svg>

      {/* Verdict badge */}
      <motion.div
        key={verdict}
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="px-3 py-1 rounded-full text-xs font-bold tracking-widest"
        style={{ color, border: `1px solid ${color}40`, background: `${color}10` }}
      >
        {verdictLabel}
      </motion.div>
    </div>
  );
}
