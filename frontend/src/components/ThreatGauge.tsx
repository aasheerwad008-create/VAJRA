"use client";

import { motion } from "framer-motion";

interface ThreatGaugeProps {
  score: number;
  verdict: string;
}

export default function ThreatGauge({ score, verdict }: ThreatGaugeProps) {
  const clampedScore = Math.max(0, Math.min(100, score));
  const angle = (clampedScore / 100) * 180 - 90; // -90 to +90 degrees

  const color =
    verdict === "VERIFIED"
      ? "#00ff88"
      : verdict === "DEEPFAKE"
      ? "#ff3366"
      : verdict === "SUSPICIOUS"
      ? "#ffcc00"
      : "#00d4ff";

  const cx = 100;
  const cy = 100;
  const r = 70;

  // Arc path helper
  const polarToCartesian = (angle: number) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  };

  const startAngle = -180;
  const endAngle = 0;
  const start = polarToCartesian(startAngle);
  const end = polarToCartesian(endAngle);
  const scoreEnd = polarToCartesian(startAngle + clampedScore * 1.8);

  return (
    <div className="flex justify-center">
      <svg viewBox="0 0 200 110" className="w-full max-w-xs">
        {/* Background arc */}
        <path
          d={`M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`}
          fill="none"
          stroke="#1a1a4a"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Score arc */}
        <motion.path
          d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${clampedScore > 50 ? 1 : 0} 1 ${scoreEnd.x} ${scoreEnd.y}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: clampedScore / 100 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
        {/* Needle */}
        <motion.line
          x1={cx}
          y1={cy}
          x2={cx + r * 0.7 * Math.cos(((angle - 90) * Math.PI) / 180)}
          y2={cy + r * 0.7 * Math.sin(((angle - 90) * Math.PI) / 180)}
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          animate={{ rotate: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 4px ${color})` }}
        />
        {/* Center dot */}
        <circle cx={cx} cy={cy} r="4" fill={color} />
        {/* Score text */}
        <text
          x={cx}
          y={cy + 20}
          textAnchor="middle"
          className="font-mono"
          fontSize="24"
          fontWeight="bold"
          fill={color}
        >
          {clampedScore.toFixed(0)}
        </text>
        {/* Labels */}
        <text x="22" y="105" fontSize="9" fill="#4a5568">0</text>
        <text x="178" y="105" fontSize="9" fill="#4a5568">100</text>
        <text x="95" y="18" fontSize="9" fill="#4a5568">50</text>
      </svg>
    </div>
  );
}
