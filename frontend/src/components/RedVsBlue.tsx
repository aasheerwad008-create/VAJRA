"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── Types ──────────────────────────────────────────────────────────────────

interface AttackEvent {
  id: string;
  timestamp: number;
  type: "voice_clone" | "deepfake_visual" | "replay_attack" | "adversarial_inject";
  label: string;
  confidence: number;
}

interface DefenseEvent {
  id: string;
  timestamp: number;
  type: "detected" | "blocked" | "proof_generated" | "anchored";
  label: string;
  detail: string;
  trustScore: number;
}

interface BattleState {
  isRunning: boolean;
  attackEvents: AttackEvent[];
  defenseEvents: DefenseEvent[];
  redScore: number;
  blueScore: number;
  currentAttack: string | null;
  currentDefense: string | null;
}

// ── Attack Scenarios ────────────────────────────────────────────────────────

const ATTACK_SCENARIOS = [
  {
    type: "voice_clone" as const,
    label: "Neural Voice Clone (RVC)",
    description: "AI-generated voice cloning using Real-time Voice Cloning",
    stages: [
      "Extracting voice embedding from target audio...",
      "Synthesizing cloned speech via neural codec...",
      "Streaming fake audio to verification endpoint...",
    ],
  },
  {
    type: "deepfake_visual" as const,
    label: "Deepfake Video Injection",
    description: "GAN-generated face swap piped through virtual camera",
    stages: [
      "Loading face-swap GAN model...",
      "Piping synthetic frames via OBS Virtual Cam...",
      "Injecting deepfake video into WebRTC stream...",
    ],
  },
  {
    type: "replay_attack" as const,
    label: "Replay Attack (Recorded Session)",
    description: "Replaying a previously recorded legitimate session",
    stages: [
      "Loading recorded session audio/video...",
      "Bypassing timestamp checks...",
      "Replaying biometric data to API...",
    ],
  },
  {
    type: "adversarial_inject" as const,
    label: "Adversarial Perturbation Bypass",
    description: "Crafted adversarial noise to fool the AI classifier",
    stages: [
      "Generating adversarial perturbation (PGD, ε=0.03)...",
      "Applying imperceptible noise to audio spectrogram...",
      "Submitting perturbed sample to bypass detection...",
    ],
  },
];

const DEFENSE_RESPONSES: Record<string, { type: DefenseEvent["type"]; label: string; detail: string }[]> = {
  voice_clone: [
    { type: "detected", label: "Codec Artifact Detected", detail: "Neural codec compression artifacts found in mel-spectrogram (confidence: 94.2%)" },
    { type: "detected", label: "Speaker Mismatch", detail: "ECAPA-TDNN embedding distance > threshold (cosine similarity: 0.31)" },
    { type: "blocked", label: "Connection Terminated", detail: "Trust score collapsed to 18.4 — verdict: DEEPFAKE" },
    { type: "proof_generated", label: "ZK Proof Generated", detail: "Fiat-Shamir STARK proof: commitment + challenge + response (latency: 12ms)" },
    { type: "anchored", label: "Fraud Event Anchored", detail: "Proof hash anchored to Polygon Amoy — immutable audit trail created" },
  ],
  deepfake_visual: [
    { type: "detected", label: "Virtual Camera Detected", detail: "rPPG WASM module detected OBS Virtual Cam (jitter score: 0.82)" },
    { type: "detected", label: "rPPG Liveness Failed", detail: "No physiological pulse signal detected — synthetic frames confirmed" },
    { type: "blocked", label: "Adversarial Shield Activated", detail: "PGD perturbation applied to collapse deepfake generator" },
    { type: "proof_generated", label: "ZK Attestation", detail: "Zero-knowledge proof of detection generated (no raw biometrics leaked)" },
    { type: "anchored", label: "On-Chain Record", detail: "CEF:0|VAJRA|FraudAttemptDetected|9| — SIEM event emitted" },
  ],
  replay_attack: [
    { type: "detected", label: "Nullifier Replay Detected", detail: "Session nullifier already consumed — replay attack blocked" },
    { type: "detected", label: "Timestamp Anomaly", detail: "Frame timestamps are non-monotonic — recorded session detected" },
    { type: "blocked", label: "Session Invalidated", detail: "Nullifier binding prevents double-use of proof" },
    { type: "proof_generated", label: "Fraud Proof Issued", detail: "Cryptographic proof of replay attempt generated" },
    { type: "anchored", label: "Alert Escalated", detail: "CEF severity 9 event → SOC escalation triggered" },
  ],
  adversarial_inject: [
    { type: "detected", label: "Perturbation Detected", detail: "Spectral analysis reveals non-natural noise pattern (L∞ norm: 0.028)" },
    { type: "detected", label: "Ensemble Disagreement", detail: "3-model ensemble produced conflicting verdicts — adversarial input suspected" },
    { type: "blocked", label: "Counter-Perturbation Applied", detail: "VAJRA adversarial shield neutralized attack with defensive noise" },
    { type: "proof_generated", label: "Defense Proof", detail: "ZK proof of adversarial detection generated" },
    { type: "anchored", label: "Threat Intelligence", detail: "Attack signature logged in STIX format for threat sharing" },
  ],
};

// ── Component ──────────────────────────────────────────────────────────────

export default function RedVsBlue() {
  const [battle, setBattle] = useState<BattleState>({
    isRunning: false,
    attackEvents: [],
    defenseEvents: [],
    redScore: 0,
    blueScore: 0,
    currentAttack: null,
    currentDefense: null,
  });

  const [selectedScenario, setSelectedScenario] = useState(0);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const eventIdRef = useRef(0);

  const nextId = useCallback(() => {
    eventIdRef.current += 1;
    return `evt-${eventIdRef.current}`;
  }, []);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach(clearTimeout);
    };
  }, []);

  const runBattle = useCallback(() => {
    // Clear any previous timeouts
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];

    const scenario = ATTACK_SCENARIOS[selectedScenario];
    const defenses = DEFENSE_RESPONSES[scenario.type] || [];

    setBattle({
      isRunning: true,
      attackEvents: [],
      defenseEvents: [],
      redScore: 0,
      blueScore: 0,
      currentAttack: null,
      currentDefense: null,
    });

    // Schedule attack events
    scenario.stages.forEach((stage, i) => {
      const t = setTimeout(() => {
        setBattle((prev) => ({
          ...prev,
          currentAttack: stage,
          redScore: Math.min(prev.redScore + 25, 100),
          attackEvents: [
            ...prev.attackEvents,
            {
              id: nextId(),
              timestamp: Date.now(),
              type: scenario.type,
              label: stage,
              confidence: 70 + Math.random() * 25,
            },
          ],
        }));
      }, (i + 1) * 1500);
      timeoutsRef.current.push(t);
    });

    // Schedule defense responses (slightly delayed)
    defenses.forEach((defense, i) => {
      const t = setTimeout(() => {
        const trustScore = defense.type === "blocked" ? 18 + Math.random() * 10 : 50 + Math.random() * 30;
        setBattle((prev) => ({
          ...prev,
          currentDefense: defense.label,
          blueScore: Math.min(prev.blueScore + 20, 100),
          defenseEvents: [
            ...prev.defenseEvents,
            {
              id: nextId(),
              timestamp: Date.now(),
              type: defense.type,
              label: defense.label,
              detail: defense.detail,
              trustScore,
            },
          ],
        }));
      }, (i + 1) * 1500 + 800);
      timeoutsRef.current.push(t);
    });

    // End battle
    const endT = setTimeout(() => {
      setBattle((prev) => ({
        ...prev,
        isRunning: false,
        blueScore: 100,
        currentAttack: "❌ Attack Failed",
        currentDefense: "✅ Threat Neutralized",
      }));
    }, (Math.max(scenario.stages.length, defenses.length) + 1) * 1500 + 1200);
    timeoutsRef.current.push(endT);
  }, [selectedScenario, nextId]);

  const typeColor = (type: DefenseEvent["type"]) => {
    switch (type) {
      case "detected": return "text-yellow-400";
      case "blocked": return "text-red-400";
      case "proof_generated": return "text-cyan-400";
      case "anchored": return "text-green-400";
      default: return "text-gray-400";
    }
  };

  return (
    <div className="space-y-4">
      {/* Scenario Selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {ATTACK_SCENARIOS.map((s, i) => (
          <button
            key={s.type}
            onClick={() => setSelectedScenario(i)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono tracking-wide border transition-all ${
              selectedScenario === i
                ? "bg-red-500/20 border-red-500/50 text-red-400"
                : "bg-vajra-800/50 border-vajra-600/30 text-gray-500 hover:text-gray-300"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Battle Arena */}
      <div className="grid grid-cols-2 gap-4">
        {/* RED TEAM — Attacker */}
        <div className="bg-red-950/20 border border-red-500/30 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="text-red-400 text-xs font-bold tracking-widest uppercase">
              Red Team — Attacker
            </span>
          </div>

          {/* Attack Progress Bar */}
          <div className="w-full h-1.5 bg-red-950 rounded-full mb-3 overflow-hidden">
            <motion.div
              className="h-full bg-red-500 rounded-full"
              animate={{ width: `${battle.redScore}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>

          {/* Current Attack */}
          <AnimatePresence mode="wait">
            {battle.currentAttack && (
              <motion.div
                key={battle.currentAttack}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="text-xs text-red-300 font-mono mb-3 p-2 bg-red-950/40 rounded"
              >
                ▶ {battle.currentAttack}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Attack Event Log */}
          <div className="space-y-1 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {battle.attackEvents.map((evt) => (
                <motion.div
                  key={evt.id}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="text-[10px] text-red-400/70 font-mono border-l border-red-500/20 pl-2"
                >
                  [{new Date(evt.timestamp).toLocaleTimeString()}] {evt.label}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* BLUE TEAM — VAJRA Defense */}
        <div className="bg-cyan-950/20 border border-cyan-500/30 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-3 h-3 rounded-full bg-cyan-500 animate-pulse" />
            <span className="text-cyan-400 text-xs font-bold tracking-widest uppercase">
              Blue Team — VAJRA Defense
            </span>
          </div>

          {/* Defense Progress Bar */}
          <div className="w-full h-1.5 bg-cyan-950 rounded-full mb-3 overflow-hidden">
            <motion.div
              className="h-full bg-cyan-500 rounded-full"
              animate={{ width: `${battle.blueScore}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>

          {/* Current Defense */}
          <AnimatePresence mode="wait">
            {battle.currentDefense && (
              <motion.div
                key={battle.currentDefense}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="text-xs text-cyan-300 font-mono mb-3 p-2 bg-cyan-950/40 rounded"
              >
                🛡 {battle.currentDefense}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Defense Event Log */}
          <div className="space-y-1 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {battle.defenseEvents.map((evt) => (
                <motion.div
                  key={evt.id}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className={`text-[10px] font-mono border-l border-cyan-500/20 pl-2 ${typeColor(evt.type)}`}
                >
                  <div>[{new Date(evt.timestamp).toLocaleTimeString()}] {evt.label}</div>
                  <div className="text-gray-500 ml-2">{evt.detail}</div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Launch Button */}
      <div className="flex justify-center pt-2">
        <motion.button
          onClick={runBattle}
          disabled={battle.isRunning}
          whileHover={{ scale: battle.isRunning ? 1 : 1.05 }}
          whileTap={{ scale: 0.97 }}
          className={`px-8 py-3 rounded-xl font-bold tracking-widest text-xs uppercase transition-all ${
            battle.isRunning
              ? "bg-gray-800 border border-gray-600 text-gray-500 cursor-not-allowed"
              : "bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30"
          }`}
        >
          {battle.isRunning ? "⚔️ BATTLE IN PROGRESS..." : "⚔️ LAUNCH ATTACK SCENARIO"}
        </motion.button>
      </div>

      {/* Battle Result */}
      <AnimatePresence>
        {!battle.isRunning && battle.blueScore === 100 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-center p-4 bg-cyan-950/30 border border-cyan-500/30 rounded-xl"
          >
            <div className="text-cyan-400 text-sm font-bold tracking-widest mb-1">
              🛡 VAJRA DEFENSE SUCCESSFUL
            </div>
            <div className="text-gray-500 text-xs">
              Attack neutralized • ZK proof generated • Fraud event anchored on-chain • CEF alert emitted to SIEM
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
