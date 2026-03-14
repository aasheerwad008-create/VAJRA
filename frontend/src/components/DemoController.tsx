"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { apiClient } from "@/lib/api";

type AttackScenario = "voice-clone" | "deepfake-visual" | "replay-attack";

interface DemoControllerProps {
  onScenarioStart?: (scenario: AttackScenario) => void;
  onScenarioEnd?: (scenario: AttackScenario, result: unknown) => void;
}

const SCENARIOS: {
  id: AttackScenario;
  label: string;
  description: string;
  emoji: string;
  color: string;
}[] = [
  {
    id: "voice-clone",
    label: "Voice Clone Attack",
    description: "Injects synthetic TTS audio; VAJRA detects codec artifacts + speaker mismatch",
    emoji: "🎙",
    color: "vajra-accent",
  },
  {
    id: "deepfake-visual",
    label: "Deepfake Visual Collapse",
    description: "Adversarial perturbation overlaid on video; disrupts face-swap model activations",
    emoji: "🎭",
    color: "yellow-400",
  },
  {
    id: "replay-attack",
    label: "ZK Replay Attack",
    description: "Attempts to reuse a previous ZK proof; nullifier binding prevents replay",
    emoji: "🔁",
    color: "red-400",
  },
];

/**
 * DemoController — demo mode toggle buttons for all 3 attack scenarios.
 *
 * Triggers each scenario via the VAJRA REST API and surfaces the
 * detection result in real time.
 */
export default function DemoController({ onScenarioStart, onScenarioEnd }: DemoControllerProps) {
  const [activeScenario, setActiveScenario] = useState<AttackScenario | null>(null);
  const [results, setResults] = useState<Partial<Record<AttackScenario, string>>>({});

  const runScenario = async (scenario: AttackScenario) => {
    if (activeScenario) return;
    setActiveScenario(scenario);
    onScenarioStart?.(scenario);

    try {
      let result: unknown;
      switch (scenario) {
        case "voice-clone":
          result = await apiClient.triggerDemoVoiceClone();
          break;
        case "deepfake-visual":
          result = await apiClient.triggerDemoDeepfake();
          break;
        case "replay-attack":
          result = await apiClient.triggerDemoReplay();
          break;
      }

      const verdict =
        typeof result === "object" && result !== null && "verdict" in result
          ? String((result as { verdict: unknown }).verdict)
          : "COMPLETE";

      setResults((prev) => ({ ...prev, [scenario]: verdict }));
      onScenarioEnd?.(scenario, result);
    } catch {
      setResults((prev) => ({ ...prev, [scenario]: "ERROR" }));
    } finally {
      setActiveScenario(null);
    }
  };

  const reset = () => setResults({});

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold tracking-widest text-gray-400 uppercase">
          Attack Scenarios
        </h3>
        {Object.keys(results).length > 0 && (
          <button
            onClick={reset}
            className="text-xs text-gray-500 hover:text-white transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      {SCENARIOS.map((s) => {
        const isRunning = activeScenario === s.id;
        const result    = results[s.id];

        const resultColor =
          result === "DEEPFAKE"   ? "text-red-400" :
          result === "SUSPICIOUS" ? "text-yellow-400" :
          result === "VERIFIED"   ? "text-vajra-green" :
          result === "ERROR"      ? "text-red-500" : "text-gray-300";

        return (
          <motion.div
            key={s.id}
            className="p-3 rounded-lg border border-vajra-600/20 bg-vajra-800/40 space-y-2"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-xs font-semibold text-white">
                  {s.emoji} {s.label}
                </p>
                <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">
                  {s.description}
                </p>
              </div>
              {result && (
                <span className={`text-xs font-mono ${resultColor} flex-shrink-0`}>
                  {result}
                </span>
              )}
            </div>

            <motion.button
              onClick={() => runScenario(s.id)}
              disabled={!!activeScenario}
              whileHover={{ scale: activeScenario ? 1 : 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`w-full py-1.5 rounded text-xs font-bold tracking-widest transition-all ${
                isRunning
                  ? "bg-vajra-accent/20 border border-vajra-accent/40 text-vajra-accent animate-pulse"
                  : activeScenario
                  ? "bg-gray-700/20 border border-gray-600/20 text-gray-600 cursor-not-allowed"
                  : "bg-vajra-600/20 border border-vajra-600/40 text-gray-300 hover:text-white"
              }`}
            >
              {isRunning ? "⟳ RUNNING…" : "▶ TRIGGER"}
            </motion.button>
          </motion.div>
        );
      })}
    </div>
  );
}
