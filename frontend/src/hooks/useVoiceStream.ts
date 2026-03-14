"use client";

import { useCallback, useRef, useState } from "react";
import type { VoiceScoreEvent } from "@/types";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8080";

const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 2;

interface UseVoiceStreamOptions {
  onScore: (score: VoiceScoreEvent) => void;
}

export function useVoiceStream({ onScore }: UseVoiceStreamOptions) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startStream = useCallback(async () => {
    try {
      const sessionId = crypto.randomUUID();
      const wsUrl = `${WS_URL.replace(/^http/, "ws")}/ws/voice/stream/${sessionId}`;

      // Try to connect WebSocket to voice-ai service via backend proxy
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => console.log("[VAJRA] WS connected");
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as VoiceScoreEvent;
          onScore(data);
        } catch {
          // ignore parse errors
        }
      };
      ws.onerror = (err) => console.error("[VAJRA] WS error", err);
      ws.onclose = () => setIsStreaming(false);

      // Get microphone
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true },
      });
      streamRef.current = mediaStream;

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(mediaStream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      let buffer: Float32Array[] = [];
      let totalSamples = 0;
      const chunkSamples = SAMPLE_RATE * CHUNK_SECONDS;

      processor.onaudioprocess = (evt) => {
        const input = evt.inputBuffer.getChannelData(0);
        buffer.push(new Float32Array(input));
        totalSamples += input.length;

        // Compute RMS for level meter
        const rms = Math.sqrt(input.reduce((acc, v) => acc + v * v, 0) / input.length);
        setAudioLevel(Math.min(1, rms * 10));

        if (totalSamples >= chunkSamples) {
          // Concatenate buffer
          const chunk = new Float32Array(totalSamples);
          let offset = 0;
          buffer.forEach((b) => {
            chunk.set(b, offset);
            offset += b.length;
          });
          buffer = [];
          totalSamples = 0;

          // Convert to PCM16 and send
          const pcm16 = float32ToPcm16(chunk.slice(0, chunkSamples));
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(pcm16.buffer);
          }
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsStreaming(true);
    } catch (err) {
      console.error("[VAJRA] Failed to start stream", err);
    }
  }, [onScore]);

  const stopStream = useCallback(() => {
    wsRef.current?.close();
    processorRef.current?.disconnect();
    audioContextRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());

    wsRef.current = null;
    processorRef.current = null;
    audioContextRef.current = null;
    streamRef.current = null;

    setIsStreaming(false);
    setAudioLevel(0);
  }, []);

  return { startStream, stopStream, isStreaming, audioLevel };
}

// ── Helpers ────────────────────────────────────────────────────────────────

function float32ToPcm16(float32: Float32Array): Int16Array {
  const pcm = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32[i]));
    pcm[i] = Math.round(clamped * 32767);
  }
  return pcm;
}
