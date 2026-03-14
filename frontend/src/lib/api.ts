/**
 * VAJRA Frontend — REST API client.
 *
 * Provides typed wrappers around all VAJRA backend endpoints.
 * The base URL is read from NEXT_PUBLIC_API_URL (default: http://localhost:8080).
 */

import type { VerifyRequest, VerifyResponse, SessionStatus } from "@/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

// ── Generic helpers ────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} → ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} → ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── API client object ──────────────────────────────────────────────────────

export const apiClient = {
  /** Submit a biometric verification request. */
  verify: (req: VerifyRequest): Promise<VerifyResponse> =>
    post<VerifyResponse>("/api/verify", req),

  /** Get the current status of a verification session. */
  getSession: (sessionId: string): Promise<SessionStatus> =>
    get<SessionStatus>(`/api/sessions/${sessionId}`),

  /** Health check — returns { status: "ok" } when the backend is up. */
  health: (): Promise<{ status: string }> =>
    get<{ status: string }>("/health"),

  /** Trigger the voice-clone demo attack scenario. */
  triggerDemoVoiceClone: (): Promise<VerifyResponse> =>
    post<VerifyResponse>("/api/demo/voice-clone", {}),

  /** Trigger the deepfake-visual demo attack scenario. */
  triggerDemoDeepfake: (): Promise<VerifyResponse> =>
    post<VerifyResponse>("/api/demo/deepfake-visual", {}),

  /** Trigger the ZK replay attack demo scenario. */
  triggerDemoReplay: (): Promise<VerifyResponse> =>
    post<VerifyResponse>("/api/demo/replay-attack", {}),
};
