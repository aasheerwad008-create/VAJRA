// VAJRA Frontend — TypeScript type definitions
export interface TrustScoreComponents {
  deepfake_model: number;
  codec_detector: number;
  speaker_match: number;
  rppg_liveness: number;
}

export interface VoiceScoreEvent {
  session_id: string;
  trust_score: number;
  verdict: "VERIFIED" | "SUSPICIOUS" | "DEEPFAKE";
  components: TrustScoreComponents;
  timestamp: number;
}

export interface VerificationState {
  trustScore: number;
  verdict: "IDLE" | "VERIFIED" | "SUSPICIOUS" | "DEEPFAKE";
  components: Partial<TrustScoreComponents>;
  proofHash: string | null;
  txHash: string | null;
  sessionId: string | null;
  isVerifying: boolean;
}

export interface VerifyAPIResponse {
  session_id: string;
  trust_score: number;
  verdict: string;
  proof_hash: string;
  tx_hash?: string;
  timestamp: string;
}

// Aliases used by lib/api.ts
export type VerifyResponse = VerifyAPIResponse;

export interface VerifyRequest {
  user_id: string;
  audio_data?: string;   // Base64-encoded audio blob
  key_proof?: string;    // HMAC-SHA256 key-possession proof
  proof_hash?: string;   // Nullifier from previous session (for replay detection)
  liveness_score?: number;
  trust_score?: number;
}

export interface SessionStatus {
  session_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  verdict?: string;
  trust_score?: number;
  proof_hash?: string;
  created_at: string;
  updated_at: string;
}
