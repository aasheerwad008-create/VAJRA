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
