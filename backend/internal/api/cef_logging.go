// VAJRA Backend — CEF (Common Event Format) security event logging.
//
// Emits structured security events in ArcSight CEF format for SIEM/SOC
// integration.  Events are written to the application logger and can be
// forwarded to Wazuh, Elastic Security, Splunk, or any CEF-compatible
// collector.
//
// CEF format:
//   CEF:0|VAJRA|IdentityDefense|2.0|<eventID>|<name>|<severity>|<extensions>
package api

import (
	"fmt"
	"time"

	"go.uber.org/zap"
)

// CEF severity levels (0–10 scale per ArcSight specification).
const (
	CEFSeverityLow      = 3
	CEFSeverityMedium   = 5
	CEFSeverityHigh     = 7
	CEFSeverityCritical = 9
)

// CEF event IDs.
const (
	CEFEventIdentityVerified     = "100"
	CEFEventFraudAttemptDetected = "200"
	CEFEventRateLimitExceeded    = "300"
	CEFEventBlockchainAnchor     = "400"
	CEFEventZKProofGenerated     = "500"
	CEFEventSuspiciousActivity   = "600"
)

// cefEvent formats a single CEF log line.
//
//	CEF:0|VAJRA|IdentityDefense|2.0|eventID|eventName|severity|extensions
func cefEvent(eventID, eventName string, severity int, extensions string) string {
	return fmt.Sprintf(
		"CEF:0|VAJRA|IdentityDefense|2.0|%s|%s|%d|%s",
		eventID, eventName, severity, extensions,
	)
}

// EmitIdentityVerified logs a successful identity verification event.
func EmitIdentityVerified(log *zap.Logger, userID, sessionID, proofHash, verdict string, trustScore float64) {
	ext := fmt.Sprintf(
		"src=%s duser=%s cs1=%s cs1Label=sessionId cs2=%s cs2Label=proofHash cs3=%s cs3Label=verdict cn1=%.1f cn1Label=trustScore rt=%d",
		userID, userID, sessionID, proofHash, verdict, trustScore, time.Now().UnixMilli(),
	)
	log.Info("cef.security_event",
		zap.String("cef", cefEvent(CEFEventIdentityVerified, "IdentityVerified", CEFSeverityLow, ext)),
		zap.String("event_type", "identity_verified"),
		zap.String("user_id", userID),
	)
}

// EmitFraudAttemptDetected logs a detected deepfake or fraud attempt.
func EmitFraudAttemptDetected(log *zap.Logger, userID, sessionID, verdict string, trustScore float64) {
	ext := fmt.Sprintf(
		"src=%s duser=%s cs1=%s cs1Label=sessionId cs2=%s cs2Label=verdict cn1=%.1f cn1Label=trustScore cat=Fraud rt=%d",
		userID, userID, sessionID, verdict, trustScore, time.Now().UnixMilli(),
	)
	severity := CEFSeverityHigh
	if trustScore < 30 {
		severity = CEFSeverityCritical
	}
	log.Warn("cef.security_event",
		zap.String("cef", cefEvent(CEFEventFraudAttemptDetected, "FraudAttemptDetected", severity, ext)),
		zap.String("event_type", "fraud_attempt_detected"),
		zap.String("user_id", userID),
		zap.Float64("trust_score", trustScore),
	)
}

// EmitRateLimitExceeded logs when an IP exceeds the rate limit.
func EmitRateLimitExceeded(log *zap.Logger, clientIP, path string) {
	ext := fmt.Sprintf(
		"src=%s request=%s cat=DoS msg=Rate%%20limit%%20exceeded rt=%d",
		clientIP, path, time.Now().UnixMilli(),
	)
	log.Warn("cef.security_event",
		zap.String("cef", cefEvent(CEFEventRateLimitExceeded, "RateLimitExceeded", CEFSeverityMedium, ext)),
		zap.String("event_type", "rate_limit_exceeded"),
		zap.String("client_ip", clientIP),
	)
}

// EmitBlockchainAnchor logs a successful blockchain proof anchoring.
func EmitBlockchainAnchor(log *zap.Logger, userID, txHash, proofHash string) {
	ext := fmt.Sprintf(
		"duser=%s cs1=%s cs1Label=txHash cs2=%s cs2Label=proofHash rt=%d",
		userID, txHash, proofHash, time.Now().UnixMilli(),
	)
	log.Info("cef.security_event",
		zap.String("cef", cefEvent(CEFEventBlockchainAnchor, "BlockchainAnchor", CEFSeverityLow, ext)),
		zap.String("event_type", "blockchain_anchor"),
		zap.String("user_id", userID),
	)
}

// EmitZKProofGenerated logs a ZK proof generation event.
func EmitZKProofGenerated(log *zap.Logger, userID, proofHash string, verified bool, latencyMs float64) {
	ext := fmt.Sprintf(
		"duser=%s cs1=%s cs1Label=proofHash cs2=%t cs2Label=verified cn1=%.2f cn1Label=latencyMs rt=%d",
		userID, proofHash, verified, latencyMs, time.Now().UnixMilli(),
	)
	log.Info("cef.security_event",
		zap.String("cef", cefEvent(CEFEventZKProofGenerated, "ZKProofGenerated", CEFSeverityLow, ext)),
		zap.String("event_type", "zk_proof_generated"),
		zap.String("user_id", userID),
	)
}

// EmitSuspiciousActivity logs suspicious activity that may indicate an attack.
func EmitSuspiciousActivity(log *zap.Logger, userID, sessionID, reason string, trustScore float64) {
	ext := fmt.Sprintf(
		"src=%s duser=%s cs1=%s cs1Label=sessionId msg=%s cn1=%.1f cn1Label=trustScore cat=Suspicious rt=%d",
		userID, userID, sessionID, reason, trustScore, time.Now().UnixMilli(),
	)
	log.Warn("cef.security_event",
		zap.String("cef", cefEvent(CEFEventSuspiciousActivity, "SuspiciousActivity", CEFSeverityMedium, ext)),
		zap.String("event_type", "suspicious_activity"),
		zap.String("user_id", userID),
		zap.String("reason", reason),
	)
}
