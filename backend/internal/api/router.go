// VAJRA Backend — HTTP router & handlers
package api

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
)

// Config holds all service configuration.
type Config struct {
	JWTSecret            string
	VoiceAIURL           string
	AdversarialURL       string
	ZKProofURL           string
	PolygonRPCURL        string
	TrustRegistryAddress string
	RedisURL             string
}

// NewRouter builds the Gin engine with all routes wired up.
func NewRouter(log *zap.Logger, pool *pgxpool.Pool, cfg *Config) http.Handler {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(ginLogger(log), gin.Recovery())

	// ── Health ────────────────────────────────────────────────────────────
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "service": "backend"})
	})

	api := r.Group("/api")

	// ── Verification pipeline ─────────────────────────────────────────────
	api.POST("/verify", verifyHandler(log, pool, cfg))

	// ── History ───────────────────────────────────────────────────────────
	api.GET("/history/:user", historyHandler(log, pool))

	// ── ZK proxy ─────────────────────────────────────────────────────────
	api.POST("/zk/register", proxyHandler(cfg.ZKProofURL+"/api/zk/register"))
	api.POST("/zk/verify", proxyHandler(cfg.ZKProofURL+"/api/zk/verify"))

	// ── Adversarial proxy ─────────────────────────────────────────────────
	api.POST("/adversarial/perturb-frame", proxyMultipartHandler(cfg.AdversarialURL+"/api/adversarial/perturb-frame"))

	return r
}

// ── Verify handler ─────────────────────────────────────────────────────────

type VerifyRequest struct {
	UserID        string  `json:"user_id" binding:"required"`
	TrustScore    float64 `json:"trust_score" binding:"required"`
	Verdict       string  `json:"verdict" binding:"required"`
	ProofHash     string  `json:"proof_hash"`
	LivenessScore float64 `json:"liveness_score"`
	KeyProof      string  `json:"key_proof"`
}

type VerifyResponse struct {
	SessionID  string  `json:"session_id"`
	TrustScore float64 `json:"trust_score"`
	Verdict    string  `json:"verdict"`
	ProofHash  string  `json:"proof_hash"`
	TxHash     string  `json:"tx_hash,omitempty"`
	Timestamp  string  `json:"timestamp"`
}

func verifyHandler(log *zap.Logger, pool *pgxpool.Pool, cfg *Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req VerifyRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		sessionID := uuid.New().String()

		// ── Step 1: ZK proof verification ─────────────────────────────────
		zkResp, err := callZKVerify(cfg.ZKProofURL, req)
		if err != nil {
			log.Warn("zk.verify_failed", zap.Error(err))
		}

		proofHash := req.ProofHash
		if zkResp != nil {
			proofHash = zkResp.ProofHash
		}

		// ── Step 2: Anchor to blockchain ──────────────────────────────────
		txHash, err := anchorToBlockchain(cfg, proofHash, req.UserID, req.Verdict)
		if err != nil {
			log.Warn("blockchain.anchor_failed", zap.Error(err))
		}

		// ── Step 3: Persist session ────────────────────────────────────────
		timestamp := time.Now().UTC().Format(time.RFC3339)
		if pool != nil {
			_, err = pool.Exec(context.Background(),
				`INSERT INTO verification_sessions
					(id, trust_score, verdict, proof_hash, tx_hash, status, completed_at)
				 VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
				sessionID, req.TrustScore, req.Verdict, proofHash, txHash, "completed",
			)
			if err != nil {
				log.Warn("db.insert_session_failed", zap.Error(err))
			}
		}

		c.JSON(200, VerifyResponse{
			SessionID:  sessionID,
			TrustScore: req.TrustScore,
			Verdict:    req.Verdict,
			ProofHash:  proofHash,
			TxHash:     txHash,
			Timestamp:  timestamp,
		})
	}
}

// ── History handler ────────────────────────────────────────────────────────

type SessionRecord struct {
	SessionID   string   `json:"session_id"`
	TrustScore  float64  `json:"trust_score"`
	Verdict     string   `json:"verdict"`
	ProofHash   string   `json:"proof_hash"`
	TxHash      string   `json:"tx_hash"`
	Status      string   `json:"status"`
	CreatedAt   string   `json:"created_at"`
	CompletedAt *string  `json:"completed_at,omitempty"`
}

func historyHandler(log *zap.Logger, pool *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		user := c.Param("user")
		if pool == nil {
			c.JSON(200, gin.H{"sessions": []SessionRecord{}})
			return
		}

		rows, err := pool.Query(context.Background(),
			`SELECT id::text, trust_score, verdict, COALESCE(proof_hash,''),
			        COALESCE(tx_hash,''), status, created_at::text,
			        completed_at::text
			 FROM verification_sessions
			 WHERE user_id = (SELECT id FROM users WHERE username = $1 LIMIT 1)
			    OR id::text = $1
			 ORDER BY created_at DESC
			 LIMIT 50`,
			user,
		)
		if err != nil {
			log.Error("db.query_failed", zap.Error(err))
			c.JSON(500, gin.H{"error": "database error"})
			return
		}
		defer rows.Close()

		var sessions []SessionRecord
		for rows.Next() {
			var s SessionRecord
			var completedAt *string
			if err := rows.Scan(
				&s.SessionID, &s.TrustScore, &s.Verdict,
				&s.ProofHash, &s.TxHash, &s.Status,
				&s.CreatedAt, &completedAt,
			); err != nil {
				continue
			}
			s.CompletedAt = completedAt
			sessions = append(sessions, s)
		}

		c.JSON(200, gin.H{"user": user, "sessions": sessions})
	}
}

// ── Proxy helpers ──────────────────────────────────────────────────────────

func proxyHandler(target string) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(500, gin.H{"error": "read body"})
			return
		}

		ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
		defer cancel()

		req, _ := http.NewRequestWithContext(ctx, c.Request.Method, target, bytes.NewReader(body))
		req.Header = c.Request.Header.Clone()

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			c.JSON(502, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()

		respBody, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
	}
}

func proxyMultipartHandler(target string) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(500, gin.H{"error": "read body"})
			return
		}

		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
		defer cancel()

		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
		req.Header = c.Request.Header.Clone()

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			c.JSON(502, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()

		respBody, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
	}
}

// ── Logging middleware ─────────────────────────────────────────────────────

func ginLogger(log *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		log.Info("http.request",
			zap.String("method", c.Request.Method),
			zap.String("path", c.Request.URL.Path),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("latency", time.Since(start)),
		)
	}
}
