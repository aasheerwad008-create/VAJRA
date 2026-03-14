// VAJRA Backend — main entry point
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/vajra/backend/internal/api"
	"github.com/vajra/backend/internal/db"
	"go.uber.org/zap"
)

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync() //nolint:errcheck

	// ── Database ─────────────────────────────────────────────────────────
	dbURL := env("DATABASE_URL", "postgresql://vajra:vajra_secret@postgres:5432/vajra")
	pool, err := db.NewPool(dbURL)
	if err != nil {
		logger.Fatal("db.connect_failed", zap.Error(err))
	}
	defer pool.Close()

	if err := db.Migrate(pool); err != nil {
		logger.Fatal("db.migrate_failed", zap.Error(err))
	}

	// ── Config ────────────────────────────────────────────────────────────
	cfg := &api.Config{
		JWTSecret:              env("JWT_SECRET", "change_me_in_production_32_char_min"),
		VoiceAIURL:             env("VOICE_AI_URL", "http://voice-ai:8001"),
		AdversarialURL:         env("ADVERSARIAL_URL", "http://adversarial-engine:8002"),
		ZKProofURL:             env("ZK_PROOF_URL", "http://zk-proof-system:8003"),
		PolygonRPCURL:          env("POLYGON_RPC_URL", "https://rpc-amoy.polygon.technology"),
		PolygonFallbackRPCURL:  env("POLYGON_FALLBACK_RPC_URL", ""),
		TrustRegistryAddress:   env("TRUST_REGISTRY_ADDRESS", "0x0000000000000000000000000000000000000000"),
		RedisURL:               env("REDIS_URL", "redis://redis:6379"),
	}

	// ── HTTP Server ───────────────────────────────────────────────────────
	router := api.NewRouter(logger, pool, cfg)

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		logger.Info("backend.listening", zap.String("addr", ":8080"))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server.error", zap.Error(err))
		}
	}()

	// ── Graceful shutdown ─────────────────────────────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("backend.shutting_down")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("server.shutdown_error", zap.Error(err))
	}
	logger.Info("backend.stopped")
}

func env(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}
