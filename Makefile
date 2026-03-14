# VAJRA — Development Commands
# Usage: make <target>

.PHONY: help test lint build docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Test ──────────────────────────────────────────────────────────────────

test: test-frontend test-backend test-voice-ai test-adversarial test-zk test-rppg ## Run all tests

test-frontend: ## Run frontend tests
	cd frontend && npm run test -- --run

test-backend: ## Run Go backend tests
	cd backend && go test ./... -v -count=1

test-voice-ai: ## Run voice-ai tests
	cd voice-ai && python -m pytest tests/ -v

test-adversarial: ## Run adversarial-engine tests
	cd adversarial-engine && python -m pytest tests/ -v

test-zk: ## Run ZK proof system tests
	cd zk-proof-system && cargo test --manifest-path host/Cargo.toml -v
	cd zk-proof-system && cargo test --manifest-path guest/Cargo.toml -v

test-rppg: ## Run rPPG WASM tests
	cd rppg-wasm && cargo test -v

test-blockchain: ## Run smart contract tests
	cd blockchain && npx hardhat test

# ── Lint ──────────────────────────────────────────────────────────────────

lint: lint-frontend lint-backend ## Run all linters

lint-frontend: ## Lint frontend
	cd frontend && npm run lint

lint-backend: ## Lint Go backend
	cd backend && go vet ./...

# ── Build ─────────────────────────────────────────────────────────────────

build: build-frontend build-backend build-zk build-rppg ## Build all components

build-frontend: ## Build frontend
	cd frontend && npm run build

build-backend: ## Build Go backend
	cd backend && go build ./...

build-zk: ## Build ZK proof system
	cd zk-proof-system && cargo build --manifest-path host/Cargo.toml
	cd zk-proof-system && cargo build --manifest-path guest/Cargo.toml

build-rppg: ## Build rPPG WASM
	cd rppg-wasm && cargo build

build-blockchain: ## Compile smart contracts
	cd blockchain && npx hardhat compile

# ── Docker ────────────────────────────────────────────────────────────────

docker-up: ## Start all services via Docker Compose
	docker compose up -d --build

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Follow logs from all services
	docker compose logs -f

# ── Utilities ─────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf frontend/.next frontend/node_modules/.cache
	rm -rf backend/vendor
	cd zk-proof-system && cargo clean --manifest-path host/Cargo.toml
	cd zk-proof-system && cargo clean --manifest-path guest/Cargo.toml
	cd rppg-wasm && cargo clean

install: ## Install all dependencies
	cd frontend && npm ci
	cd blockchain && npm ci
	cd voice-ai && pip install -r requirements.txt
	cd adversarial-engine && pip install -r requirements.txt
