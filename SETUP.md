# VAJRA — Setup Guide

Step-by-step instructions to get VAJRA running. Choose between **Docker (recommended)** for a one-command start, or **local development** for component-level work.

---

## Prerequisites

| Tool | Version | Required For |
|------|---------|-------------|
| Docker & Docker Compose | 24+ / v2 | Docker setup (recommended) |
| Python | 3.11+ | Voice AI, Adversarial Engine |
| Go | 1.22+ | Backend API |
| Rust | 1.79+ | ZK Proof System, rPPG WASM |
| Node.js | 20+ | Frontend, Blockchain |
| NVIDIA GPU + CUDA | Optional | GPU-accelerated inference |

---

## Quick Start (Docker)

The fastest way to run the full platform:

```bash
# 1. Clone the repository
git clone https://github.com/aasheerwad008-create/VAJRA.git
cd VAJRA

# 2. Configure environment
cp .env.example .env
# Edit .env with your secrets (see Environment Variables below)

# 3. Start all services
docker-compose up --build
```

Services start in dependency order. Once healthy:

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8080 |
| Voice AI | http://localhost:8001 |
| Adversarial Engine | http://localhost:8002 |
| ZK Proof System | http://localhost:8003 |

> **GPU support:** Docker Compose reserves NVIDIA GPUs for `voice-ai` and `adversarial-engine` automatically. Falls back to CPU if no GPU is available.

### Stop Services

```bash
docker-compose down          # Stop and remove containers
docker-compose down -v       # Also remove volumes (database data)
```

---

## Do I Need to Train Datasets?

**No.** VAJRA uses pretrained models that work out of the box:

- **EfficientNet-B0** (ImageNet-1K) — spectrogram backbone, downloaded via `timm`
- **ECAPA-TDNN** (VoxCeleb1+2) — speaker verification, downloaded via `SpeechBrain`
- **RawNet2** — Kaiming-initialised sinc filters, no external weights needed
- **Codec Detector** — Kaiming-initialised 1-D CNN, no external weights needed

### When Should You Train?

Training is optional and only needed if:

- You want to **fine-tune** models on your own audio domain
- You have a **custom deepfake dataset** (e.g. specific TTS systems)
- You want to improve detection for **specific codec types**
- You're doing **research** and need to reproduce results

See the [ML Training section in the README](README.md#ml-training-optional) for training instructions.

---

## Pretrained Weights Setup

### Automatic Download (Recommended)

```bash
cd voice-ai

# Download all pretrained weights (~100 MB)
python -m pretrained.setup_weights

# Verify what's cached
python -m pretrained.setup_weights --status

# Show registry details
python -m pretrained.setup_weights --info
```

### What Gets Downloaded

| Weight | Source | Size | Downloaded By |
|--------|--------|------|---------------|
| EfficientNet-B0 backbone | ImageNet-1K | 20.5 MB | `timm` library |
| ECAPA-TDNN encoder | VoxCeleb1+2 | 83.0 MB | `speechbrain` library |
| RawNet2 | *(random init)* | 0 MB | Built-in |
| Codec Detector | *(random init)* | 0 MB | Built-in |

### Docker Setup

When using Docker, pretrained weights are downloaded automatically during container startup. The `voice_models` Docker volume persists them across restarts:

```yaml
# docker-compose.yml (already configured)
volumes:
  - voice_models:/app/models
```

### Custom Weight Directory

Override the weight cache directory:

```bash
# Via CLI flag
python -m pretrained.setup_weights --model-dir /path/to/weights

# Via environment variable
export MODEL_DIR=/path/to/weights
python -m pretrained.setup_weights
```

---

## Local Development Setup

For working on individual components without Docker.

### 1. Voice AI (Python)

```bash
cd voice-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
python -m pretrained.setup_weights

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Run tests
python -m pytest tests/ -v
```

### 2. Adversarial Engine (Python)

```bash
cd adversarial-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# Run tests
python -m pytest tests/ -v
```

### 3. Backend API (Go)

```bash
cd backend

# Download dependencies
go mod download

# Build
go build ./...

# Run the server
go run cmd/server/main.go

# Run tests
go test ./... -v

# Lint
go vet ./...
```

### 4. ZK Proof System (Rust)

```bash
cd zk-proof-system

# Build host + guest
cargo build --manifest-path host/Cargo.toml
cargo build --manifest-path guest/Cargo.toml

# Run the service
cargo run --manifest-path host/Cargo.toml

# Run tests
cargo test --manifest-path host/Cargo.toml
cargo test --manifest-path guest/Cargo.toml
```

### 5. rPPG WASM Module (Rust)

```bash
cd rppg-wasm

# Build
cargo build

# Run tests
cargo test
```

### 6. Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev
# → http://localhost:3000

# Run tests
npm run test -- --run

# Build for production
npm run build
```

### 7. Blockchain (Hardhat)

```bash
cd blockchain

# Install dependencies
npm install

# Compile contracts
npx hardhat compile

# Run tests
npx hardhat test

# Deploy to Polygon Amoy (set DEPLOYER_PRIVATE_KEY in .env)
npm run deploy:amoy
```

---

## Environment Variables

Copy `.env.example` and configure:

```bash
cp .env.example .env
```

### Required Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `vajra` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `vajra_secret` | PostgreSQL password |
| `POSTGRES_DB` | `vajra` | Database name |
| `JWT_SECRET` | *(change me)* | JWT signing secret (min 32 chars) |
| `API_KEY` | *(change me)* | Backend API key |

### Blockchain (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYGON_AMOY_RPC_URL` | Public RPC | Polygon Amoy RPC endpoint |
| `DEPLOYER_PRIVATE_KEY` | Placeholder | Wallet private key for deployment |
| `TRUST_REGISTRY_ADDRESS` | `0x000...` | Deployed contract address |

### Service URLs (Docker defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_AI_URL` | `http://voice-ai:8001` | Voice AI service URL |
| `ADVERSARIAL_URL` | `http://adversarial-engine:8002` | Adversarial engine URL |
| `ZK_PROOF_URL` | `http://zk-proof-system:8003` | ZK proof system URL |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |

---

## Makefile Shortcuts

```bash
make install         # Install all dependencies
make build           # Build all components
make test            # Run all test suites
make lint            # Run all linters
make docker-up       # Start Docker services
make docker-down     # Stop Docker services
make docker-logs     # Follow Docker logs
make clean           # Remove build artifacts
```

### Per-Component Targets

```bash
make test-frontend      # Frontend tests (Vitest)
make test-backend       # Go backend tests
make test-voice-ai      # Voice AI pytest
make test-adversarial   # Adversarial engine pytest
make test-zk            # ZK proof system (Cargo)
make test-rppg          # rPPG WASM tests
make test-blockchain    # Smart contract tests (Hardhat)
```

---

## Troubleshooting

### Pretrained weights download fails

```bash
# Check status
cd voice-ai && python -m pretrained.setup_weights --status

# Force re-download
python -m pretrained.setup_weights --force -v
```

If you're behind a firewall, the timm and SpeechBrain libraries need access to:
- `huggingface.co` (model downloads)
- `github.com` (SpeechBrain configs)

### Docker: GPU not detected

Ensure NVIDIA Container Toolkit is installed:

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If no GPU is available, services fall back to CPU automatically.

### Docker: Services unhealthy

AI containers need time to load PyTorch models. Check logs:

```bash
docker-compose logs -f voice-ai
docker-compose logs -f adversarial-engine
```

Health check `start_period` is 120s for voice-ai and 60s for adversarial-engine.

### Port conflicts

Default ports: 3000 (frontend), 8080 (backend), 8001 (voice-ai), 8002 (adversarial), 8003 (zk-proof), 5432 (postgres), 6379 (redis). Change in `docker-compose.yml` or `.env` if needed.
