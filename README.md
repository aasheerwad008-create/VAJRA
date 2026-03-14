# VAJRA — Zero-Trust AI Identity Defense Platform

> **Stop AI voice cloning, deepfake video attacks, and identity fraud in financial systems.**

VAJRA is a production-grade, three-layer security platform that provides real-time identity verification using voice AI, adversarial video protection, zero-knowledge proofs, and blockchain-anchored audit trails.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  VAJRA Platform                         │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Voice AI │  │  Adversarial │  │  ZK Proof       │  │
│  │ :8001    │  │  Engine :8002│  │  System :8003   │  │
│  │          │  │              │  │                 │  │
│  │Layer 1A  │  │  Layer 1B    │  │  Layer 2        │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬────────┘  │
│       └───────────────┼──────────────────┘            │
│                       ▼                               │
│              ┌──────────────────┐                     │
│              │  Backend (Go)    │ Layer 3: Blockchain │
│              │  :8080           │ ─────────────────── │
│              │                  │ Polygon Amoy        │
│              └────────┬─────────┘ VajraTrustRegistry │
│                       │                               │
│              ┌────────▼─────────┐                     │
│              │  Frontend        │                     │
│              │  (Next.js) :3000 │                     │
│              └──────────────────┘                     │
│                                                       │
│  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │  PostgreSQL     │  │  Redis Streams              │ │
│  │  :5432          │  │  :6379                      │ │
│  └────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Three-Layer Security Architecture

### Layer 1A — AI Active Destruction Engine (Voice AI)

Three-model ensemble for deepfake detection:

| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| Deepfake Spectrogram Classifier | EfficientNet-B0 | 128-bin mel spectrogram, 2s window | REAL/FAKE probability |
| Neural Codec Artifact Detector | 1D CNN | Raw 16kHz waveform, 2s window | HUMAN/ENCODEC/SOUNDSTREAM |
| Speaker Verification | ECAPA-TDNN (SpeechBrain) | Waveform | 192-d embedding + cosine similarity |

**Ensemble Trust Score:**
```
trust_score = 0.35 × deepfake_model
            + 0.25 × codec_detector
            + 0.20 × speaker_match
            + 0.20 × rPPG_liveness
```

**Verdict thresholds:**
- `score < 40` → 🔴 **DEEPFAKE**
- `score 40–70` → 🟡 **SUSPICIOUS**
- `score > 70` → 🟢 **VERIFIED**

### Layer 1B — Adversarial Video Shield

Imperceptible perturbations that collapse deepfake generators:

- **FGSM** — Fast Gradient Sign Method (Sobel gradient approximation)
- **PGD** — Projected Gradient Descent (iterative, L∞ constraint)
- **Adversarial Illumination** — Sinusoidal structured light pattern

rPPG liveness detection from video frames (bandpass filter + FFT heart-rate estimation).

### Layer 2 — Zero-Knowledge Identity Attestation

SHA-256 commitment-based ZK circuit proving:
- Speaker verification passed (score ≥ 70)
- Liveness detection passed (score ≥ 0.6)
- Private key ownership verified (HMAC proof)

**No raw biometric data is revealed** — only a commitment hash.

> **Note:** The current implementation provides the ZK API surface and commitment scheme. Integration with RISC Zero zkVM for full ZK-STARK proofs is the production upgrade path.

### Layer 3 — Blockchain Trust Registry

Solidity smart contract `VajraTrustRegistry` deployed on Polygon Amoy Testnet:

- Stores `proofHash`, `txHash`, `timestamp`, `verdict` on-chain
- Emits `IdentityVerified` / `FraudAttemptDetected` events
- Append-only, immutable audit trail
- Role-based access control (owner + authorized verifiers)

---

## ML Training

The project includes a production-grade training pipeline for the deepfake detection models using **pretrained + fine-tuning transfer learning**. Both the **Spectrogram Classifier** (EfficientNet-B0) and the **Codec Artifact Detector** (1-D CNN) can be trained on the [ASVspoof 2024](https://www.asvspoof.org/) dataset.

### Training Strategy: Pretrained + Fine-Tuning

The spectrogram model uses **two-stage transfer learning** for faster convergence and better accuracy:

| Stage | Strategy | Learning Rate | Epochs |
|-------|----------|---------------|--------|
| Stage 1 | Freeze EfficientNet-B0 backbone (ImageNet weights), train classifier head only | 1e-4 | 10 |
| Stage 2 | Unfreeze last backbone layers, fine-tune entire network | 1e-5 (backbone), 1e-4 (head) | 5 |

### Training Features

- **Pretrained + fine-tuning** transfer learning (ImageNet → deepfake detection)
- **Mixed precision training** (torch.cuda.amp) for GPU acceleration
- **Gradient clipping** for stable training
- **AdamW** optimizer with cosine-annealing learning rate schedule
- **Early stopping** to prevent overfitting
- **Checkpoint management** — saves best model, periodic snapshots, and final weights
- **Comprehensive metrics** — accuracy, precision, recall, F1, ROC-AUC, EER
- **Data loader prefetching** with pinned memory for throughput
- **Advanced augmentations** — Gaussian noise, pitch shift, time stretch, reverb, time masking
- **Experiment tracking** — JSON-based experiment logs with system metadata
- **Resume support** — continue training from any checkpoint
- **Model export** — PyTorch (.pt) and ONNX (.onnx) formats

### Quick Train

```bash
cd voice-ai

# Train spectrogram model (pretrained + fine-tuning, recommended)
python train.py --model spectrogram --data-root /data/ASVspoof2024/LA

# Train codec detector
python train.py --model codec --data-root /data/ASVspoof2024/LA

# Train both models
python train.py --model all --data-root /data/ASVspoof2024/LA

# Full pipeline: train → evaluate → export
python train.py --model all --data-root /data/ASVspoof2024/LA --full-pipeline

# Legacy from-scratch training (no pretrained weights)
python train.py --model spectrogram --data-root /data/ASVspoof2024/LA --no-pretrained --epochs 30

# Resume from a checkpoint
python train.py --model spectrogram --data-root /data/ASVspoof2024/LA --resume checkpoints/spec_best.pt
```

### Individual Training Scripts

```bash
cd voice-ai

# Train spectrogram model with custom stage settings
python -m training.train_spectrogram --data-root /data/ASVspoof2024/LA \
    --stage1-epochs 10 --stage2-epochs 5

# Train codec detector
python -m training.train_codec --data-root /data/ASVspoof2024/LA --epochs 15

# Master pipeline (all steps)
python -m training.train_all --data-root /data/ASVspoof2024/LA

# Evaluate trained models
python -m evaluation.evaluate_models --data-root /data/ASVspoof2024/LA \
    --checkpoint-dir checkpoints

# Export models to PyTorch + ONNX
python -m export.export_models --checkpoint-dir checkpoints --output-dir models
```

### Training CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `all` | `spectrogram`, `codec`, or `all` |
| `--data-root` | *(required)* | Path to ASVspoof 2024 `LA/` directory |
| `--epochs` | `30` | Maximum training epochs (legacy mode) |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `3e-4` | Peak learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--checkpoint-dir` | `checkpoints` | Output directory for model weights |
| `--resume` | — | Path to checkpoint to resume from |
| `--patience` | `7` | Early-stopping patience (epochs) |
| `--no-pretrained` | — | Use from-scratch training (legacy mode) |
| `--full-pipeline` | — | Run train → evaluate → export |

After training, use `python -m export.export_models --checkpoint-dir checkpoints` to convert the trained models to ONNX format for production deployment.

### Training Pipeline Architecture

```
voice-ai/
├── datasets/                  # Dataset download & preparation
│   ├── download_datasets.py   # Multi-dataset download automation
│   └── dataset_builder.py     # Unified real/fake dataset builder
├── preprocessing/             # Audio preprocessing
│   ├── audio_processor.py     # Resample, normalize, trim/pad
│   ├── spectrogram_generator.py  # 128×128 mel spectrogram
│   └── augmentations.py       # Noise, pitch, stretch, reverb
├── models/
│   ├── spectrogram_model.py   # Pretrained EfficientNet-B0 (freeze/unfreeze)
│   ├── codec_detector.py      # 1-D CNN codec artifact detector
│   └── ensemble.py            # Runtime ensemble (unchanged)
├── training/                  # Training pipeline
│   ├── trainer.py             # Core engine (AMP, grad clip, early stop)
│   ├── train_spectrogram.py   # Two-stage transfer learning
│   ├── train_codec.py         # Codec detector training
│   └── train_all.py           # Master pipeline script
├── evaluation/                # Model evaluation
│   ├── metrics.py             # Accuracy, F1, ROC-AUC, EER
│   └── evaluate_models.py     # Checkpoint evaluation
├── experiments/               # Experiment tracking
│   └── experiment_logger.py   # JSON-based experiment logs
└── export/                    # Model export
    └── export_models.py       # PyTorch + ONNX export
```

---

## Quick Start

### Prerequisites

- Docker 24+ and Docker Compose v2
- (Optional) Node.js 20+ for blockchain deployment

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your secrets
```

### 2. Start all services

```bash
docker-compose up
```

The full platform will start. Services come up in dependency order:

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8080 |
| Voice AI | http://localhost:8001 |
| Adversarial Engine | http://localhost:8002 |
| ZK Proof System | http://localhost:8003 |

### 3. (Optional) Deploy smart contract

```bash
cd blockchain
npm install
# Set DEPLOYER_PRIVATE_KEY and POLYGON_AMOY_RPC_URL in .env
npm run deploy:amoy
# Update TRUST_REGISTRY_ADDRESS in .env
```

---

## API Reference

### Voice AI Service (`:8001`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/voice/enroll` | Enrol user speaker embedding |
| `POST` | `/api/voice/verify` | Verify voice against enrolled embedding |
| `WS` | `/ws/voice/stream/{session_id}` | Real-time 2s chunk streaming |

**Enroll:**
```bash
curl -X POST http://localhost:8001/api/voice/enroll \
  -F "user_id=alice" \
  -F "audio=@voice_sample.wav"
```

**Verify:**
```bash
curl -X POST http://localhost:8001/api/voice/verify \
  -F "user_id=alice" \
  -F "audio=@voice_challenge.wav"
```

### Adversarial Engine (`:8002`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/adversarial/perturb-frame` | Apply adversarial perturbation to video frame |
| `POST` | `/api/liveness/rppg` | rPPG liveness detection from frames |

**Perturb frame:**
```bash
curl -X POST http://localhost:8002/api/adversarial/perturb-frame \
  -F "algorithm=pgd" \
  -F "epsilon=0.03" \
  -F "pgd_steps=20" \
  -F "frame=@frame.jpg"
```

### ZK Proof System (`:8003`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/zk/register` | Register identity commitment |
| `POST` | `/api/zk/verify` | Generate ZK proof |

### Backend API (`:8080`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/verify` | Full verification pipeline |
| `GET` | `/api/history/{user}` | Verification history |
| `POST` | `/api/zk/register` | Proxy to ZK service |
| `POST` | `/api/adversarial/perturb-frame` | Proxy to adversarial engine |

---

## Frontend Dashboard

The Next.js dashboard at `http://localhost:3000` provides:

- **Live spectrogram visualization** — real-time mel-spectrogram canvas
- **Threat score gauge** — animated SVG gauge (0-100)
- **Component breakdown** — per-model score bars
- **Video feed + adversarial shield** — live camera with FGSM perturbation toggle
- **ZK proof status** — step-by-step proof pipeline visualization
- **Blockchain explorer widget** — Polygon Amoy TX links
- **QR certificate generator** — downloadable verification certificate

---

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| Frontend | Next.js 14, React 18, TypeScript, TailwindCSS, Framer Motion |
| Voice AI | Python 3.11, FastAPI, PyTorch, Librosa, SpeechBrain (ECAPA-TDNN) |
| Video Engine | Python 3.11, OpenCV, NumPy, SciPy |
| ZK Proof System | Rust 1.79, Axum, SHA-256 commitments |
| Backend API | Go 1.22, Gin, pgx |
| Blockchain | Solidity 0.8.24, Hardhat, Ethers.js v6, Polygon Amoy |
| Database | PostgreSQL 16 |
| Queue | Redis 7 Streams |
| Infrastructure | Docker, Docker Compose |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Voice verification latency | < 2 seconds |
| ZK proof generation | < 1 second |
| Blockchain confirmation | < 3 seconds (Polygon Amoy) |

---

## Project Structure

```
vajra/
├── docker-compose.yml          # Full stack orchestration
├── .env.example                # Environment variable template
├── README.md
│
├── voice-ai/                   # Layer 1A: AI deepfake detection
│   ├── Dockerfile
│   ├── main.py                 # FastAPI service
│   ├── train.py                # ML training pipeline
│   ├── schemas.py              # Pydantic models
│   ├── storage.py              # PostgreSQL embedding store
│   ├── requirements.txt
│   ├── models/
│   │   ├── ensemble.py         # Three-model weighted ensemble
│   │   ├── speaker.py          # ECAPA-TDNN speaker embedder
│   │   └── export.py           # ONNX model export utility
│   └── data/
│       └── asvspoof.py         # ASVspoof 2024 dataset loader
│
├── adversarial-engine/         # Layer 1B: Adversarial video shield + rPPG
│   ├── Dockerfile
│   ├── main.py                 # FastAPI service (FGSM, PGD, illumination)
│   └── requirements.txt
│
├── zk-proof-system/            # Layer 2: ZK attestation
│   ├── Dockerfile
│   ├── Cargo.toml
│   └── src/main.rs             # Rust/Axum ZK commitment service
│
├── backend/                    # Orchestration microservice (Go)
│   ├── Dockerfile
│   ├── go.mod
│   ├── cmd/server/main.go
│   └── internal/
│       ├── api/router.go       # Gin router + handlers
│       └── db/db.go            # PostgreSQL pool + migrations
│
├── blockchain/                 # Layer 3: Trust registry
│   ├── Dockerfile
│   ├── hardhat.config.js
│   ├── package.json
│   ├── contracts/
│   │   └── VajraTrustRegistry.sol
│   ├── scripts/
│   │   └── deploy.js
│   └── test/
│       └── VajraTrustRegistry.test.js
│
└── frontend/                   # Dashboard (Next.js 14)
    ├── Dockerfile
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.js
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   ├── page.tsx         # Main dashboard
        │   └── globals.css
        ├── components/
        │   ├── ThreatGauge.tsx
        │   ├── SpectrogramView.tsx
        │   ├── VideoFeed.tsx
        │   ├── ZKProofStatus.tsx
        │   ├── BlockchainExplorer.tsx
        │   ├── QRCertificate.tsx
        │   └── StatusBar.tsx
        ├── hooks/
        │   └── useVoiceStream.ts
        └── types/
            └── index.ts
```

---

## Security

- All biometric data processed in-memory, never logged
- ZK proofs commit to biometric checks without revealing raw data
- Blockchain records contain only hashes — no PII on-chain
- JWT authentication on backend API endpoints
- Docker network isolation between services

---

## License

MIT
