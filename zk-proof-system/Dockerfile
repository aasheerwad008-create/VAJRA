FROM rust:1.79-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest and member manifests for layer caching
COPY Cargo.toml ./
COPY host/Cargo.toml ./host/
COPY guest/Cargo.toml ./guest/

# Stub source files so Cargo can resolve the dependency graph
RUN mkdir -p host/src guest/src && \
    echo 'fn main() {}' > host/src/main.rs && \
    echo 'fn main() {}' > guest/src/main.rs

RUN cargo build --release --package zk-proof-system 2>&1 || true

# Copy real source files
COPY host/src ./host/src
COPY guest/src ./guest/src

# Touch to invalidate cached stubs
RUN touch host/src/main.rs guest/src/main.rs

RUN cargo build --release --package zk-proof-system 2>&1

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/zk-proof-system .

EXPOSE 8003

CMD ["./zk-proof-system"]
