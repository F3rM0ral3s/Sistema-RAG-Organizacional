#!/bin/bash
set -euo pipefail

BIN="/workspace/llama.cpp/build/bin/llama-server"
MODEL="/workspace/models/Qwen3.5-4B-Q8_0.gguf"
HOST="0.0.0.0"
THREADS="$(nproc)"
BATCH=2048
UBATCH=512

LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

# ── Server 1: Light tasks (guard + expander) ────────────────────────────────
# High concurrency, small context — handles many short requests fast
LIGHT_PORT=8080
LIGHT_PARALLEL=16
LIGHT_CTX_PER_SLOT=2048
LIGHT_CTX=$((LIGHT_PARALLEL * LIGHT_CTX_PER_SLOT))
LIGHT_LOG="$LOG_DIR/llama-light.log"

# ── Server 2: Generator (RAG answer generation) ─────────────────────────────
# Low concurrency, large context — handles few heavy requests with 30 chunks
GEN_PORT=8082
GEN_PARALLEL=2
GEN_CTX_PER_SLOT=49152
GEN_CTX=$((GEN_PARALLEL * GEN_CTX_PER_SLOT))
GEN_LOG="$LOG_DIR/llama-generator.log"

# ── Lifecycle ────────────────────────────────────────────────────────────────
LIGHT_PID=""
GEN_PID=""

stop_all() {
    echo ""
    echo "Stopping llama-servers..."
    [ -n "$LIGHT_PID" ] && kill "$LIGHT_PID" 2>/dev/null || true
    [ -n "$GEN_PID" ] && kill "$GEN_PID" 2>/dev/null || true
    fuser -k "${LIGHT_PORT}/tcp" 2>/dev/null || true
    fuser -k "${GEN_PORT}/tcp" 2>/dev/null || true
    echo "All llama-servers stopped."
    exit 0
}

if [ "${1:-}" = "stop" ]; then
    fuser -k "${LIGHT_PORT}/tcp" 2>/dev/null || true
    fuser -k "${GEN_PORT}/tcp" 2>/dev/null || true
    echo "All llama-servers stopped."
    exit 0
fi

trap stop_all SIGINT SIGTERM

# Stop any existing instances
fuser -k "${LIGHT_PORT}/tcp" 2>/dev/null || true
fuser -k "${GEN_PORT}/tcp" 2>/dev/null || true
sleep 0.5

# ── Launch light server ─────────────────────────────────────────────────────
echo "Starting light server on :${LIGHT_PORT} (${LIGHT_PARALLEL} slots × ${LIGHT_CTX_PER_SLOT} ctx)..."
"$BIN" \
    -m "$MODEL" \
    --host "$HOST" --port "$LIGHT_PORT" \
    -c "$LIGHT_CTX" \
    -t "$THREADS" \
    --flash-attn on \
    -ngl 99 \
    --parallel "$LIGHT_PARALLEL" \
    --no-cache-prompt \
    --reasoning off \
    --reasoning-budget 0 \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --log-file "$LIGHT_LOG" &
LIGHT_PID=$!

# Wait for light server to be ready before starting the second instance
echo "Waiting for light server to be ready..."
for i in $(seq 1 60); do
    if curl -s -o /dev/null "http://127.0.0.1:${LIGHT_PORT}/health" 2>/dev/null; then
        echo "Light server ready."
        break
    fi
    sleep 2
done

# ── Launch generator server ─────────────────────────────────────────────────
echo "Starting generator server on :${GEN_PORT} (${GEN_PARALLEL} slots × ${GEN_CTX_PER_SLOT} ctx)..."
"$BIN" \
    -m "$MODEL" \
    --host "$HOST" --port "$GEN_PORT" \
    -c "$GEN_CTX" \
    -t "$THREADS" \
    --flash-attn on \
    -ngl 99 \
    --parallel "$GEN_PARALLEL" \
    --no-cache-prompt \
    --reasoning off \
    --reasoning-budget 0 \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --log-file "$GEN_LOG" &
GEN_PID=$!

echo "Light server pid=$LIGHT_PID, Generator server pid=$GEN_PID"
echo "Press Ctrl+C to stop both."
wait -n "$LIGHT_PID" "$GEN_PID" 2>/dev/null || true
# If one exits, stop the other
stop_all
