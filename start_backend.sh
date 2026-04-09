#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV=".venv"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CHILD_PID=""

stop_backend() {
    echo ""
    echo "Stopping backend..."
    [ -n "$CHILD_PID" ] && kill "$CHILD_PID" 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    echo "Backend stopped."
    exit 0
}

if [ "${1:-}" = "stop" ]; then
    fuser -k 8000/tcp 2>/dev/null || true
    echo "Backend stopped."
    exit 0
fi

trap stop_backend SIGINT SIGTERM

# Stop any existing instance before starting
fuser -k 8000/tcp 2>/dev/null || true
sleep 0.5

echo "Starting FastAPI backend..."

source "$VENV/bin/activate"
cd "$SCRIPT_DIR"

python3 -m uvicorn backend.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    2>&1 | tee "$LOG_DIR/backend.log" &
CHILD_PID=$!

echo "Backend started (pid $CHILD_PID). Press Ctrl+C to stop."
wait "$CHILD_PID"
