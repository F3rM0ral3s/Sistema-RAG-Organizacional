#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

stop_nginx() {
    echo ""
    echo "Stopping NGINX..."
    if [ -f /tmp/nginx.pid ]; then
        kill "$(cat /tmp/nginx.pid)" 2>/dev/null || true
    fi
    fuser -k 80/tcp 2>/dev/null || true
    echo "NGINX stopped."
    exit 0
}

if [ "${1:-}" = "stop" ]; then
    stop_nginx
fi

trap stop_nginx SIGINT SIGTERM

# Stop any existing instance before starting
if [ -f /tmp/nginx.pid ]; then
    kill "$(cat /tmp/nginx.pid)" 2>/dev/null || true
fi
fuser -k 80/tcp 2>/dev/null || true
sleep 0.5

echo "Starting NGINX..."
nginx -c "$SCRIPT_DIR/nginx/nginx.conf"
echo "NGINX started (pid $(cat /tmp/nginx.pid)). Press Ctrl+C to stop."

# Stay alive so the trap can catch Ctrl+C
while kill -0 "$(cat /tmp/nginx.pid)" 2>/dev/null; do
    sleep 1
done
