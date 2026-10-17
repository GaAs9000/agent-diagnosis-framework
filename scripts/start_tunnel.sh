#!/bin/bash
# scripts/start_tunnel.sh
# Establishes persistent SSH port forwarding from local macOS 127.0.0.1:8001 to aml-b 127.0.0.1:8001
set -euo pipefail

LOCAL_PORT=8001
REMOTE_PORT=8001
REMOTE_HOST="aml-b"
SOCKET_DIR="/Users/jiashen/.ssh/sockets"
SOCKET_PATH="${SOCKET_DIR}/aml-b-${LOCAL_PORT}.sock"
PID_DIR="/tmp/ssh_tunnels"
PID_FILE="${PID_DIR}/aml_b_${LOCAL_PORT}.pid"

mkdir -p "$SOCKET_DIR"
mkdir -p "$PID_DIR"

echo "[1/4] Checking local port ${LOCAL_PORT} availability..."
if lsof -ti :${LOCAL_PORT} >/dev/null 2>&1; then
  OCCUPYING_PID=$(lsof -ti :${LOCAL_PORT} | head -n 1)
  echo "WARNING: Port ${LOCAL_PORT} is currently occupied by PID ${OCCUPYING_PID}."
  echo "Attempting to release port ${LOCAL_PORT}..."
  kill "${OCCUPYING_PID}" 2>/dev/null || true
  sleep 1
  if lsof -ti :${LOCAL_PORT} >/dev/null 2>&1; then
    kill -9 "${OCCUPYING_PID}" 2>/dev/null || true
    sleep 1
  fi
fi

if lsof -ti :${LOCAL_PORT} >/dev/null 2>&1; then
  echo "ERROR: Unable to release port ${LOCAL_PORT}." >&2
  exit 1
fi
echo "✓ Port ${LOCAL_PORT} is free."

echo "[2/4] Cleaning any stale socket files..."
rm -f "$SOCKET_PATH"

echo "[3/4] Establishing persistent SSH tunnel..."
ssh -M -S "$SOCKET_PATH" \
    -f -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=10 \
    -o BatchMode=yes \
    -L 127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} \
    "$REMOTE_HOST"

echo "[4/4] Verifying tunnel process and recording PID..."
sleep 1
if ssh -S "$SOCKET_PATH" -O check "$REMOTE_HOST" 2>/dev/null; then
  TUNNEL_PID=$(lsof -ti :${LOCAL_PORT} | head -n 1)
  echo "$TUNNEL_PID" > "$PID_FILE"
  echo "✓ Persistent SSH tunnel successfully established!"
  echo "  - Local Endpoint:  http://127.0.0.1:${LOCAL_PORT}/v1"
  echo "  - Remote Host:     ${REMOTE_HOST}:${REMOTE_PORT}"
  echo "  - Control Socket:  ${SOCKET_PATH}"
  echo "  - Process PID:     ${TUNNEL_PID}"
else
  echo "ERROR: Failed to establish SSH tunnel." >&2
  exit 1
fi
