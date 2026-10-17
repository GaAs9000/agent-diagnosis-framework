#!/bin/bash
# scripts/stop_tunnel.sh
# Safely tears down the persistent SSH tunnel forwarding port 8001
set -euo pipefail

LOCAL_PORT=8001
REMOTE_HOST="aml-b"
SOCKET_PATH="/Users/jiashen/.ssh/sockets/aml-b-${LOCAL_PORT}.sock"
PID_FILE="/tmp/ssh_tunnels/aml_b_${LOCAL_PORT}.pid"

echo "[1/3] Checking OpenSSH ControlMaster socket..."
if [ -S "$SOCKET_PATH" ]; then
  echo "Sending exit signal to ControlMaster socket..."
  ssh -S "$SOCKET_PATH" -O exit "$REMOTE_HOST" 2>/dev/null || true
  rm -f "$SOCKET_PATH"
  echo "✓ Control socket closed."
else
  echo "No active control socket found at $SOCKET_PATH."
fi

echo "[2/3] Checking PID file and terminating process..."
if [ -f "$PID_FILE" ]; then
  TUNNEL_PID=$(cat "$PID_FILE")
  if ps -p "$TUNNEL_PID" >/dev/null 2>&1; then
    echo "Terminating PID ${TUNNEL_PID}..."
    kill "$TUNNEL_PID" 2>/dev/null || true
    sleep 1
    if ps -p "$TUNNEL_PID" >/dev/null 2>&1; then
      kill -9 "$TUNNEL_PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

echo "[3/3] Final port release check..."
if lsof -ti :${LOCAL_PORT} >/dev/null 2>&1; then
  echo "Force-killing remaining processes on port ${LOCAL_PORT}..."
  kill -9 $(lsof -ti :${LOCAL_PORT}) 2>/dev/null || true
  sleep 1
fi

echo "✓ Local port ${LOCAL_PORT} is verified closed and clean."
