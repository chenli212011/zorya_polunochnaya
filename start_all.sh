#!/usr/bin/env bash
# start_all.sh — bring up the full stack in one shot:
#   1. Main FastAPI app on :APP_PORT (default 8070)
#   2. cloudflared quick tunnel pointing at :MCP_PORT
#   3. MCP server on :MCP_PORT, with MCP_PUBLIC_URL set to the tunnel URL
#
# Ctrl+C stops all three cleanly.
#
# Each service's stdout/stderr is captured to its own logfile under logs/,
# and tailed in this terminal with a [prefix] so you can see everything.

set -euo pipefail
cd "$(dirname "$0")"

APP_PORT="${APP_PORT:-8070}"
MCP_PORT="${MCP_PORT:-8071}"
LOG_DIR="logs"
APP_LOG="$LOG_DIR/server.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"
MCP_LOG="$LOG_DIR/mcp-server.log"

mkdir -p "$LOG_DIR"

APP_PID=""
TUNNEL_PID=""
MCP_PID=""
CLEANED_UP=0

# ----- helpers --------------------------------------------------------------

ensure_venv() {
  local venv="$1" reqs="$2"
  if [ ! -d "$venv" ]; then
    echo "[setup] Creating virtualenv $venv"
    python3 -m venv "$venv"
  fi
  echo "[setup] Installing $reqs into $venv"
  "$venv/bin/pip" install --quiet --upgrade pip
  "$venv/bin/pip" install --quiet -r "$reqs"
}

port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_http_200() {
  # wait_http_200 <url> <max_seconds>
  local url="$1" max="$2"
  for _ in $(seq 1 $((max * 2))); do
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q 200; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

cleanup() {
  [ "$CLEANED_UP" = "1" ] && return
  CLEANED_UP=1
  trap '' INT TERM
  echo
  echo "[shutdown] Stopping services..."
  # SIGTERM all direct children of this script (servers, tails, etc.)
  pkill -TERM -P $$ 2>/dev/null || true
  # Best-effort: also explicitly term the known service PIDs
  for pid in "$APP_PID" "$TUNNEL_PID" "$MCP_PID"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  # Anything still alive gets SIGKILL
  pkill -KILL -P $$ 2>/dev/null || true
  for pid in "$APP_PID" "$TUNNEL_PID" "$MCP_PID"; do
    [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "[shutdown] Done."
}
trap cleanup INT TERM EXIT

# ----- preflight ------------------------------------------------------------

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found in PATH. Install it (or check /usr/local/bin/cloudflared)." >&2
  exit 1
fi

for p in "$APP_PORT" "$MCP_PORT"; do
  if port_in_use "$p"; then
    echo "ERROR: port $p is already in use. Aborting." >&2
    exit 1
  fi
done

ensure_venv ".venv"     "requirements.txt"
ensure_venv ".venv-mcp" "requirements-mcp.txt"

# ----- 1. main app ----------------------------------------------------------

echo "[setup] Starting main app on :$APP_PORT"
: > "$APP_LOG"
.venv/bin/uvicorn app:app --host 127.0.0.1 --port "$APP_PORT" --log-level info >> "$APP_LOG" 2>&1 &
APP_PID=$!

if ! wait_http_200 "http://localhost:$APP_PORT/api/health" 15; then
  echo "ERROR: main app didn't become healthy in 15s. See $APP_LOG" >&2
  exit 1
fi
echo "[setup] Main app ready  (pid $APP_PID)"

# ----- 2. cloudflared -------------------------------------------------------

echo "[setup] Starting cloudflared tunnel for :$MCP_PORT"
: > "$TUNNEL_LOG"
cloudflared tunnel --url "http://localhost:$MCP_PORT" >> "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

TUNNEL_URL=""
for _ in $(seq 1 60); do
  TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
  [ -n "$TUNNEL_URL" ] && break
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "ERROR: cloudflared exited before producing a tunnel URL. See $TUNNEL_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [ -z "$TUNNEL_URL" ]; then
  echo "ERROR: cloudflared didn't produce a tunnel URL within 60s. See $TUNNEL_LOG" >&2
  exit 1
fi
echo "[setup] Tunnel ready    (pid $TUNNEL_PID)  → $TUNNEL_URL"

# ----- 3. MCP server --------------------------------------------------------

echo "[setup] Starting MCP server on :$MCP_PORT (MCP_PUBLIC_URL=$TUNNEL_URL)"
: > "$MCP_LOG"
MCP_PUBLIC_URL="$TUNNEL_URL" .venv-mcp/bin/python -u mcp_server.py >> "$MCP_LOG" 2>&1 &
MCP_PID=$!

if ! wait_http_200 "http://localhost:$MCP_PORT/health" 15; then
  echo "ERROR: MCP server didn't become healthy in 15s. See $MCP_LOG" >&2
  exit 1
fi
echo "[setup] MCP server ready (pid $MCP_PID)"

# ----- banner ---------------------------------------------------------------

cat <<BANNER

=============================================================================
  Zorya Polunochnaya — full stack running
=============================================================================

  Main app:    http://localhost:$APP_PORT
  MCP server:  http://localhost:$MCP_PORT
  Tunnel URL:  $TUNNEL_URL

  >>> Paste this URL into claude.ai → Settings → Connectors → Add custom
      connector:

          $TUNNEL_URL

  Logs:
    [app]     $APP_LOG
    [tunnel]  $TUNNEL_LOG
    [mcp]     $MCP_LOG

  Press Ctrl+C to stop everything.

=============================================================================
BANNER

# ----- multiplexed log tail -------------------------------------------------
# awk fflush() instead of `sed -u` because macOS sed doesn't support -u.

( tail -F "$APP_LOG"    2>/dev/null | awk '{print "[app]    " $0; fflush()}' ) &
( tail -F "$TUNNEL_LOG" 2>/dev/null | awk '{print "[tunnel] " $0; fflush()}' ) &
( tail -F "$MCP_LOG"    2>/dev/null | awk '{print "[mcp]    " $0; fflush()}' ) &

# ----- monitor: exit if any service dies ------------------------------------

while kill -0 "$APP_PID" "$TUNNEL_PID" "$MCP_PID" 2>/dev/null; do
  sleep 2
done
echo
echo "[monitor] One of the services exited unexpectedly. Tearing down..."
