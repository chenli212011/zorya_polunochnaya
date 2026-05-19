#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv-mcp"
PORT="${MCP_PORT:-8071}"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/mcp-server.log"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV" ]; then
  echo "[start-mcp] Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[start-mcp] Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements-mcp.txt

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[start-mcp] Port $PORT already in use. Aborting." >&2
  exit 1
fi

echo "[start-mcp] Launching MCP server on http://localhost:$PORT"
echo "[start-mcp] Logs streaming below (also written to $LOG_FILE). Ctrl+C to stop."
echo "------------------------------------------------------------"

exec python -u mcp_server.py 2>&1 | tee -a "$LOG_FILE"
