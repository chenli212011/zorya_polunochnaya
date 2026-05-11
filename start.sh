#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PORT=8070
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/server.log"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV" ]; then
  echo "[start] Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[start] Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[start] Port $PORT is already in use. Aborting." >&2
  exit 1
fi

echo "[start] Launching server on http://localhost:$PORT"
echo "[start] Logs streaming below (also written to $LOG_FILE). Press Ctrl+C to stop."
echo "------------------------------------------------------------"

# Unbuffered Python output so logs appear in real time, tee to file + stdout.
exec python -u -m uvicorn app:app --host 127.0.0.1 --port "$PORT" --log-level info 2>&1 | tee -a "$LOG_FILE"
