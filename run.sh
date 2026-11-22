#!/usr/bin/env bash
# One-command local start for macOS / Linux
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
echo "Starting AgenticRAG-document-assistant at http://127.0.0.1:8000"
exec python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
