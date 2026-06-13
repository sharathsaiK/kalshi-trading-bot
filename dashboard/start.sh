#!/bin/bash
# Starts the API backend + dashboard dev server together.
# Run: bash start.sh  (from the dashboard/ directory)

REPO="$HOME/Projects/AI-Futures-Trader"
DASHBOARD="$(cd "$(dirname "$0")" && pwd)"

echo "[start] API server..."
cd "$REPO"
source venv/bin/activate
python3 api_server.py &
API_PID=$!

echo "[start] Dashboard..."
cd "$DASHBOARD"
npm run dev &
DASH_PID=$!

# Kill both on Ctrl+C
trap "kill $API_PID $DASH_PID 2>/dev/null; exit" INT TERM
wait
