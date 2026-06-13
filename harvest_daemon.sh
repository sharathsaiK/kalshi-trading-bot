#!/bin/bash
# harvest_daemon.sh — runs continuously, harvesting training + holdout data every hour.
# Managed by launchd: ~/Library/LaunchAgents/com.kalshi.harvest.plist

KALSHI_DIR="$HOME/Projects/kalshi"
PYTHON="$KALSHI_DIR/venv/bin/python3"
SCRIPT="$KALSHI_DIR/harvest_training_data.py"
INTERVAL=3600  # seconds between harvests (1 hour)

cd "$KALSHI_DIR"

while true; do
    echo ""
    echo "=========================================="
    echo "HARVEST START: $(date)"
    echo "=========================================="

    echo "--- Training data ---"
    "$PYTHON" -u "$SCRIPT" --all

    echo ""
    echo "--- Holdout data ---"
    "$PYTHON" -u "$SCRIPT" --all --holdout

    echo ""
    echo "HARVEST DONE: $(date). Sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
done
