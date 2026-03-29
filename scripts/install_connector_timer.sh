#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SRC="${REPO_ROOT}/packaging/systemd/ai-trading-connectors.service"
TIMER_SRC="${REPO_ROOT}/packaging/systemd/ai-trading-connectors.timer"
SERVICE_DST="/etc/systemd/system/ai-trading-connectors.service"
TIMER_DST="/etc/systemd/system/ai-trading-connectors.timer"

sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
sudo install -m 0644 "${TIMER_SRC}" "${TIMER_DST}"
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-connectors.timer
sudo systemctl start ai-trading-connectors.service

echo "=== connector timer status ==="
sudo systemctl status ai-trading-connectors.timer --no-pager -l | sed -n '1,120p'
echo "=== connector service recent logs ==="
sudo journalctl -u ai-trading-connectors.service -n 80 -o cat --no-pager
