#!/usr/bin/env bash
# Invoked by ai-trading-systemd-failure-alert@.service when a unit fails.
# Thin delegation to the connector dispatcher; it holds the Slack webhook logic.
set -euo pipefail
unit="${1:?usage: systemd_failure_alert.sh <unit-name>}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo}/venv/bin/python" "${repo}/scripts/connector_incident_dispatch.py" --unit-failed "${unit}"
