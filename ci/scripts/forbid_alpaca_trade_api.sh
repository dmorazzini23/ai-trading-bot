#!/usr/bin/env bash
set -euo pipefail

# Fail if legacy alpaca-trade-api is installed
if pip show alpaca-trade-api >/dev/null 2>&1; then
  echo "alpaca-trade-api detected; only alpaca-py is supported" >&2
  exit 1
fi
