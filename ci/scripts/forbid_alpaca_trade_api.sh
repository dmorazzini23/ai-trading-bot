#!/usr/bin/env bash
set -euo pipefail

# Ensure the runtime Alpaca SDK is present at the supported version
REQUIRED_VERSION="3.2.0"
if ! pip show alpaca-trade-api >/dev/null 2>&1; then
  echo "alpaca-trade-api==${REQUIRED_VERSION} is required" >&2
  exit 1
fi

version=$(pip show alpaca-trade-api | awk '/^Version: / {print $2}')
if [ "${version}" != "${REQUIRED_VERSION}" ]; then
  echo "alpaca-trade-api==${REQUIRED_VERSION} required; found ${version}" >&2
  exit 1
fi
