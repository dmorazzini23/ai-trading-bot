#!/usr/bin/env bash
set -euo pipefail

# Ensure the runtime Alpaca SDK is present at the supported version.
REQUIRED_VERSION="0.42.1"
if ! pip show alpaca-py >/dev/null 2>&1; then
  echo "alpaca-py==${REQUIRED_VERSION} is required" >&2
  exit 1
fi

version=$(pip show alpaca-py | awk '/^Version: / {print $2}')
if [ "${version}" != "${REQUIRED_VERSION}" ]; then
  echo "alpaca-py==${REQUIRED_VERSION} required; found ${version}" >&2
  exit 1
fi
