#!/usr/bin/env bash
set -e

echo "[+] Running benchmarks..."
make benchmark

echo "[+] Profiling indicators..."
make profile

echo "[+] Running backtest..."
make backtest

echo "[+] Running grid search..."
make gridsearch

echo "[+] All done!"
