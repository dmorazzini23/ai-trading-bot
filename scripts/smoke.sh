#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from ai_trading.config import get_settings
S = get_settings()
print("settings_ok", {"interval": S.interval, "iterations": S.iterations, "seed": S.seed, "model_path": S.model_path})
PY

python - <<'PY'
import numpy as np, pandas as pd, pandas_ta
print("numpy", np.__version__, "hasNaN", hasattr(np, "NaN"))
print("pandas", pd.__version__)
print("pandas_ta OK")
PY

python - <<'PY'
import importlib
m = importlib.import_module("ai_trading.core.bot_engine")
print("bot_engine_import_ok", hasattr(m, "run_all_trades_worker"))
PY
