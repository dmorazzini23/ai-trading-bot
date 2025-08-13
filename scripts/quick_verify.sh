#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from ai_trading.settings import get_settings
S = get_settings()
print("interval:", S.interval, type(S.interval))
print("iterations:", S.iterations, type(S.iterations))
print("seed:", S.seed, type(S.seed))
print("model_path_abs:", S.model_path_abs)
print("trade_cooldown_min:", S.trade_cooldown_min)
print("trade_cooldown:", S.trade_cooldown)
print("use_rl_agent:", S.use_rl_agent)
PY

python - <<'PY'
# smoke: heavy libs only imported if present or RL enabled
import importlib, sys
mods = ["numpy","pandas","pandas_ta","psutil"]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "MISSING:", e, file=sys.stderr)
PY
