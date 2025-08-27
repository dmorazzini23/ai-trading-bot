#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1

echo "== Import smoke =="
python -X dev -c "import ai_trading; print('IMPORT_OK')"

echo "== CLI dry-run =="
python -m ai_trading --dry-run || { echo 'dry-run failed'; exit 1; }

echo "== Heavy import check =="
python - <<'PY'
import importlib, sys
heavy={"torch","gymnasium","pandas","pyarrow","sklearn","matplotlib"}
before=set(sys.modules)
importlib.import_module('ai_trading.core.bot_engine')
after=set(sys.modules)
loaded=sorted(m for m in (after-before) if m.split('.')[0] in heavy)
print("HEAVY_LOADED_DURING_IMPORT:", loaded)
assert not loaded, f"Heavy stacks imported at import-time: {loaded}"
print("HEAVY_IMPORT_OK")
PY

echo "== Ruff (core) =="
if command -v ruff >/dev/null 2>&1; then
  ruff check ai_trading/main.py ai_trading/core/bot_engine.py ai_trading/process_manager.py || true
else
  echo "ruff not installed; skipping lint"
fi

echo "== Selftest OK =="
