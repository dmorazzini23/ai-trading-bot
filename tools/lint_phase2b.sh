#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts

echo "Python & tool versions" > artifacts/tool-versions.txt
python -V          >> artifacts/tool-versions.txt 2>&1 || true
ruff --version     >> artifacts/tool-versions.txt 2>&1 || true
mypy --version     >> artifacts/tool-versions.txt 2>&1 || true
pytest --version   >> artifacts/tool-versions.txt 2>&1 || true

# Pass 1: quick auto-fix
ruff check . --fix --unsafe-fixes || true

# Pass 2: import sort & cleanup
ruff check . --select I,F,E,UP,PL,BLE --fix || true

# Final report
ruff check . > artifacts/ruff.txt || true
mypy ai_trading > artifacts/mypy.txt || true
pytest -n auto --disable-warnings -q > artifacts/pytest.txt || true

# Top rules summary
grep -oE "\[[A-Z0-9]+\]" artifacts/ruff.txt | sort | uniq -c | sort -nr | head -n 50 > artifacts/ruff-top-rules.txt || true
echo "Done. See artifacts/ for reports."
