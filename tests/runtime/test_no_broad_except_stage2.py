import json
import subprocess
import sys
from pathlib import Path  # noqa: F401

MODULES = [
    "ai_trading/strategies/momentum.py",
    "ai_trading/strategies/mean_reversion.py",
    "ai_trading/execution/liquidity.py",
    "ai_trading/execution/order_router.py",
    "ai_trading/position/calculators.py",
    "ai_trading/risk/engine.py",
    "ai_trading/rl_trading/train.py",
    "ai_trading/monitoring/aggregator.py",
    "ai_trading/monitoring/processor.py",
    "ai_trading/signals/factors.py",
    "ai_trading/portfolio/optimizer.py",
    "ai_trading/portfolio/risk_parity.py",
    "ai_trading/portfolio/turnover.py",
    "ai_trading/data/cleanroom.py",
    "ai_trading/data/pipeline.py",
]

def test_stage2_modules_have_no_broad_except():
    cmd = [sys.executable, "tools/audit_exceptions.py", "--paths", *MODULES]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert p.returncode == 0
    first_line = p.stdout.splitlines()[0]
    data = json.loads(first_line)
    offenders = {f: hits for f, hits in data.get("by_file", {}).items() if hits}
    assert offenders == {}, f"broad except present in: {list(offenders.keys())}"
