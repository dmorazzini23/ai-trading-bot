import json
import subprocess
import sys

PATHS = [
    "ai_trading/logging.py",
    "ai_trading/risk/engine.py",
    "ai_trading/broker/alpaca.py",
    "ai_trading/strategies/mean_reversion.py",
    "ai_trading/execution/liquidity.py",
    "ai_trading/portfolio/optimizer.py",
    "ai_trading/strategies/regime_detection.py",
    "ai_trading/position/legacy_manager.py",
    "ai_trading/strategies/signals.py",
    "ai_trading/strategies/multi_timeframe.py",
    "ai_trading/position/correlation_analyzer.py",
    "ai_trading/position/profit_taking.py",
    "ai_trading/production_system.py",
    "ai_trading/rl_trading/train.py",
    "ai_trading/utils/base.py",
    "ai_trading/position/intelligent_manager.py",
    "ai_trading/risk/position_sizing.py",
    "ai_trading/strategies/metalearning.py",
    "ai_trading/risk/adaptive_sizing.py",
    "ai_trading/risk/manager.py",
]

p = subprocess.run([sys.executable, "tools/audit_exceptions.py", "--paths", *PATHS], capture_output=True, text=True)
assert p.returncode == 0, p.stderr
data = json.loads(p.stdout.splitlines()[0])
offenders = {f: hits for f, hits in data.get("by_file", {}).items() if hits}
assert offenders == {}, f"broad except present in: {list(offenders)}"
