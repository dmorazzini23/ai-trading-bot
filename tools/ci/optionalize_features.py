from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ("ai_trading/utils/base.py", "pandas_market_calendars", "use_market_calendar_lib"),
    ("ai_trading/main.py", "performance_monitor", "enable_performance_monitoring"),
]

TEMPLATE = """from importlib.util import find_spec
from ai_trading.config import get_settings
S = get_settings()
if getattr(S, "{flag}", False):
    if find_spec("{module}") is None:
        raise RuntimeError("Feature enabled but module '{module}' not installed")
    from {module} import {symbol}  # noqa: F401
"""

def ensure_block(path: Path, module: str, flag: str, symbol: str = "*"):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    guard = f"find_spec(\"{module}\")"
    if guard in txt or f"ENABLE_{flag.upper()}" in txt:
        return False
    # Append a safe, reviewable block at end (human may inline later)
    path.write_text(txt + "\n\n" + TEMPLATE.format(flag=flag, module=module, symbol=symbol), encoding="utf-8")
    return True

def main():
    changed = 0
    for rel, module, flag in TARGETS:
        p = ROOT / rel
        if p.exists():
            changed += ensure_block(p, module, flag)

if __name__ == "__main__":
    main()
