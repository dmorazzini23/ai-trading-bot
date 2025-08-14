import pathlib
import re


def test_no_legacy_trade_execution_imports():
    root = pathlib.Path(__file__).resolve().parent
    bad = []
    for p in root.rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\bfrom\s+ai_trading\s+import\s+trade_execution\b", txt) or \
           re.search(r"\bimport\s+ai_trading\.trade_execution\b", txt):
            bad.append(str(p))
    assert not bad, f"Legacy trade_execution import found in: {bad}"
