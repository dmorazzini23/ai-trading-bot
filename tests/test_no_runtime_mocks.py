import pathlib
import re

MOCK_PATTERNS = [
    r"class\s+Mock\w+\s*\(",
    r"class\s+_Mock\w+\s*\(",
]


def test_no_mocks_in_production_tree():
    root = pathlib.Path(__file__).resolve().parents[1] / "ai_trading"
    offenders = []
    for p in root.rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if any(re.search(pat, txt) for pat in MOCK_PATTERNS):
            offenders.append(str(p))
    assert (
        not offenders
    ), f"Mock classes must live under tests/mocks, found in: {offenders}"
