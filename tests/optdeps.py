import pytest

HINT = {
    "pandas":       'pip install "ai-trading-bot[pandas]"',
    "matplotlib":   'pip install "ai-trading-bot[plot]"',
    "sklearn":      'pip install "ai-trading-bot[ml]"',
    "torch":        'pip install "ai-trading-bot[ml]"',
    "ta":           'pip install "ai-trading-bot[ta]"',
    "talib":        'pip install "ai-trading-bot[ta]"',
}

def require(pkg: str):
    """Skip current test module unless `pkg` is importable, with a clear install hint."""
    return pytest.importorskip(pkg, reason=f"Install with: {HINT.get(pkg, 'pip install ' + pkg)}")
