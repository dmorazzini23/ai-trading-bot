from __future__ import annotations
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLOBS = [
    "ai_trading/core/**/*.py",
    "ai_trading/execution/**/*.py",
    "ai_trading/risk/**/*.py",
    "trade_execution/**/*.py",
]

FILES = []
for g in GLOBS:
    FILES += [p for p in ROOT.glob(g)]

def _t(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def test_no_apca_env_in_core():
    offenders = [p for p in FILES if "APCA_" in _t(p)]
    assert not offenders, f"Forbidden APCA_* in: {offenders}"

def _find_bad_ctor(pattern: str, need_kw: str):
    bad = []
    rx = re.compile(pattern, re.DOTALL)
    for p in FILES:
        txt = _t(p)
        for m in rx.finditer(txt):
            if need_kw not in m.group(0):
                bad.append(p); break
    return bad

def test_explicit_creds_in_core():
    bad = []
    bad += _find_bad_ctor(r"\bREST\s*\([^)]*\)", "key_id=")
    bad += _find_bad_ctor(r"\bTradingClient\s*\([^)]*\)", "api_key=")
    bad += _find_bad_ctor(r"\bStockHistoricalDataClient\s*\([^)]*\)", "api_key=")
    bad += _find_bad_ctor(r"\bCryptoHistoricalDataClient\s*\([^)]*\)", "api_key=")
    assert not bad, f"Missing explicit Alpaca creds in: {sorted(set(bad))}"
