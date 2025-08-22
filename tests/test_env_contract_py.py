from __future__ import annotations
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [p for p in ROOT.rglob("*.py")
         if "tests" not in str(p)
         and "venv" not in str(p)
         and ".venv" not in str(p)
         and "node_modules" not in str(p)]

def _t(p): return p.read_text(encoding="utf-8", errors="ignore")

def test_no_apca_env_anywhere_in_python():
    offenders = [p for p in FILES if "APCA_" in _t(p)]
    assert not offenders, f"Forbidden APCA_* in: {offenders}"

def _find_bad_ctor(pattern, need_kw):
    rx = re.compile(pattern, re.DOTALL)
    bad = []
    for p in FILES:
        txt = _t(p)
        for m in rx.finditer(txt):
            if need_kw not in m.group(0):
                bad.append(p); break
    return bad

def test_all_python_has_explicit_alpaca_creds():
    bad = []
    bad += _find_bad_ctor(r"\bREST\s*\([^)]*\)", "key_id=")
    bad += _find_bad_ctor(r"\bTradingClient\s*\([^)]*\)", "api_key=")
    bad += _find_bad_ctor(r"\bStockHistoricalDataClient\s*\([^)]*\)", "api_key=")
    bad += _find_bad_ctor(r"\bCryptoHistoricalDataClient\s*\([^)]*\)", "api_key=")
    assert not bad, f"Missing explicit Alpaca creds in: {sorted(set(bad))}"
