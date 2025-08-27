from __future__ import annotations
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [p for p in ROOT.rglob("*.py")
         if "tests" not in str(p)
         and "venv" not in str(p)
         and ".venv" not in str(p)
         and "node_modules" not in str(p)]
PY = [p for p in ROOT.rglob("*.py") if "venv" not in str(p)]  # AI-AGENT-REF: include tests

def _t(p): return p.read_text(encoding="utf-8", errors="ignore")

def test_no_apca_env_anywhere_in_python():
    offenders = [p for p in FILES if "APCA_" in _t(p)]
    assert not offenders, f"Forbidden APCA_* in: {offenders}"

def _find_bad_ctor(pattern, need_kw, unless_kw=None):
    rx = re.compile(pattern, re.DOTALL)
    bad = []
    for p in FILES:
        txt = _t(p)
        for m in rx.finditer(txt):
            segment = m.group(0)
            if unless_kw and unless_kw in segment:
                continue
            if need_kw not in segment:
                bad.append(p)
                break
    return bad

def test_all_python_has_explicit_alpaca_creds():
    bad = []
    bad += _find_bad_ctor(r"\bREST\s*\([^)]*\)", "key_id=")
    bad += _find_bad_ctor(
        r"\bTradingClient\s*\([^)]*\)", "api_key=", unless_kw="oauth_token"
    )
    bad += _find_bad_ctor(
        r"\bStockHistoricalDataClient\s*\([^)]*\)",
        "api_key=",
        unless_kw="oauth_token",
    )
    bad += _find_bad_ctor(
        r"\bCryptoHistoricalDataClient\s*\([^)]*\)", "api_key="
    )
    assert not bad, f"Missing explicit Alpaca creds in: {sorted(set(bad))}"


def test_tradingclient_sets_paper_not_base_url():
    offenders_missing_paper = []
    offenders_with_base = []
    rx = re.compile(r"\bTradingClient\s*\((?P<args>[^)]*)\)", re.DOTALL)
    for p in PY:
        t = p.read_text(encoding="utf-8", errors="ignore")
        for m in rx.finditer(t):
            args = m.group("args")
            if "paper=" not in args:
                offenders_missing_paper.append(p)
            if "base_url=" in args:
                offenders_with_base.append(p)
    assert not offenders_with_base, (
        f"TradingClient must not set base_url=: {sorted(set(offenders_with_base))}"
    )
    assert not offenders_missing_paper, (
        f"TradingClient must set paper=: {sorted(set(offenders_missing_paper))}"
    )  # AI-AGENT-REF: enforce paper flag without base_url
