import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

RUNTIME_DIRS = ("ai_trading",)
FORBIDDEN = ("from alpaca.data", "StockHistoricalDataClient", "CryptoHistoricalDataClient")

def test_no_alpaca_py_data_clients_in_runtime():
    offenders = []
    for p in ROOT.rglob("*.py"):
        s = str(p)
        if not s.startswith(str(ROOT / "ai_trading")):
            continue
        if "tests" in s:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if any(f in txt for f in FORBIDDEN):
            offenders.append(s)
    assert not offenders, (
        "alpaca-py data clients must not be used in runtime fetch path: " + str(offenders)
    )
