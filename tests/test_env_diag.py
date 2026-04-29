from ai_trading.diagnostics.env_diag import gather_alpaca_diag


def test_gather_alpaca_diag_reports_missing_base_url(monkeypatch):
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)

    diag = gather_alpaca_diag()

    assert diag["trading_base_url"] == ""
    assert diag["trading_base_url_missing"] is True
    assert diag["environment"] == "missing"
    assert diag["paper"] is False
