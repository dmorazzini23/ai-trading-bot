import pytest


def test_main_exits_when_alpaca_sdk_missing(monkeypatch, caplog):
    import ai_trading.main as m

    monkeypatch.setattr(m, "ALPACA_AVAILABLE", False)
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit) as excinfo:
            m.main([])
    assert excinfo.value.code == 1
    assert any("pip install alpaca-py" in record.getMessage() for record in caplog.records)
