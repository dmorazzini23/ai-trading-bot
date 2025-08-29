from ai_trading.core import bot_engine


def test_parse_local_positions_creates_trade_log(tmp_path, monkeypatch):
    """Smoke test: reading positions initializes the trade log file."""

    log_path = tmp_path / "trades.csv"
    # Point the bot engine to our temporary log file and reset singleton
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    # Ensure file does not yet exist
    assert not log_path.exists()

    # Parsing positions should trigger trade log initialization
    positions = bot_engine._parse_local_positions()

    assert positions == {}
    assert log_path.exists()
    assert log_path.stat().st_size > 0

