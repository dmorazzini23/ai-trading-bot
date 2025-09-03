from ai_trading.core import bot_engine


def test_parse_local_positions_creates_trade_log(tmp_path, monkeypatch):
    """Smoke test: reading positions initializes the trade log file."""

    log_path = tmp_path / "trades.jsonl"
    # Point the bot engine to our temporary log file and reset singleton
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    # Ensure file does not yet exist
    assert not log_path.exists()

    # Parsing positions should trigger trade log initialization
    positions = bot_engine._parse_local_positions()

    assert positions == {}
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_trade_logger_records_entry(tmp_path, monkeypatch):
    """Trade logger writes header then appends entries on first use."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("AAPL", 100.0, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert len(lines) == 2
    assert "AAPL" in lines[1]


def test_get_trade_logger_creates_header_when_missing(tmp_path, monkeypatch):
    """get_trade_logger should create the file with a header if absent."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    bot_engine.get_trade_logger()

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_read_trade_log_initializes_file_with_header(tmp_path, monkeypatch):
    """_read_trade_log initializes missing file and writes header."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    df = bot_engine._read_trade_log(str(log_path))

    assert df is None
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_existing_empty_log_gets_header_and_entry(tmp_path, monkeypatch):
    """Existing empty log gets header and first entry on startup."""

    log_path = tmp_path / "trades.jsonl"
    log_path.touch()
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("MSFT", 123.0, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert len(lines) == 2
    assert "MSFT" in lines[1]

