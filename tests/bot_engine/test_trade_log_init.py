from ai_trading.core import bot_engine
import pytest
import logging
from pathlib import Path


def _as_classmethod(path: Path):
    """Return a classmethod returning ``path`` for monkeypatching ``Path`` hooks."""

    return classmethod(lambda cls: path)


def test_parse_local_positions_creates_trade_log(tmp_path, monkeypatch):
    """Smoke test: reading positions initializes the trade log file."""

    log_path = tmp_path / "trades.jsonl"
    # Point the bot engine to our temporary log file and reset singleton
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

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
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

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
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    bot_engine.get_trade_logger()

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_read_trade_log_initializes_file_with_header(tmp_path, monkeypatch):
    """_read_trade_log initializes missing file and writes header."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

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
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("MSFT", 123.0, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert len(lines) == 2
    assert "MSFT" in lines[1]


def test_get_trade_logger_creates_missing_directory(tmp_path, monkeypatch):
    """get_trade_logger creates the parent directory when absent."""

    log_dir = tmp_path / "nested"
    log_path = log_dir / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    bot_engine.get_trade_logger()

    assert log_dir.exists()
    assert log_path.exists()


def test_get_trade_logger_falls_back_when_dir_not_writable(tmp_path, monkeypatch, caplog):
    """get_trade_logger falls back to a user state dir when the target is read-only."""

    state_home = tmp_path / "state-home"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    log_dir = tmp_path / "readonly"
    log_dir.mkdir()
    log_dir.chmod(0o555)
    log_path = log_dir / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    fallback_path = state_home / "ai-trading-bot" / log_path.name
    assert logger_instance.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert fallback_path.exists()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bot_engine.get_trade_logger()

    assert not any(record.message == "TRADE_LOG_FALLBACK_USER_STATE" for record in caplog.records)


def test_get_trade_logger_falls_back_on_dir_creation_permission_error(tmp_path, monkeypatch, caplog):
    """get_trade_logger falls back when os.makedirs raises PermissionError."""

    state_home = tmp_path / "state-permission"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o555)
    log_path = parent / "child" / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    fallback_path = state_home / "ai-trading-bot" / log_path.name
    assert logger_instance.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert fallback_path.exists()
    assert not caplog.records


def test_trade_log_fallback_uses_tempdir_when_everything_blocked(tmp_path, monkeypatch):
    """Trade log path resolves into a tempdir when no candidate directory is writable."""

    # Unwritable XDG state home so the primary fallback fails.
    state_home = tmp_path / "state-home"
    state_home.mkdir()
    state_home.chmod(0o555)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    # Unwritable home directory prevents ~/.local/state usage.
    home_dir = tmp_path / "home-readonly"
    home_dir.mkdir()
    home_dir.chmod(0o555)
    monkeypatch.setattr(bot_engine.Path, "home", _as_classmethod(home_dir))

    # Project directory with non-writable logs/ to block cwd-based fallback.
    project_root = tmp_path / "project"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True)
    logs_dir.chmod(0o555)
    monkeypatch.setattr(bot_engine.Path, "cwd", _as_classmethod(project_root))

    # Deterministic temp dir for this test.
    temp_parent = tmp_path / "tempdir"
    temp_parent.mkdir()
    monkeypatch.setattr(bot_engine.tempfile, "gettempdir", lambda: str(temp_parent))

    blocked_roots = [state_home.resolve(), home_dir.resolve(), logs_dir.resolve()]

    original_is_dir_writable = bot_engine._is_dir_writable

    def fake_is_dir_writable(path: str) -> bool:
        resolved = Path(path).resolve(strict=False)
        for root in blocked_roots:
            try:
                if resolved.is_relative_to(root):
                    return False
            except ValueError:
                continue
        return original_is_dir_writable(path)

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)

    log_name = "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(logs_dir / log_name))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger_instance = bot_engine.get_trade_logger()

    expected_dir = temp_parent / "ai-trading-bot"
    expected_path = expected_dir / log_name
    assert logger_instance.path == str(expected_path)
    assert bot_engine.TRADE_LOG_FILE == str(expected_path)
    assert expected_path.exists()
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(expected_path)
