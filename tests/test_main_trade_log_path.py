from pathlib import Path

import pytest

import ai_trading.main_trade_log_path as main_trade_log_path
import ai_trading.core.bot_engine as bot_engine


def test_default_trade_log_path_env_override_fallback(tmp_path, monkeypatch):
    """default_trade_log_path falls back when env override is unwritable."""

    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    monkeypatch.setenv("TRADE_LOG_PATH", str(locked_dir / "trades.jsonl"))
    monkeypatch.delenv("AI_TRADING_TRADE_LOG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    original_access = bot_engine.os.access

    def fake_access(path, mode):
        if Path(path) == locked_dir:
            return False
        return original_access(path, mode)

    monkeypatch.setattr(bot_engine.os, "access", fake_access)

    resolved = Path(bot_engine.default_trade_log_path())

    assert resolved == tmp_path / "logs" / "trades.jsonl"
    assert resolved.parent.is_dir()


def test_ensure_trade_log_path_creates_file(tmp_path, monkeypatch):
    """ensure_trade_log_path creates the trade log header if missing."""

    log_path = tmp_path / "trades.csv"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    main_trade_log_path.ensure_trade_log_path()

    assert log_path.exists()
    assert log_path.read_text().splitlines()[0].startswith("symbol,entry_time")


def test_ensure_trade_log_path_unwritable(tmp_path, monkeypatch):
    """ensure_trade_log_path exits when the path is not writable."""

    log_path = tmp_path / "subdir" / "trades.csv"
    parent = log_path.parent
    parent.mkdir()
    parent.chmod(0o400)
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    with pytest.raises(SystemExit) as exc:
        main_trade_log_path.ensure_trade_log_path()
    assert exc.value.code == 1

    parent.chmod(0o700)

