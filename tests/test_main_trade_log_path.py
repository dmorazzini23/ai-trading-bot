import pytest

import ai_trading.main_trade_log_path as main_trade_log_path
import ai_trading.core.bot_engine as bot_engine


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

