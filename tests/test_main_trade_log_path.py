from pathlib import Path


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





