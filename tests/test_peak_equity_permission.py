import ai_trading.core.bot_engine as bot
from pathlib import Path
import pytest

def test_peak_equity_permission(monkeypatch, tmp_path, caplog):
    peak = tmp_path / "peak.txt"
    peak.touch()
    peak.chmod(0)
    equity = tmp_path / "equity.txt"
    equity.write_text("0")
    monkeypatch.setattr(bot, "PEAK_EQUITY_FILE", str(peak))
    monkeypatch.setattr(bot, "EQUITY_FILE", str(equity))
    monkeypatch.setattr(bot, "_PEAK_EQUITY_PERMISSION_LOGGED", False)
    caplog.set_level("WARNING")

    assert bot._current_drawdown() == 0.0
    assert "permission denied" in caplog.text.lower()

    caplog.clear()
    assert bot._current_drawdown() == 0.0
    assert caplog.text == ""
