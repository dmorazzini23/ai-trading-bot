import sys

from ai_trading.__main__ import main


def test_main_returns_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ai_trading", "--dry-run"])
    assert main() == 0
