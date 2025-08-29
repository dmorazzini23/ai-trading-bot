from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.data import universe


def test_env_override_path_preferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Env var should override packaged CSV."""
    # AI-AGENT-REF: verify env override path
    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(csv_path))
    try:
        path = universe.locate_tickers_csv()
        assert path == str(csv_path.resolve())
        symbols = universe.load_universe()
        assert symbols == ["AAPL", "MSFT"]
    finally:
        monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)


def test_package_fallback_loads_packaged_csv():
    """Fallback uses packaged CSV when env not set."""  # AI-AGENT-REF: ensure fallback works
    path = universe.locate_tickers_csv()
    assert path is not None and path.endswith("ai_trading/data/tickers.csv")
    symbols = universe.load_universe()
    assert isinstance(symbols, list) and len(symbols) > 0


def test_missing_package_returns_empty_and_logs(monkeypatch: pytest.MonkeyPatch):
    """Missing package should log and return []"""  # AI-AGENT-REF: test missing pkg case

    def boom(_name: str):
        raise ModuleNotFoundError("ai_trading.data not importable")

    monkeypatch.setattr(universe, "pkg_files", boom, raising=True)
    monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)

    called: list[tuple[str, dict]] = []

    def fake_error(msg, *, extra):
        called.append((msg, extra))

    monkeypatch.setattr(universe.logger, "error", fake_error)
    syms = universe.load_universe()
    assert syms == []
    assert called and called[0][0] == "TICKERS_FILE_MISSING"


def test_malformed_empty_csv_logs_and_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Malformed/empty CSV should log read failure."""  # AI-AGENT-REF: test read guard

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(empty_csv))

    called: list[tuple[str, dict]] = []

    def fake_error(msg, *, extra):
        called.append((msg, extra))

    monkeypatch.setattr(universe.logger, "error", fake_error)

    syms = universe.load_universe()
    assert syms == []
    assert called and called[0][0] == "TICKERS_FILE_READ_FAILED"


def test_brk_dot_b_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """BRK.B should be normalized to BRK-B for Yahoo Finance."""

    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("symbol\nBRK.B\n", encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(csv_path))
    try:
        symbols = universe.load_universe()
    finally:
        monkeypatch.delenv("AI_TRADING_TICKERS_CSV", raising=False)
    assert symbols == ["BRK-B"]

