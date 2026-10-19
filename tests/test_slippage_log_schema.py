"""The slippage CSV has one schema: every header writer and every row writer agree."""
from __future__ import annotations

import csv
from pathlib import Path

from ai_trading.execution import slippage_log
from ai_trading.slippage.recorder import SLIPPAGE_LOG_COLUMNS, ensure_slippage_log_header, log_slippage
from ai_trading.tca.rollups import load_slippage_records


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_record_fill_rows_match_the_shared_header(tmp_path, monkeypatch):
    path = tmp_path / "slippage.csv"
    monkeypatch.setattr(slippage_log, "SLIPPAGE_LOG_PATH", path)
    slippage_log._EWMA_BPS.clear()

    slippage_log.record_fill(symbol="AAPL", side="buy", qty=3, expected_price=100.0, fill_price=100.25)
    slippage_log.record_fill(symbol="MSFT", side="sell", qty=2, expected_price=200.0, fill_price=199.0)

    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(SLIPPAGE_LOG_COLUMNS)
    rows = _rows(path)
    assert [len(row) for row in rows] == [len(SLIPPAGE_LOG_COLUMNS)] * 2
    assert rows[0]["side"] == "buy" and rows[0]["quantity"] == "3"
    assert float(rows[0]["expected_price"]) == 100.0 and float(rows[0]["actual_price"]) == 100.25
    assert float(rows[0]["slippage_bps"]) == 25.0
    assert float(rows[1]["slippage_bps"]) == 50.0


def test_recorder_and_record_fill_share_one_file_shape(tmp_path, monkeypatch):
    path = tmp_path / "slippage.csv"
    monkeypatch.setattr(slippage_log, "SLIPPAGE_LOG_PATH", path)
    slippage_log._EWMA_BPS.clear()

    log_slippage("AAPL", 100.0, 100.5, path, side="buy", quantity=1)
    slippage_log.record_fill(symbol="AAPL", side="sell", qty=1, expected_price=100.0, fill_price=99.5)

    rows = _rows(path)
    assert [float(row["slippage_bps"]) for row in rows] == [50.0, 50.0]
    loaded = load_slippage_records(path)
    assert [(row["side"], row["is_bps"]) for row in loaded] == [("buy", 50.0), ("sell", 50.0)]


def test_bot_engine_header_matches_the_shared_columns(tmp_path, monkeypatch):
    from ai_trading.core import bot_engine

    path = tmp_path / "logs" / "slippage.csv"
    monkeypatch.setattr(bot_engine, "SLIPPAGE_LOG_FILE", str(path))

    bot_engine.ensure_slippage_log_file()

    assert path.read_text().splitlines() == [",".join(SLIPPAGE_LOG_COLUMNS)]
    bot_engine.ensure_slippage_log_file()  # idempotent: never rewrites an existing file
    assert path.read_text().splitlines() == [",".join(SLIPPAGE_LOG_COLUMNS)]


def test_vwap_rows_match_the_shared_columns(tmp_path):
    from ai_trading.core.execution_flow import vwap_slippage_row

    path = tmp_path / "slippage.csv"
    ensure_slippage_log_header(path)
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow(vwap_slippage_row("2026-10-19T14:30:00+00:00", "SPY", "buy", "4", 500.0, 500.1))
        csv.writer(handle).writerow(vwap_slippage_row("2026-10-19T14:31:00+00:00", "SPY", "sell", None, 500.0, 499.9))

    rows = _rows(path)
    assert [row["quantity"] for row in rows] == ["4", "0"]
    assert [float(row["slippage_bps"]) for row in rows] == [2.0, 2.0]
    assert all(len(row) == len(SLIPPAGE_LOG_COLUMNS) for row in rows)
    assert [row["is_bps"] for row in load_slippage_records(path)] == [2.0, 2.0]


def test_legacy_five_column_rows_still_load_under_the_new_header(tmp_path):
    path = tmp_path / "slippage.csv"
    ensure_slippage_log_header(path)
    path.open("a").write("2026-10-19T14:30:00+00:00,MSFT,200.0,199.0,-100.0\n")

    loaded = load_slippage_records(path)
    assert [(row["symbol"], row["side"], row["is_bps"]) for row in loaded] == [("MSFT", "unknown", 50.0)]
