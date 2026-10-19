from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_trading.telemetry import trade_updates_stream as tus
from ai_trading.tools import trade_updates_parity as parity

SESSION = "2026-10-16"


def _stream(order_id: str, event: str, filled_qty: float, price: float, ts: str) -> dict[str, Any]:
    return {
        "received_at": ts,
        "event": event,
        "event_timestamp": ts,
        "order_id": order_id,
        "symbol": "AAPL",
        "filled_qty": filled_qty,
        "filled_avg_price": price,
    }


def _polled(order_id: str, qty: float, price: float, ts: str, fill_id: str) -> dict[str, Any]:
    return {
        "event": "fill_recorded",
        "order_id": order_id,
        "symbol": "AAPL",
        "fill_qty": qty,
        "fill_price": price,
        "fill_id": fill_id,
        "ts": ts,
    }


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    tus.reset_trade_updates_stream_state()
    parity._REFRESH_CACHE.clear()
    yield
    tus.reset_trade_updates_stream_state()
    parity._REFRESH_CACHE.clear()


def test_matched_with_partial_fills_cumulative_qty() -> None:
    stream = [
        _stream("o-1", "new", 0, 0, "2026-10-16T14:30:00Z"),
        _stream("o-1", "partial_fill", 4, 100.0, "2026-10-16T14:30:01Z"),
        _stream("o-1", "fill", 10, 100.01, "2026-10-16T14:30:02Z"),
    ]
    polled = [
        _polled("o-1", 4, 100.0, "2026-10-16T14:30:05Z", "f1"),
        _polled("o-1", 6, 100.0166667, "2026-10-16T14:30:07Z", "f2"),
        _polled("o-1", 6, 100.0166667, "2026-10-16T14:30:07Z", "f2"),  # duplicate row
    ]
    report = parity.compute_trade_updates_parity(stream, polled, session=SESSION)
    assert report["stream_fill_count"] == 2
    assert report["stream_fill_orders"] == 1
    assert report["polled_fill_count"] == 2
    assert report["matched"] == 1
    assert report["stream_only"] == report["polled_only"] == report["qty_mismatch"] == 0
    assert report["price_delta_bps_p95"] is not None and report["price_delta_bps_p95"] < 0.01
    assert report["latency_seconds_median"] == 5.0
    assert report["clean"] is True
    assert report["mismatches"] == []


def test_mismatched_sets_reported() -> None:
    stream = [
        _stream("o-1", "fill", 10, 100.0, "2026-10-16T14:30:00Z"),
        _stream("o-2", "fill", 5, 50.0, "2026-10-16T14:31:00Z"),
        _stream("o-3", "partial_fill", 3, 20.0, "2026-10-16T14:32:00Z"),
        _stream("o-5", "fill", 1, 200.0, "2026-10-16T14:33:00Z"),
    ]
    polled = [
        _polled("o-1", 10, 100.0, "2026-10-16T14:30:02Z", "a"),
        _polled("o-3", 5, 20.0, "2026-10-16T14:32:02Z", "b"),
        _polled("o-4", 7, 30.0, "2026-10-16T14:34:02Z", "c"),
        _polled("o-5", 1, 200.1, "2026-10-16T14:33:01Z", "d"),
    ]
    report = parity.compute_trade_updates_parity(stream, polled, session=SESSION)
    assert report["matched"] == 2
    assert report["stream_only"] == 1
    assert report["polled_only"] == 1
    assert report["qty_mismatch"] == 1
    kinds = {item["order_id"]: item["kind"] for item in report["mismatches"]}
    assert kinds == {"o-2": "stream_only", "o-3": "qty_mismatch", "o-4": "polled_only"}
    # o-1 delta 0 bps, o-5 delta ~5 bps.
    assert report["price_delta_bps_median"] == pytest.approx(2.4988, abs=1e-3)
    assert report["price_delta_bps_p95"] == pytest.approx(4.7476, abs=1e-3)
    assert report["latency_seconds_median"] == 1.5
    assert report["clean"] is False


def test_empty_inputs_zero_path() -> None:
    report = parity.compute_trade_updates_parity([], [], session=SESSION)
    assert report["matched"] == 0
    assert report["price_delta_bps_median"] is None
    assert report["latency_seconds_median"] is None
    assert report["clean"] is False


def test_session_filter_uses_new_york_date(tmp_path: Path) -> None:
    stream_path = tmp_path / "stream.jsonl"
    rows = [
        _stream("o-1", "fill", 1, 10.0, "2026-10-16T23:59:00Z"),  # 19:59 ET on the 16th
        _stream("o-2", "fill", 1, 10.0, "2026-10-17T03:00:00Z"),  # 23:00 ET on the 16th
        _stream("o-3", "fill", 1, 10.0, "2026-10-17T05:00:00Z"),  # 01:00 ET on the 17th
    ]
    stream_path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not json\n", encoding="utf-8")
    loaded = parity.load_stream_records(stream_path, SESSION)
    assert [row["order_id"] for row in loaded] == ["o-1", "o-2"]


def test_build_session_report_missing_artifact(tmp_path: Path) -> None:
    report = parity.build_session_report(
        SESSION, stream_path=tmp_path / "missing.jsonl", polled_path=tmp_path / "missing_fills.jsonl"
    )
    assert report["available"] is False
    assert report["reason"] == "stream_artifact_missing"
    assert report["stream_fill_count"] == 0
    assert report["polled_fill_count"] == 0


def test_refresh_parity_latest_writes_artifact_and_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream_path = tmp_path / "stream.jsonl"
    fills_path = tmp_path / "fill_events.jsonl"
    latest_path = tmp_path / "latest.json"
    stream_path.write_text(json.dumps(_stream("o-1", "fill", 2, 10.0, "2026-10-16T15:00:00Z")) + "\n", encoding="utf-8")
    fills_path.write_text(json.dumps(_polled("o-1", 2, 10.0, "2026-10-16T15:00:03Z", "x")) + "\n", encoding="utf-8")
    monkeypatch.setenv(tus.STREAM_ARTIFACT_ENV, str(stream_path))
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", str(fills_path))
    monkeypatch.setenv(tus.PARITY_LATEST_ENV, str(latest_path))
    fills_before = fills_path.read_bytes()

    report = parity.refresh_parity_latest(SESSION)

    assert report["matched"] == 1 and report["available"] is True
    assert json.loads(latest_path.read_text(encoding="utf-8"))["matched"] == 1
    assert tus.observe_trade_updates_stream_state()["parity"]["matched"] == 1
    assert fills_path.read_bytes() == fills_before
    # Unchanged inputs reuse the cached summary.
    latest_path.unlink()
    assert parity.refresh_parity_latest(SESSION)["matched"] == 1
    assert not latest_path.exists()


def test_cli_outputs_json_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "stream.jsonl"
    empty.write_text("", encoding="utf-8")
    output = tmp_path / "report.json"
    rc = parity.main(
        [
            "--session", SESSION,
            "--stream-path", str(empty),
            "--polled-path", str(tmp_path / "none.jsonl"),
            "--output", str(output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["session"] == SESSION
    assert payload["available"] is False
    assert payload["reason"] == "no_stream_events_for_session"
    assert "trade_updates_parity session=2026-10-16" in captured.err
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cli_rejects_bad_session() -> None:
    with pytest.raises(SystemExit):
        parity.main(["--session", "16-10-2026"])
