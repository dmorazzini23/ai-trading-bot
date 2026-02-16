from __future__ import annotations

from pathlib import Path

from ai_trading.analytics.execution_report import write_daily_execution_report
from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
    write_tca_record,
)


def test_execution_report_daily_rollup(tmp_path: Path) -> None:
    tca_path = tmp_path / "tca.jsonl"
    bench = ExecutionBenchmark(arrival_price=100.0, mid_at_arrival=100.0)
    filled = FillSummary(fill_vwap=100.2, total_qty=10.0, fees=0.0, status="filled")
    rec1 = build_tca_record(
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        benchmark=bench,
        fill=filled,
        sleeve="day",
        regime_profile="balanced",
        provider="alpaca",
        order_type="limit",
    )
    rec2 = build_tca_record(
        client_order_id="cid-2",
        symbol="AAPL",
        side="sell",
        benchmark=bench,
        fill=filled,
        sleeve="day",
        regime_profile="balanced",
        provider="alpaca",
        order_type="limit",
    )
    write_tca_record(str(tca_path), rec1)
    write_tca_record(str(tca_path), rec2)

    out_dir = tmp_path / "reports"
    report = write_daily_execution_report(
        tca_path=str(tca_path),
        output_dir=str(out_dir),
        formats=("json", "csv"),
    )
    assert report["records"] == 2
    assert report["groups"]
    assert any(path.suffix == ".json" for path in out_dir.iterdir())
    assert any(path.suffix == ".csv" for path in out_dir.iterdir())
