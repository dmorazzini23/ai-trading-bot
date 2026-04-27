from __future__ import annotations

from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.monitoring import metrics


def test_advanced_metrics_handles_no_losses_without_dividing_by_zero() -> None:
    frame = pd.DataFrame({"return": [0.01, 0.02, 0.0]})

    result = metrics.compute_advanced_metrics(frame)

    assert result["win_rate"] == pytest.approx(200 / 3)
    assert result["profit_factor"] > 1_000_000
    assert result["sortino"] > 0
    assert result["calmar"] > 0


def test_calculate_atr_forward_fills_and_clamps_bad_prices() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, None, 12.0],
            "low": [9.0, 8.0, None],
            "close": [9.5, 9.0, 11.0],
        }
    )

    atr = metrics.calculate_atr(frame, period=2)

    assert atr.tolist() == pytest.approx([1e-8, 2.0, 3.0])
    assert metrics.calculate_atr(pd.DataFrame({"high": [1.0]})).empty


def test_metrics_collector_trims_large_histograms_and_summarizes() -> None:
    collector = metrics.MetricsCollector()

    for value in range(1001):
        collector.observe_latency("submit_latency_ms", float(value), labels={"symbol": "AAPL"})

    key = next(iter(collector.histograms))
    assert len(collector.histograms[key]) == 500
    assert collector.histograms[key][0] == 501.0

    summary = collector.get_metrics_summary()
    stats = summary["histograms_stats"][key]
    assert stats["count"] == 500
    assert stats["min"] == pytest.approx(501.0)
    assert stats["max"] == pytest.approx(1000.0)


def test_performance_monitor_uses_cache_until_forced_refresh() -> None:
    monitor = metrics.PerformanceMonitor()
    monitor.record_trade(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 100.0,
            "latency_ms": 3.0,
            "success": True,
            "return": 0.01,
            "timestamp": datetime(2026, 4, 21, 14, 30, tzinfo=UTC),
        }
    )
    monitor.record_trade(
        {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 10,
            "price": 101.0,
            "latency_ms": 4.0,
            "success": True,
            "return": -0.005,
            "timestamp": datetime(2026, 4, 21, 15, 30, tzinfo=UTC),
        }
    )

    first = monitor.get_performance_metrics(force_refresh=True)
    monitor.record_trade({"symbol": "MSFT", "side": "buy", "success": False, "return": 0.02})
    cached = monitor.get_performance_metrics()
    refreshed = monitor.get_performance_metrics(force_refresh=True)

    assert cached is first
    assert cached["total_trades"] == 2
    assert refreshed["total_trades"] == 3
    assert refreshed["counters"]
