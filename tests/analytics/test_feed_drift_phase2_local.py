from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.analytics import feed_drift as fd


def test_signal_side_agreement_and_drift_metrics() -> None:
    assert fd.normalize_signal_side("entry_long") == "buy"
    assert fd.normalize_signal_side("sell_short") == "sell"
    assert fd.normalize_signal_side("hold") == "flat"
    assert fd.normalize_signal_side("surprise") is None

    gate = fd.derive_signal_agreement(
        outcome="blocked_degraded_feed",
        side="short",
        signal_strength="0.7",
        signal_confidence="0.8",
        price_drift_bps="30",
        drift_disagreement_bps=25,
    )
    trigger = fd.derive_signal_agreement(outcome="", signal_side="buy", price_drift_bps=5, drift_disagreement_bps="bad")

    assert gate["decision_class"] == "gate"
    assert gate["signal_side"] == "sell"
    assert gate["signal_disagreement"] is True
    assert gate["signal_disagreement_reason"] == "price_drift_threshold"
    assert trigger["decision_class"] == "trigger"
    assert trigger["signal_agreement"] is True
    assert trigger["drift_threshold_bps"] == 25.0

    metrics = fd.compute_drift_metrics(
        execution_price=101.0,
        reference_price=100.0,
        execution_bid=100.9,
        execution_ask=101.1,
        reference_bid=99.9,
        reference_ask=100.1,
        execution_volume=50.0,
        reference_volume=100.0,
    )

    assert metrics["price_drift_bps"] == pytest.approx(100.0)
    assert metrics["spread_ratio"] == pytest.approx((0.2 / 101.0) / (0.2 / 100.0))
    assert metrics["volume_coverage_ratio"] == 0.5
    assert fd.compute_drift_metrics(execution_price=None, reference_price=100.0)["price_drift_bps"] is None


def test_reference_snapshot_and_minute_bar_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get(path: str, *, params=None):
        calls.append((path, dict(params or {})))
        if "trades" in path:
            return {"trade": {"p": "101.5"}}
        if "quotes" in path:
            return {"quote": {"bp": "101.0", "ap": "102.0"}}
        return None

    monkeypatch.setattr(fd, "get_reference_feed", lambda feed=None: feed or "delayed_sip")
    monkeypatch.setattr(fd, "_alpaca_data_get", fake_get)

    snapshot = fd.fetch_reference_snapshot("AAPL")

    assert snapshot["price"] == 101.5
    assert snapshot["mid"] == 101.5
    assert calls[0][1]["feed"] == "delayed_sip"

    def fake_bar_get(_path: str, *, params=None):
        return {
            "bars": {
                "AAPL": [
                    {"t": "2026-04-27T15:59:00Z", "c": 99.0, "v": 10},
                    {"t": "2026-04-27T16:00:30Z", "close": 100.0, "volume": 20},
                    {"t": "bad", "c": 101.0},
                ]
            }
        }

    monkeypatch.setattr(fd, "_alpaca_data_get", fake_bar_get)
    bar = fd.fetch_reference_minute_bar_snapshot(
        "AAPL",
        decision_ts=datetime(2026, 4, 27, 16, tzinfo=UTC),
        feed="sip",
    )

    assert bar["price"] == 100.0
    assert bar["volume"] == 20.0
    assert bar["feed"] == "sip"


def test_extractors_and_data_get_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fd, "get_alpaca_data_v2_base", lambda: "https://data.example/v2/")
    monkeypatch.setattr(fd, "alpaca_get", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert fd._alpaca_data_get("stocks/AAPL/trades/latest") is None  # noqa: SLF001
    assert fd._extract_trade_price({"trade": {"price": "bad"}, "last": "103.25"}) == 103.25  # noqa: SLF001
    assert fd._extract_trade_price({"trade": {"p": "-1"}}) is None  # noqa: SLF001
    assert fd._extract_bid_ask({"bid_price": "1.1", "ask_price": "1.3"}) == (1.1, 1.3)  # noqa: SLF001
    assert fd._extract_bid_ask(None) == (None, None)  # noqa: SLF001
    assert fd._safe_median([1.0, float("nan"), 3.0]) == 2.0  # noqa: SLF001
    assert fd._safe_median([float("inf")]) is None  # noqa: SLF001


def test_symbol_reliability_scores_group_gate_and_trigger_disagreements() -> None:
    rows = [
        {
            "symbol": "aapl",
            "outcome": "triggered",
            "metrics": {"price_drift_bps": 20, "spread_ratio": 1.2, "volume_coverage_ratio": 0.9},
            "signal_metrics": {"decision_class": "trigger", "signal_disagreement": True},
        },
        {
            "symbol": "AAPL",
            "outcome": "skip_gate",
            "metrics": {"price_drift_bps": -30, "spread_ratio": 0.8, "volume_coverage_ratio": 1.1},
            "signal_disagreement": "false",
        },
        {
            "symbol": "AAPL",
            "outcome": "stale_quote",
            "metrics": {"price_drift_bps": 10, "spread_ratio": "bad", "volume_coverage_ratio": 1.0},
            "signal_metrics": {"decision_class": "gate", "signal_disagreement": "yes"},
        },
        {"symbol": "", "metrics": {"price_drift_bps": 999}},
        {"symbol": "MSFT", "metrics": {"price_drift_bps": 1}},
    ]

    scores = fd.build_symbol_reliability_scores(rows, min_samples=3, drift_disagreement_bps=25)

    assert set(scores) == {"AAPL"}
    assert scores["AAPL"]["sample_count"] == 3
    assert scores["AAPL"]["median_abs_price_drift_bps"] == 20.0
    assert scores["AAPL"]["trigger_disagreement_rate"] == 1.0
    assert scores["AAPL"]["gate_disagreement_rate"] == 0.5
    assert 0.0 <= scores["AAPL"]["reliability_score"] <= 1.0
