from __future__ import annotations

from ai_trading.telemetry import runtime_state


def test_update_broker_status_clears_stale_last_error_on_recovery() -> None:
    runtime_state.reset_all_states()

    runtime_state.update_broker_status(
        connected=False,
        status="degraded",
        last_error="timeout",
    )
    runtime_state.update_broker_status(
        connected=True,
        status="reachable",
    )

    snapshot = runtime_state.observe_broker_status()
    assert snapshot["connected"] is True
    assert snapshot["status"] == "reachable"
    assert snapshot["last_error"] is None


def test_observe_data_provider_state_returns_deep_copy() -> None:
    runtime_state.reset_all_states()
    runtime_state.update_data_provider_state(
        active="alpaca",
        timeframe="1Min",
        using_backup=True,
    )

    snapshot = runtime_state.observe_data_provider_state()
    snapshot["timeframes"]["1Min"] = False

    latest = runtime_state.observe_data_provider_state()
    assert latest["timeframes"]["1Min"] is True


def test_quote_status_tracks_symbol_snapshots_independently() -> None:
    runtime_state.reset_all_states()

    runtime_state.update_quote_status(
        allowed=True,
        symbol="AAPL",
        status="ready",
        source="broker_nbbo",
        bid=100.0,
        ask=100.02,
        quote_age_ms=250.0,
        quote_timestamp="2026-05-01T14:30:00+00:00",
        spread_bps=2.0,
        max_spread_bps=10.0,
        max_quote_age_ms=1000.0,
        gate_reason="ok",
    )
    runtime_state.update_quote_status(
        allowed=False,
        symbol="MSFT",
        status="blocked",
        bid=410.0,
        ask=410.08,
        quote_age_ms=3500.0,
    )

    latest = runtime_state.observe_quote_status()
    aapl = runtime_state.observe_symbol_quote_status("aapl")
    msft = runtime_state.observe_symbol_quote_status("MSFT")

    assert latest["symbol"] == "MSFT"
    assert aapl["symbol"] == "AAPL"
    assert aapl["bid"] == 100.0
    assert aapl["ask"] == 100.02
    assert aapl["quote_age_ms"] == 250.0
    assert aapl["quote_timestamp"] == "2026-05-01T14:30:00+00:00"
    assert aapl["spread_bps"] == 2.0
    assert aapl["max_spread_bps"] == 10.0
    assert aapl["max_quote_age_ms"] == 1000.0
    assert aapl["gate_reason"] == "ok"
    assert msft["symbol"] == "MSFT"
    assert msft["allowed"] is False


def test_latest_quote_status_does_not_mix_fields_across_symbols() -> None:
    runtime_state.reset_all_states()

    runtime_state.update_quote_status(
        allowed=True,
        symbol="MSFT",
        status="ready",
        bid=414.0,
        ask=421.0,
        last_price=421.0,
        spread_bps=167.66,
    )
    runtime_state.update_quote_status(
        allowed=True,
        symbol="AAPL",
        status="ready",
        bid=276.08,
        ask=276.11,
        quote_age_ms=435.0,
    )

    latest = runtime_state.observe_quote_status()
    msft = runtime_state.observe_symbol_quote_status("MSFT")
    aapl = runtime_state.observe_symbol_quote_status("AAPL")

    assert latest["symbol"] == "AAPL"
    assert latest["bid"] == 276.08
    assert latest["ask"] == 276.11
    assert latest["quote_age_ms"] == 435.0
    assert latest["last_price"] is None
    assert latest["spread_bps"] is None
    assert msft["last_price"] == 421.0
    assert msft["spread_bps"] == 167.66
    assert aapl["last_price"] is None
    assert aapl["spread_bps"] is None
