from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, safe_validate_pretrade
from ai_trading.runtime.paper_sampling import (
    evaluate_paper_sampling_order,
    paper_sampling_deficit_snapshot,
    release_paper_sampling_order,
    reserve_paper_sampling_order,
)


def _cfg(**updates):
    values = {
        "paper_sampling_enabled": True,
        "paper_sampling_allowed_symbols": ("AAPL", "AMZN", "MSFT"),
        "paper_sampling_max_trades_per_day": 1,
        "paper_sampling_stratified_fairness_enabled": False,
        "paper_sampling_symbol_fairness_max_lead": 1,
        "paper_sampling_max_trades_per_symbol_per_day": 4,
        "paper_sampling_max_trades_per_side_per_day": 6,
        "paper_sampling_max_opening_trades_per_day": 3,
        "paper_sampling_max_midday_trades_per_day": 4,
        "paper_sampling_max_closing_trades_per_day": 3,
        "paper_sampling_reserved_opening_trades_per_day": 0,
        "paper_sampling_reserved_midday_trades_per_day": 0,
        "paper_sampling_reserved_closing_trades_per_day": 0,
        "paper_sampling_max_notional_per_order": 750.0,
        "paper_sampling_passive_only": True,
        "paper_sampling_relax_edge_gates_enabled": False,
        "execution_mode": "paper",
        "paper": True,
        "alpaca_base_url": "https://paper-api.alpaca.markets",
        "launch_profile": "paper_trade",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_config_rejects_paper_sampling_outside_paper_mode() -> None:
    with pytest.raises(ValueError, match="PAPER_SAMPLING_ENABLED requires EXECUTION_MODE=paper"):
        TradingConfig.from_env(
            {
                "APP_ENV": "prod",
                "EXECUTION_MODE": "live",
                "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
                "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
                "MAX_DRAWDOWN_THRESHOLD": "0.2",
            }
        )


def test_config_rejects_paper_sampling_for_live_canary_profile() -> None:
    with pytest.raises(ValueError, match="non-live launch profile"):
        TradingConfig.from_env(
            {
                "APP_ENV": "test",
                "EXECUTION_MODE": "paper",
                "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
                "AI_TRADING_LAUNCH_PROFILE": "live_canary",
                "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
                "MAX_DRAWDOWN_THRESHOLD": "0.2",
            }
        )


def test_config_exposes_conservative_paper_sampling_knobs() -> None:
    cfg = TradingConfig.from_env(
        {
            "APP_ENV": "test",
            "EXECUTION_MODE": "paper",
            "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
            "AI_TRADING_LAUNCH_PROFILE": "paper_trade",
            "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
            "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS": "AAPL,AMZN,MSFT",
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_ONLY": "1",
            "AI_TRADING_PAPER_SAMPLING_RELAX_EDGE_GATES_ENABLED": "0",
            "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER": "750",
            "AI_TRADING_PAPER_SAMPLING_MIN_EXPECTED_NET_EDGE_BPS": "4.0",
            "AI_TRADING_PAPER_SAMPLING_EDGE_MIN_EXPECTED_BPS": "4.0",
            "AI_TRADING_PAPER_SAMPLING_EDGE_COST_MIN_RATIO": "1.10",
            "AI_TRADING_PAPER_SAMPLING_MAX_MANUAL_EDGE_PENALTY_BPS": "5.0",
            "MAX_DRAWDOWN_THRESHOLD": "0.2",
        }
    )

    assert cfg.paper_sampling_allowed_symbols == ("AAPL", "AMZN", "MSFT")
    assert cfg.paper_sampling_stratified_fairness_enabled is True
    assert cfg.paper_sampling_symbol_fairness_max_lead == 1
    assert cfg.paper_sampling_reserved_opening_trades_per_day == 1
    assert cfg.paper_sampling_reserved_midday_trades_per_day == 1
    assert cfg.paper_sampling_reserved_closing_trades_per_day == 1
    assert cfg.paper_sampling_max_notional_per_order == 750.0
    assert cfg.paper_sampling_passive_only is True
    assert cfg.paper_sampling_relax_edge_gates_enabled is False
    assert cfg.paper_sampling_min_expected_net_edge_bps == 4.0
    assert cfg.paper_sampling_edge_min_expected_bps == 4.0
    assert cfg.paper_sampling_edge_cost_min_ratio == 1.10
    assert cfg.paper_sampling_max_manual_edge_penalty_bps == 5.0


@pytest.mark.parametrize(
    ("env_key", "env_value", "message"),
    (
        (
            "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS",
            "AAPL,NVDA",
            "must be limited to AAPL,AMZN,MSFT",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER",
            "751",
            "must be <= 750",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_RELAX_EDGE_GATES_ENABLED",
            "1",
            "must remain disabled",
        ),
        (
            "AI_TRADING_PAPER_SAMPLING_PASSIVE_ONLY",
            "0",
            "must remain enabled",
        ),
    ),
)
def test_config_rejects_unsafe_paper_sampling_policy(
    env_key: str,
    env_value: str,
    message: str,
) -> None:
    env = {
        "APP_ENV": "test",
        "EXECUTION_MODE": "paper",
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        "AI_TRADING_LAUNCH_PROFILE": "paper_trade",
        "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
        "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS": "AAPL,AMZN,MSFT",
        "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER": "750",
        "AI_TRADING_PAPER_SAMPLING_PASSIVE_ONLY": "1",
        "AI_TRADING_PAPER_SAMPLING_RELAX_EDGE_GATES_ENABLED": "0",
        "MAX_DRAWDOWN_THRESHOLD": "0.2",
    }
    env[env_key] = env_value

    with pytest.raises(ValueError, match=message):
        TradingConfig.from_env(env)


def test_config_rejects_session_reservations_above_daily_capacity() -> None:
    with pytest.raises(
        ValueError,
        match="reserved session slots must not exceed",
    ):
        TradingConfig.from_env(
            {
                "APP_ENV": "test",
                "EXECUTION_MODE": "paper",
                "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
                "AI_TRADING_LAUNCH_PROFILE": "paper_trade",
                "AI_TRADING_PAPER_SAMPLING_ENABLED": "1",
                "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS": "AAPL,AMZN,MSFT",
                "AI_TRADING_PAPER_SAMPLING_MAX_TRADES_PER_DAY": "2",
                "AI_TRADING_PAPER_SAMPLING_RESERVED_OPENING_TRADES_PER_DAY": "1",
                "AI_TRADING_PAPER_SAMPLING_RESERVED_MIDDAY_TRADES_PER_DAY": "1",
                "AI_TRADING_PAPER_SAMPLING_RESERVED_CLOSING_TRADES_PER_DAY": "1",
                "MAX_DRAWDOWN_THRESHOLD": "0.2",
            }
        )


def test_paper_sampling_higher_cap_allows_one_share_msft_sample() -> None:
    cfg = _cfg(
        paper_sampling_allowed_symbols=("AAPL", "AMZN", "MSFT"),
        paper_sampling_max_notional_per_order=750.0,
    )

    decision = evaluate_paper_sampling_order(
        cfg,
        symbol="MSFT",
        side="buy",
        qty=1,
        price=525.0,
    )

    assert decision.allowed is True
    assert decision.qty == 1
    assert decision.reason == "OK"
    assert decision.details["max_notional_per_order"] == 750.0


def test_paper_sampling_symbol_short_size_and_daily_caps(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg()

    short_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell_short",
        qty=1,
        price=100.0,
    )
    assert short_decision.allowed is False
    assert short_decision.reason == "PAPER_SAMPLING_SHORT_BLOCK"

    symbol_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="NVDA",
        side="buy",
        qty=1,
        price=100.0,
    )
    assert symbol_decision.allowed is False
    assert symbol_decision.reason == "PAPER_SAMPLING_SYMBOL_BLOCK"

    size_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=10,
        price=800.0,
    )
    assert size_decision.allowed is False
    assert size_decision.reason == "PAPER_SAMPLING_MAX_NOTIONAL_BLOCK"
    assert size_decision.qty == 0

    capped_decision = evaluate_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=10,
        price=100.0,
    )
    assert capped_decision.allowed is True
    assert capped_decision.qty == 7

    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )
    second = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "PAPER_SAMPLING_DAILY_CAP_BLOCK"


def test_paper_sampling_reduce_orders_do_not_consume_daily_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg()
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)

    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )
    closing_sell = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell",
        qty=1,
        price=100.0,
        now=now,
        consumes_daily_slot=False,
    )
    next_entry = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )

    assert first.allowed is True
    assert closing_sell.allowed is True
    assert closing_sell.details["consumes_daily_slot"] is False
    assert next_entry.allowed is False
    assert next_entry.reason == "PAPER_SAMPLING_DAILY_CAP_BLOCK"
    assert next_entry.details["count"] == 1


def test_paper_sampling_release_restores_daily_symbol_side_and_session_quota(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(paper_sampling_max_trades_per_day=1)
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)

    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )
    release_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        now=now,
    )
    second = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=now,
    )

    assert first.allowed is True
    assert second.allowed is True
    state_path = tmp_path / "runtime" / "paper_sampling_state_latest.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["by_symbol"] == {"AMZN": 1}
    assert payload["by_side"] == {"buy": 1}
    assert payload["by_session"] == {"midday": 1}


def test_paper_sampling_symbol_side_and_session_quotas(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        paper_sampling_max_trades_per_day=8,
        paper_sampling_max_trades_per_symbol_per_day=2,
        paper_sampling_max_trades_per_side_per_day=3,
        paper_sampling_max_opening_trades_per_day=2,
        paper_sampling_max_midday_trades_per_day=4,
        paper_sampling_max_closing_trades_per_day=3,
    )
    opening = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    midday = datetime(2026, 5, 8, 17, 0, tzinfo=UTC)

    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )
    second = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )
    opening_block = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )

    assert first.allowed is True
    assert first.details["session_bucket"] == "opening"
    assert second.allowed is True
    assert opening_block.allowed is False
    assert opening_block.reason == "PAPER_SAMPLING_SESSION_DAILY_QUOTA_BLOCK"
    assert opening_block.details["quota_key"] == "session:opening"

    aapl_midday = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=midday,
    )
    symbol_block = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell",
        qty=1,
        price=100.0,
        now=midday,
    )

    assert aapl_midday.allowed is True
    assert symbol_block.allowed is False
    assert symbol_block.reason == "PAPER_SAMPLING_SYMBOL_DAILY_QUOTA_BLOCK"
    assert symbol_block.details["quota_key"] == "symbol:AAPL"

    side_block = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=midday,
    )
    assert side_block.allowed is False
    assert side_block.reason == "PAPER_SAMPLING_SIDE_DAILY_QUOTA_BLOCK"
    assert side_block.details["quota_key"] == "side:buy"


def test_paper_sampling_stratified_fairness_prevents_four_msft_entries(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        paper_sampling_max_trades_per_day=12,
        paper_sampling_max_trades_per_symbol_per_day=12,
        paper_sampling_max_trades_per_side_per_day=12,
        paper_sampling_max_midday_trades_per_day=12,
        paper_sampling_stratified_fairness_enabled=True,
        paper_sampling_symbol_fairness_max_lead=1,
    )
    midday = datetime(2026, 7, 20, 17, 0, tzinfo=UTC)

    attempts = [
        reserve_paper_sampling_order(
            cfg,
            symbol="MSFT",
            side="buy",
            qty=1,
            price=500.0,
            now=midday,
        )
        for _ in range(4)
    ]

    assert [decision.allowed for decision in attempts] == [True, False, False, False]
    assert {
        decision.reason for decision in attempts[1:]
    } == {"PAPER_SAMPLING_SYMBOL_FAIRNESS_BLOCK"}
    state_path = tmp_path / "runtime" / "paper_sampling_state_latest.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["by_symbol"] == {"MSFT": 1}
    assert payload["symbol_targets"] == {"AAPL": 4, "AMZN": 4, "MSFT": 4}
    assert payload["symbol_deficits"] == {"AAPL": 4, "AMZN": 4, "MSFT": 3}

    for symbol in ("AAPL", "AMZN", "MSFT"):
        decision = reserve_paper_sampling_order(
            cfg,
            symbol=symbol,
            side="buy",
            qty=1,
            price=100.0,
            now=midday,
        )
        assert decision.allowed is True

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["by_symbol"] == {"AAPL": 1, "AMZN": 1, "MSFT": 2}


def test_paper_sampling_scarce_daily_capacity_rotates_deterministically(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        paper_sampling_max_trades_per_day=1,
        paper_sampling_max_trades_per_symbol_per_day=1,
        paper_sampling_max_trades_per_side_per_day=1,
        paper_sampling_max_midday_trades_per_day=1,
        paper_sampling_stratified_fairness_enabled=True,
    )

    def _allowed_symbol(now: datetime) -> str:
        for symbol in ("AAPL", "AMZN", "MSFT"):
            decision = reserve_paper_sampling_order(
                cfg,
                symbol=symbol,
                side="buy",
                qty=1,
                price=100.0,
                now=now,
            )
            if decision.allowed:
                return symbol
            assert decision.reason == "PAPER_SAMPLING_SYMBOL_RESERVATION_BLOCK"
        raise AssertionError("one governed symbol must own the rotated daily slot")

    first = _allowed_symbol(datetime(2026, 7, 20, 17, 0, tzinfo=UTC))
    second = _allowed_symbol(datetime(2026, 7, 21, 17, 0, tzinfo=UTC))

    assert first != second


def test_paper_sampling_reserves_capacity_for_future_sessions(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        paper_sampling_max_trades_per_day=3,
        paper_sampling_max_trades_per_symbol_per_day=3,
        paper_sampling_max_trades_per_side_per_day=3,
        paper_sampling_max_opening_trades_per_day=3,
        paper_sampling_reserved_midday_trades_per_day=1,
        paper_sampling_reserved_closing_trades_per_day=1,
        paper_sampling_stratified_fairness_enabled=True,
    )
    opening = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)

    first = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )
    second = reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "PAPER_SAMPLING_FUTURE_SESSION_RESERVATION_BLOCK"
    assert second.details["future_reserved_slots"] == 2


def test_paper_sampling_release_uses_original_reservation_session(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(paper_sampling_max_trades_per_day=3)
    opening = datetime(2026, 7, 20, 14, 59, tzinfo=UTC)
    midday = datetime(2026, 7, 20, 15, 1, tzinfo=UTC)

    reserved = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=opening,
    )
    release_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        now=midday,
        reservation_token=str(reserved.details["reservation_token"]),
    )

    state_path = tmp_path / "runtime" / "paper_sampling_state_latest.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["count"] == 0
    assert payload["by_session"] == {}
    assert payload["observed_by_session"] == {}
    assert payload["reservations"] == []


def test_paper_sampling_tracks_exit_role_without_consuming_or_blocking(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(paper_sampling_max_trades_per_day=1)
    midday = datetime(2026, 7, 20, 17, 0, tzinfo=UTC)

    entry = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
        now=midday,
    )
    exit_order = reserve_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell",
        qty=1,
        price=101.0,
        now=midday,
        consumes_daily_slot=False,
    )

    assert entry.allowed is True
    assert exit_order.allowed is True
    state_path = tmp_path / "runtime" / "paper_sampling_state_latest.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["observed_count"] == 2
    assert payload["observed_by_role"] == {"entry": 1, "exit": 1}
    assert payload["observed_by_side_role"] == {"buy:entry": 1, "sell:exit": 1}
    assert payload["observed_by_stratum"] == {
        "AAPL:buy:entry:midday": 1,
        "AAPL:sell:exit:midday": 1,
    }

    release_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="sell",
        now=midday,
        consumes_daily_slot=False,
        reservation_token=str(exit_order.details["reservation_token"]),
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["observed_count"] == 1
    assert payload["observed_by_role"] == {"entry": 1}


def test_paper_sampling_deficit_snapshot_is_governed_read_only_and_deterministic(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        paper_sampling_allowed_symbols=("AAPL", "AMZN", "GOOGL", "MSFT"),
        paper_sampling_max_trades_per_day=12,
        paper_sampling_max_trades_per_symbol_per_day=4,
        paper_sampling_stratified_fairness_enabled=True,
    )
    state_path = tmp_path / "runtime" / "paper_sampling_state_latest.json"
    state_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": "2.0.0",
        "artifact_type": "paper_sampling_state",
        "date": "2026-07-20",
        "count": 3,
        "by_symbol": {"AAPL": 1, "AMZN": 1, "GOOGL": 9, "MSFT": 1},
        "observed_by_symbol_session": {
            "AAPL:opening": 1,
            "GOOGL:opening": 9,
        },
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    before = state_path.read_text(encoding="utf-8")
    opening = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)
    midday = datetime(2026, 7, 20, 17, 0, tzinfo=UTC)

    first = paper_sampling_deficit_snapshot(cfg, now=opening)
    repeated = paper_sampling_deficit_snapshot(cfg, now=opening)
    later_session = paper_sampling_deficit_snapshot(cfg, now=midday)

    assert first == repeated
    assert first["configured_symbols"] == ["MSFT", "AAPL", "AMZN"]
    assert set(first["counts"]) == {"AAPL", "AMZN", "MSFT"}
    assert first["counts"] == {"AAPL": 1, "AMZN": 1, "MSFT": 1}
    assert first["deficits"] == {"AAPL": 3, "AMZN": 3, "MSFT": 3}
    assert first["priority_reason"] == "session_deficit"
    assert first["priority_symbols"] == ["MSFT", "AMZN"]
    assert later_session["priority_reason"] == "balanced"
    assert later_session["priority_symbols"] == []
    assert state_path.read_text(encoding="utf-8") == before


def test_paper_sampling_does_not_bypass_oms_order_size_block() -> None:
    cfg = _cfg(max_order_dollars=50.0)
    sampling = evaluate_paper_sampling_order(
        cfg,
        symbol="AAPL",
        side="buy",
        qty=1,
        price=100.0,
    )
    assert sampling.allowed is True

    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        qty=sampling.qty,
        notional=100.0,
        limit_price=100.0,
        bar_ts=datetime(2026, 5, 8, 15, 0, tzinfo=UTC),
        client_order_id="paper-sampling-test",
        last_price=100.0,
        mid=100.0,
    )
    allowed, reason, _details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=None,
        rate_limiter=SlidingWindowRateLimiter(),
    )

    assert allowed is False
    assert reason == "ORDER_SIZE_BLOCK"
