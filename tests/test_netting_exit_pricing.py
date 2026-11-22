"""Urgent netting exits (protective exit or full close) are priced to fill.

Every relaxation here is paired with the partial trim, the add and the entry,
which must keep the bar-close limit and meet the same blocks as before.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core import bot_engine
from ai_trading.core.netting_exit_pricing import (
    PRICE_BASIS_BAR_CLOSE,
    PRICE_BASIS_QUOTE,
    URGENT_EXIT_FALLBACK_BUFFER_BPS,
    URGENT_EXIT_QUOTE_BUFFER_BPS,
    urgent_exit_limit_price,
)
from ai_trading.core.netting_submit_execution import execute_netting_submission
from ai_trading.core.netting_submit_prelude import prepare_netting_submit_prelude
from ai_trading.core.netting_symbol_cycle import process_netting_symbol
from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, validate_pretrade
from tests.core.test_submit_runtime import _base_cfg, _DummyExecEngine, _reset_submit_state
from tests.test_netting_submit_execution import _base_kwargs as _submission_kwargs
from tests.test_netting_submit_prelude import _base_kwargs as _prelude_kwargs
from tests.test_netting_symbol_cycle import _make_net_target, _make_processor

# Tuesday 2026-04-21 10:30 ET: a regular session.
BAR_TS = datetime(2026, 4, 21, 14, 30, tzinfo=UTC)
NOW = BAR_TS + timedelta(minutes=1)
CLOSE = 100.0


def _price(side: str, **kwargs: Any) -> Any:
    base: dict[str, Any] = {
        "symbol": "AAPL",
        "side": side,
        "bid": 99.5,
        "ask": 99.6,
        "quote_age_ms": 100.0,
        "max_quote_age_ms": 2_000,
        "fallback_price": CLOSE,
    }
    base.update(kwargs)
    return urgent_exit_limit_price(**base)


# ---- pricing -----------------------------------------------------------------


def test_buffers_are_small_and_inside_the_collar() -> None:
    assert URGENT_EXIT_QUOTE_BUFFER_BPS == 10.0
    assert URGENT_EXIT_FALLBACK_BUFFER_BPS == 50.0
    assert URGENT_EXIT_FALLBACK_BUFFER_BPS / 10_000.0 < 0.03


def test_sell_is_priced_below_the_bid() -> None:
    price = _price("sell", bid=100.0, ask=100.2)

    assert price.basis == PRICE_BASIS_QUOTE
    assert price.reference_price == 100.0
    # 100.00 * (1 - 10 bps) lands exactly on a tick and is not pushed further.
    assert price.limit_price == pytest.approx(99.90)
    assert price.quote_unusable_reason is None


def test_buy_to_cover_is_priced_above_the_ask() -> None:
    price = _price("buy", bid=99.8, ask=100.0)

    assert price.basis == PRICE_BASIS_QUOTE
    assert price.reference_price == 100.0
    assert price.limit_price == pytest.approx(100.10)


@pytest.mark.parametrize(
    ("side", "touch", "expected"),
    [
        ("sell", 10.03, 10.01),  # 10.01997 rounds down, away from the bid
        ("buy", 10.03, 10.05),  # 10.04003 rounds up, away from the ask
        ("sell", 0.004, 0.01),  # a sell never rounds to zero
    ],
)
def test_tick_rounding_keeps_the_limit_marketable(side: str, touch: float, expected: float) -> None:
    price = _price(side, bid=touch, ask=touch)

    assert price.limit_price == pytest.approx(expected)
    if touch > 0.01:
        assert (price.limit_price <= touch) if side == "sell" else (price.limit_price >= touch)


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        ({"bid": None, "ask": None}, "missing"),
        ({"bid": 99.5, "ask": None}, "missing"),
        ({"bid": 0.0, "ask": 99.6}, "missing"),
        ({"bid": 99.7, "ask": 99.6}, "crossed"),
        ({"quote_age_ms": 2_500.0}, "stale"),
    ],
)
def test_without_a_usable_quote_the_bar_close_is_moved_by_the_wider_buffer(
    quote: dict[str, Any], reason: str
) -> None:
    sell = _price("sell", **quote)
    cover = _price("buy", **quote)

    assert sell.basis == cover.basis == PRICE_BASIS_BAR_CLOSE
    assert sell.quote_unusable_reason == cover.quote_unusable_reason == reason
    assert sell.limit_price == pytest.approx(99.50)
    assert cover.limit_price == pytest.approx(100.50)


def test_unknown_age_or_disabled_staleness_keeps_the_quote() -> None:
    assert _price("sell", quote_age_ms=None).basis == PRICE_BASIS_QUOTE
    assert _price("sell", quote_age_ms=60_000.0, max_quote_age_ms=0).basis == PRICE_BASIS_QUOTE


def test_unpriceable_order_is_left_alone() -> None:
    assert _price("sell_short") is None
    assert _price("sell", bid=None, ask=None, fallback_price=0.0) is None


# ---- prelude: the urgent price reaches pretrade, nothing else changes ----------


def _quote(bid: float | None, ask: float | None, *, age_s: float = 0.1) -> Any:
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    quote_ts = NOW - timedelta(seconds=age_s) if bid is not None else None

    def _resolve(runtime: Any, symbol: str, side: str, fallback_price: float) -> tuple[Any, ...]:
        arrival = (ask if side == "buy" else bid) if mid is not None else fallback_price
        return ("broker_nbbo" if mid is not None else "bar_close", bid, ask, mid, arrival, quote_ts)

    return _resolve


_ORDERS: dict[str, dict[str, Any]] = {
    # name: current shares, order delta, side, urgent, protective
    "protective": {"current": 10, "delta": -10, "side": "sell", "urgent": True, "protective": True},
    "full_close": {"current": 10, "delta": -10, "side": "sell", "urgent": True, "protective": False},
    "cover_close": {"current": -10, "delta": 10, "side": "buy", "urgent": True, "protective": False},
    "trim": {"current": 10, "delta": -5, "side": "sell", "urgent": False, "protective": False},
    "add": {"current": 10, "delta": 5, "side": "buy", "urgent": False, "protective": False},
    "entry": {"current": 0, "delta": 10, "side": "buy", "urgent": False, "protective": False},
}
_UNCHANGED = ("trim", "add", "entry")


def _pretrade_cfg(**updates: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "seed": "seed",
        "execution_mode": "paper",
        "execution_require_realtime_nbbo": False,
        "paper_sampling_enabled": False,
        "paper_sampling_allowed_symbols": (),
        "max_order_dollars": 0.0,
        "max_order_shares": 0,
        "price_collar_pct": 0.03,
        "quote_max_age_ms": 2_000,
        "rth_only": True,
        "allow_extended": False,
    }
    base.update(updates)
    return SimpleNamespace(**base)


class _Ledger:
    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False


def _prelude(order: str, *, quote: Any, cfg: SimpleNamespace | None = None) -> tuple[Any, list[OrderIntent]]:
    spec = _ORDERS[order]
    pretrade_cfg = cfg or _pretrade_cfg()
    seen: list[OrderIntent] = []

    def _validate(intent: OrderIntent, **_kwargs: Any) -> tuple[bool, str, dict[str, Any]]:
        seen.append(intent)
        limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)
        return validate_pretrade(intent, cfg=pretrade_cfg, ledger=_Ledger(), rate_limiter=limiter)

    kwargs = _prelude_kwargs()
    kwargs.update(
        cfg=pretrade_cfg,
        now=NOW,
        bar_ts=BAR_TS,
        price=CLOSE,
        side=spec["side"],
        current_shares=float(spec["current"]),
        delta_shares=int(spec["delta"]),
        opening_trade=spec["current"] == 0,
        resolve_order_quote_basis_func=quote,
        safe_validate_pretrade_func=_validate,
        protective_exit=spec["protective"],
        urgent_exit=spec["urgent"],
    )
    return prepare_netting_submit_prelude(**cast(Any, kwargs)), seen


def test_urgent_exits_go_to_pretrade_at_the_marketable_price(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    sell, sell_seen = _prelude("full_close", quote=_quote(99.5, 99.6))
    cover, cover_seen = _prelude("cover_close", quote=_quote(99.5, 99.6))

    assert sell.blocked_reason is None and cover.blocked_reason is None
    assert sell.exit_limit_price == pytest.approx(99.40)  # 99.50 bid - 10 bps, down to the tick
    assert cover.exit_limit_price == pytest.approx(99.70)  # 99.60 ask + 10 bps, up to the tick
    assert sell_seen[0].limit_price == sell.exit_limit_price
    assert sell_seen[0].urgent_exit is True and sell_seen[0].protective_exit is False
    assert sell_seen[0].position_qty == 10.0
    assert sell.execution_intent_context.pretrade_intent.limit_price == sell.exit_limit_price
    assert sell.snapshot_updates["urgent_exit_pricing"]["price_basis"] == PRICE_BASIS_QUOTE
    markers = [r for r in caplog.records if r.getMessage().startswith("URGENT_EXIT_LIMIT_PRICE")]
    assert len(markers) == 2
    assert getattr(markers[0], "price_basis") == PRICE_BASIS_QUOTE
    assert getattr(markers[0], "symbol") == "AAPL"


@pytest.mark.parametrize("order", _UNCHANGED)
def test_trim_add_and_entry_keep_the_bar_close_limit(order: str) -> None:
    result, seen = _prelude(order, quote=_quote(99.5, 99.6))

    assert result.blocked_reason is None
    assert result.exit_limit_price is None
    assert seen[0].limit_price == CLOSE
    assert seen[0].urgent_exit is False and seen[0].protective_exit is False
    assert "urgent_exit_pricing" not in result.snapshot_updates


# ---- collar ------------------------------------------------------------------


@pytest.mark.parametrize("order", ["protective", "full_close"])
def test_collar_passes_an_urgent_exit_after_a_gap_from_the_bar_close(order: str) -> None:
    # The market gapped 5% below the 100.00 bar close.
    result, seen = _prelude(order, quote=_quote(95.0, 95.1))

    assert result.blocked_reason is None
    assert seen[0].limit_price == pytest.approx(94.90)


@pytest.mark.parametrize("order", _UNCHANGED)
def test_collar_still_blocks_trim_add_and_entry_after_the_same_gap(order: str) -> None:
    result, seen = _prelude(order, quote=_quote(95.0, 95.1))

    assert result.blocked_reason == "PRICE_COLLAR_BLOCK"
    assert seen[0].limit_price == CLOSE


# ---- stale and missing quotes --------------------------------------------------


def test_protective_exit_with_a_stale_quote_goes_out_at_the_bar_close_fallback() -> None:
    # Stale and 10% away from the close: neither staleness nor the collar holds it.
    result, seen = _prelude("protective", quote=_quote(90.0, 90.1, age_s=10.0))

    assert result.blocked_reason is None
    assert result.exit_limit_price == pytest.approx(99.50)
    assert seen[0].protective_exit is True
    pricing = result.snapshot_updates["urgent_exit_pricing"]
    assert pricing["price_basis"] == PRICE_BASIS_BAR_CLOSE
    assert pricing["quote_unusable_reason"] == "stale"


@pytest.mark.parametrize("order", ["full_close", *_UNCHANGED])
def test_stale_quote_still_blocks_a_normal_full_close_trim_add_and_entry(order: str) -> None:
    result, _seen = _prelude(order, quote=_quote(99.5, 99.6, age_s=10.0))

    assert result.blocked_reason == "STALE_QUOTE_BLOCK"


def test_protective_exit_without_a_quote_goes_out_at_the_bar_close_fallback() -> None:
    result, _seen = _prelude("protective", quote=_quote(None, None))

    assert result.blocked_reason is None
    assert result.exit_limit_price == pytest.approx(99.50)
    assert result.snapshot_updates["urgent_exit_pricing"]["quote_unusable_reason"] == "missing"


@pytest.mark.parametrize("order", ["full_close", *_UNCHANGED])
def test_missing_quote_still_blocks_a_normal_full_close_trim_add_and_entry(order: str) -> None:
    result, _seen = _prelude(order, quote=_quote(None, None))

    assert result.blocked_reason == "DERISK_DATA_QUALITY_BLOCK"


def test_hard_gates_still_hold_a_protective_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    crossed, _ = _prelude("protective", quote=_quote(99.7, 99.6))
    assert crossed.blocked_reason == "QUOTE_SANITY_BLOCK"

    kwargs = _prelude_kwargs()
    seen: list[OrderIntent] = []
    kwargs.update(
        cfg=_pretrade_cfg(),
        now=NOW,
        bar_ts=BAR_TS,
        price=CLOSE,
        side="sell",
        current_shares=10.0,
        delta_shares=-10,
        opening_trade=False,
        kill_switch_active=True,
        resolve_order_quote_basis_func=_quote(None, None),
        safe_validate_pretrade_func=lambda intent, **_k: (
            seen.append(intent),
            validate_pretrade(
                intent,
                cfg=_pretrade_cfg(),
                ledger=_Ledger(),
                rate_limiter=SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100),
            ),
        )[1],
        protective_exit=True,
        urgent_exit=True,
    )
    killed = prepare_netting_submit_prelude(**cast(Any, kwargs))
    assert killed.blocked_reason == "KILL_SWITCH_BLOCK"

    monkeypatch.setenv("AI_TRADING_DATA_DEGRADED", "1")
    degraded, _ = _prelude("protective", quote=_quote(99.5, 99.6))
    assert degraded.blocked_reason == "DERISK_DATA_QUALITY_BLOCK"


# ---- execution-quality gate ---------------------------------------------------


@pytest.mark.parametrize("order", ["protective", "full_close", "cover_close"])
def test_execution_quality_exemption_fires_for_an_urgent_exit(order: str) -> None:
    result, _seen = _prelude(order, quote=_quote(99.5, 99.7), cfg=_pretrade_cfg(execution_max_spread_bps=5.0))

    assert result.blocked_reason is None


@pytest.mark.parametrize("order", _UNCHANGED)
def test_execution_quality_gate_still_blocks_trim_add_and_entry(order: str) -> None:
    result, _seen = _prelude(order, quote=_quote(99.5, 99.7), cfg=_pretrade_cfg(execution_max_spread_bps=5.0))

    assert result.blocked_reason == "EXECUTION_QUALITY_SPREAD_BLOCK"


def _wide_spread_intent(**updates: Any) -> OrderIntent:
    base: dict[str, Any] = {
        "symbol": "AAPL",
        "side": "sell",
        "qty": 10,
        "notional": 1_000.0,
        "limit_price": 99.4,
        "bar_ts": BAR_TS,
        "client_order_id": "cid-eq",
        "last_price": 99.5,
        "mid": 99.6,
        "bid": 99.5,
        "ask": 99.7,
        "quote_quality_ok": True,
        "urgent_exit": True,
        "position_qty": 10.0,
    }
    base.update(updates)
    return OrderIntent(**cast(Any, base))


@pytest.mark.parametrize(
    "updates",
    [
        {"position_qty": None},  # no real position: not proven to reduce
        {"position_qty": 5.0},  # larger than the position
        {"position_qty": -10.0},  # adds to a short
        {"urgent_exit": False},  # a reducing order that is not urgent
    ],
)
def test_execution_quality_exemption_needs_an_urgent_exit_that_reduces(updates: dict[str, Any]) -> None:
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)
    cfg = _pretrade_cfg(execution_max_spread_bps=5.0)
    allowed, reason, _ = validate_pretrade(_wide_spread_intent(**updates), cfg=cfg, ledger=_Ledger(), rate_limiter=limiter)
    assert (allowed, reason) == (False, "EXECUTION_QUALITY_SPREAD_BLOCK")

    allowed, reason, _ = validate_pretrade(_wide_spread_intent(), cfg=cfg, ledger=_Ledger(), rate_limiter=limiter)
    assert (allowed, reason) == (True, "OK")


# ---- the cycle decides once and threads it through --------------------------


def _capture_cycle(
    positions: dict[str, int], target_dollars: float, **overrides: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processor, _records = _make_processor(positions=dict(positions), **overrides)
    prelude_calls: list[dict[str, Any]] = []
    submit_calls: list[dict[str, Any]] = []
    real_prelude = processor.prepare_submit_prelude_func
    real_submit = processor.execute_submission_func

    def _prelude_spy(**kwargs: Any) -> Any:
        prelude_calls.append(kwargs)
        result = real_prelude(**kwargs)
        result.exit_limit_price = 9.89 if kwargs["urgent_exit"] else None
        return result

    def _submit_spy(**kwargs: Any) -> Any:
        submit_calls.append(kwargs)
        return real_submit(**kwargs)

    processor.prepare_submit_prelude_func = _prelude_spy
    processor.execute_submission_func = _submit_spy
    net_target = _make_net_target(bar_ts=processor.now, target_dollars=target_dollars)
    process_netting_symbol(processor=processor, symbol="AAPL", net_target=net_target, orders_submitted=0)
    return prelude_calls, submit_calls


def test_cycle_marks_a_full_close_urgent_and_sends_its_limit() -> None:
    prelude_calls, submit_calls = _capture_cycle({"AAPL": 10}, 0.0)

    assert prelude_calls[0]["urgent_exit"] is True
    assert submit_calls[0]["urgent_exit"] is True
    assert submit_calls[0]["protective_exit"] is False
    assert submit_calls[0]["limit_price"] == 9.89
    assert submit_calls[0]["price"] == 10.0  # decision price unchanged


def test_a_full_close_clipped_later_stays_urgent() -> None:
    prelude_calls, submit_calls = _capture_cycle(
        {"AAPL": 10},
        0.0,
        enforce_participation_cap_func=lambda **k: (True, float(k["order_qty"]) / 2.0, "LIQ_PARTICIPATION_SCALE"),
    )

    assert submit_calls[0]["delta_shares"] == -5
    assert prelude_calls[0]["urgent_exit"] is True
    assert submit_calls[0]["urgent_exit"] is True


@pytest.mark.parametrize(
    ("positions", "target_dollars"),
    [({"AAPL": 10}, 50.0), ({"AAPL": 10}, 150.0), ({}, 100.0)],
    ids=["trim", "add", "entry"],
)
def test_cycle_leaves_trim_add_and_entry_unmarked(positions: dict[str, int], target_dollars: float) -> None:
    prelude_calls, submit_calls = _capture_cycle(positions, target_dollars)

    assert prelude_calls[0]["urgent_exit"] is False
    assert submit_calls[0]["urgent_exit"] is False
    assert submit_calls[0]["limit_price"] is None


# ---- submission and the submit-time pretrade ----------------------------------


def test_submission_sends_the_urgent_limit_and_keeps_the_decision_price() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tca_prices: list[float] = []
    kwargs = _submission_kwargs()
    kwargs["submit_order_func"] = lambda *a, **k: (calls.append((a, k)), SimpleNamespace(status="new"))[1]
    real_tca = kwargs["build_order_metrics_and_tca_func"]

    def _tca(**k: Any) -> Any:
        tca_prices.append(k["price"])
        return real_tca(**k)

    kwargs["build_order_metrics_and_tca_func"] = _tca
    kwargs.update(limit_price=kwargs["price"] - 0.11, urgent_exit=True, protective_exit=True)

    execute_netting_submission(**cast(Any, kwargs))

    assert calls[0][1]["price"] == pytest.approx(kwargs["price"] - 0.11)
    assert calls[0][1]["urgent_exit"] is True and calls[0][1]["protective_exit"] is True
    assert tca_prices == [kwargs["price"]]


def test_submission_of_a_normal_order_is_unchanged() -> None:
    calls: list[dict[str, Any]] = []
    kwargs = _submission_kwargs()
    kwargs["submit_order_func"] = lambda *a, **k: (calls.append(k), SimpleNamespace(status="new"))[1]

    execute_netting_submission(**cast(Any, kwargs))

    assert calls[0]["price"] == kwargs["price"]
    assert "urgent_exit" not in calls[0] and "protective_exit" not in calls[0]


def _runtime_submit(
    monkeypatch: pytest.MonkeyPatch, *, held: str, side: str, qty: int, **exec_kwargs: Any
) -> tuple[Any, _DummyExecEngine]:
    engine = _DummyExecEngine()
    positions = [SimpleNamespace(symbol="AAPL", qty=held, side="long")] if held else []
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: positions))
    stale_quote_ts = datetime.now(UTC) - timedelta(seconds=30)
    _reset_submit_state(monkeypatch)
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: _base_cfg(quote_max_age_ms=5))
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_a, **_k: ("broker_nbbo", 99.95, 100.05, 100.0, 100.0, stale_quote_ts),
    )
    order = bot_engine.submit_order(ctx, "AAPL", qty, side, price=99.5, **exec_kwargs)
    return order, engine


def test_submit_time_pretrade_lets_a_protective_exit_through_a_stale_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    order, engine = _runtime_submit(
        monkeypatch, held="10", side="sell", qty=10, closing_position=True, urgent_exit=True, protective_exit=True
    )

    assert order is not None
    assert engine.calls[0]["price"] == 99.5
    assert "urgent_exit" not in engine.calls[0]["kwargs"]
    assert "protective_exit" not in engine.calls[0]["kwargs"]


@pytest.mark.parametrize(
    ("held", "side", "qty", "exec_kwargs"),
    [
        ("10", "sell", 10, {"closing_position": True, "urgent_exit": True}),  # normal full close
        ("10", "sell", 5, {"closing_position": True}),  # trim
        ("10", "buy", 5, {"urgent_exit": True, "protective_exit": True}),  # add: flags ignored
        ("", "buy", 5, {"urgent_exit": True, "protective_exit": True}),  # entry: flags ignored
    ],
    ids=["full_close", "trim", "add", "entry"],
)
def test_submit_time_pretrade_still_blocks_other_orders_on_a_stale_quote(
    monkeypatch: pytest.MonkeyPatch, held: str, side: str, qty: int, exec_kwargs: dict[str, Any]
) -> None:
    order, engine = _runtime_submit(monkeypatch, held=held, side=side, qty=qty, **exec_kwargs)

    assert order is None
    assert engine.calls == []
    assert engine.skips[-1]["reason"] == "STALE_QUOTE_BLOCK"


# ---- fractional broker quantity -------------------------------------------------


def test_fractional_position_qty_is_truncated_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    positions = [
        SimpleNamespace(symbol="AAPL", qty="10.5"),
        SimpleNamespace(symbol="MSFT", qty="-3.75"),
        SimpleNamespace(symbol="NVDA", qty="7"),
        SimpleNamespace(symbol="TSLA", qty="0.4"),
    ]
    ctx = SimpleNamespace(api=SimpleNamespace(get_all_positions=lambda: positions))
    caplog.set_level(logging.WARNING)

    assert bot_engine.compute_current_positions(ctx) == {"AAPL": 10, "MSFT": -3, "NVDA": 7, "TSLA": 0}
    warned = {
        getattr(r, "symbol", None)
        for r in caplog.records
        if r.getMessage().startswith("POSITION_FRACTIONAL_QTY_TRUNCATED")
    }
    assert warned == {"AAPL", "MSFT", "TSLA"}


@pytest.mark.parametrize("qty", ["abc", "NaN", "Infinity"])
def test_unparseable_position_qty_still_fails_closed(qty: str) -> None:
    ctx = SimpleNamespace(api=SimpleNamespace(list_positions=lambda: [SimpleNamespace(symbol="AAPL", qty=qty)]))

    with pytest.raises(RuntimeError, match="broker_positions_unavailable"):
        bot_engine.compute_current_positions(ctx)
