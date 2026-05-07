from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.core.netting_cycle_setup import (
    NettingPreparationError,
    prepare_netting_cycle_inputs,
)


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict | None]] = []

    def info(self, event: str, extra: dict | None = None, **_: object) -> None:
        self.events.append((event, extra))

    def warning(self, event: str, extra: dict | None = None, **_: object) -> None:
        self.events.append((event, extra))

    def debug(self, event: str, extra: dict | None = None, **_: object) -> None:
        self.events.append((event, extra))


def _enabled_sleeves() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="intraday",
            enabled=True,
            timeframe="1m",
            entry_threshold=0.1,
            exit_threshold=0.05,
            flip_threshold=0.2,
            reentry_threshold=0.1,
            deadband_dollars=10.0,
            deadband_shares=1.0,
            cost_k=1.0,
            edge_scale_bps=10.0,
            turnover_cap_dollars=1000.0,
            max_symbol_dollars=5000.0,
            max_gross_dollars=10000.0,
        )
    ]


def _prepare_kwargs(
    *,
    state: SimpleNamespace,
    runtime: SimpleNamespace,
    logger: _Logger,
    env: dict[str, object],
    pre_rank_calls: list[list[str]] | None = None,
) -> dict[str, object]:
    def _get_env(key: str, default: object = None, cast: object = None) -> object:
        return env.get(key, default)

    return {
        "state": state,
        "runtime": runtime,
        "cfg": SimpleNamespace(),
        "now": datetime(2026, 4, 19, 14, 30, tzinfo=UTC),
        "market_open_now": True,
        "breakers": SimpleNamespace(record_failure=lambda _dep, _error: None),
        "logger": logger,
        "get_env": _get_env,
        "build_sleeve_configs_func": lambda _cfg: _enabled_sleeves(),
        "resolve_runtime_sleeve_whitelist_func": lambda: set(),
        "maybe_update_allocation_state_func": lambda *_args, **_kwargs: None,
        "allocation_weights_for_sleeves_func": lambda _cfg, _sleeves: {"intraday": 1.0},
        "load_learned_overrides_func": lambda _cfg: {},
        "load_candidate_universe_func": lambda _runtime: [],
        "pre_rank_execution_candidates_func": (
            lambda items: (
                pre_rank_calls.append(list(items)) if pre_rank_calls is not None else None
            )
            or list(items)
        ),
        "ensure_data_fetcher_func": lambda _runtime: None,
        "retry_idempotent_func": lambda fn, **_kwargs: fn(),
        "compute_current_positions_func": lambda _runtime: {},
        "classify_exception_func": lambda exc, **_kwargs: exc,
        "handle_error_func": lambda *_args, **_kwargs: None,
        "merge_managed_position_symbols_func": lambda items, positions: list(items)
        + [symbol for symbol in positions if symbol not in items],
        "pending_orders_block_scope_func": lambda: "none",
        "get_cycle_budget_context_func": lambda: None,
        "resolve_adaptive_order_cap_func": lambda **_kwargs: (
            None,
            {"mode": "none", "headroom_ratio": 1.0},
        ),
        "select_symbols_with_budget_rotation_func": lambda items, positions, **_kwargs: (
            list(items),
            0,
            0,
        ),
        "pending_order_blocked_symbols_attr": "_pending_order_blocked_symbols",
        "pending_order_sample_limit": 5,
    }


def test_prepare_netting_cycle_inputs_returns_filtered_symbols_and_positions() -> None:
    state = SimpleNamespace(
        canary_mode_logged=False,
        last_loop_duration=0.0,
        netting_symbol_budget_cursor=0,
    )
    runtime = SimpleNamespace(
        tickers=["AAPL", "MSFT", "NVDA"],
        execution_engine=SimpleNamespace(
            _resolve_order_submit_cap=lambda: (1, "static"),
        ),
        _pending_order_blocked_symbols=["NVDA"],
    )
    cfg = SimpleNamespace()
    logger = _Logger()
    breaker_calls: list[str] = []
    breakers = SimpleNamespace(record_failure=lambda dep, error: breaker_calls.append(dep))
    sleeves = [
        SimpleNamespace(
            name="intraday",
            enabled=True,
            timeframe="1m",
            entry_threshold=0.1,
            exit_threshold=0.05,
            flip_threshold=0.2,
            reentry_threshold=0.1,
            deadband_dollars=10.0,
            deadband_shares=1.0,
            cost_k=1.0,
            edge_scale_bps=10.0,
            turnover_cap_dollars=1000.0,
            max_symbol_dollars=5000.0,
            max_gross_dollars=10000.0,
        )
    ]

    result = prepare_netting_cycle_inputs(
        state=state,
        runtime=runtime,
        cfg=cfg,
        now=datetime(2026, 4, 19, 14, 30, tzinfo=UTC),
        market_open_now=True,
        breakers=breakers,
        logger=logger,
        get_env=lambda key, default=None, cast=None: {
            "AI_TRADING_CANARY_SYMBOLS": "AAPL,MSFT,NVDA",
            "AI_TRADING_CANARY_PERCENT": 0.0,
            "AI_TRADING_EXEC_SYMBOLS_PER_ORDER": 1,
            "AI_TRADING_EXEC_SYMBOL_BUDGET_MIN": 1,
            "AI_TRADING_EXEC_SYMBOL_BUDGET_MAX": 1,
            "AI_TRADING_WARMUP_MODE": False,
        }.get(key, default),
        build_sleeve_configs_func=lambda _cfg: sleeves,
        resolve_runtime_sleeve_whitelist_func=lambda: set(),
        maybe_update_allocation_state_func=lambda *_args, **_kwargs: None,
        allocation_weights_for_sleeves_func=lambda _cfg, _sleeves: {"intraday": 1.0},
        load_learned_overrides_func=lambda _cfg: {"foo": "bar"},
        load_candidate_universe_func=lambda _runtime: [],
        pre_rank_execution_candidates_func=lambda items: list(items),
        ensure_data_fetcher_func=lambda _runtime: None,
        retry_idempotent_func=lambda fn, **_kwargs: fn(),
        compute_current_positions_func=lambda _runtime: {"AAPL": 2, "SPY": 1},
        classify_exception_func=lambda exc, **_kwargs: exc,
        handle_error_func=lambda *_args, **_kwargs: None,
        merge_managed_position_symbols_func=lambda items, positions: list(items) + [
            symbol for symbol in positions if symbol not in items
        ],
        pending_orders_block_scope_func=lambda: "symbol",
        get_cycle_budget_context_func=lambda: None,
        resolve_adaptive_order_cap_func=lambda **_kwargs: (1, {"mode": "static", "headroom_ratio": 1.0}),
        select_symbols_with_budget_rotation_func=lambda items, positions, **_kwargs: (
            list(items[:1]),
            0,
            int(sum(1 for symbol in items if symbol in positions)),
        ),
        pending_order_blocked_symbols_attr="_pending_order_blocked_symbols",
        pending_order_sample_limit=5,
    )

    assert result is not None
    assert [str(getattr(sleeve, "name", "")) for sleeve in result.sleeves] == ["intraday"]
    assert result.allocation_weights == {"intraday": 1.0}
    assert result.learned_overrides == {"foo": "bar"}
    assert "SPY" in result.positions
    assert result.symbols == ["AAPL"]


def test_symbol_prune_shadow_mode_preserves_candidates_before_prerank() -> None:
    state = SimpleNamespace(
        canary_mode_logged=False,
        last_loop_duration=0.0,
        netting_symbol_budget_cursor=0,
    )
    runtime = SimpleNamespace(tickers=["AAPL", "MSFT"], execution_engine=None)
    logger = _Logger()
    pre_rank_calls: list[list[str]] = []

    result = prepare_netting_cycle_inputs(
        **_prepare_kwargs(
            state=state,
            runtime=runtime,
            logger=logger,
            env={
                "AI_TRADING_SYMBOL_PRUNE_ENABLED": True,
                "AI_TRADING_SYMBOL_PRUNE_MODE": "shadow",
                "AI_TRADING_SYMBOL_PRUNE_DISABLED_SYMBOLS": "MSFT",
                "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_ENABLED": False,
            },
            pre_rank_calls=pre_rank_calls,
        )
    )

    assert result is not None
    assert pre_rank_calls == [["AAPL", "MSFT"]]
    assert result.symbols == ["AAPL", "MSFT"]
    assert any(event == "SYMBOL_UNIVERSE_PRUNE_EVALUATED" for event, _ in logger.events)


def test_symbol_prune_enforce_removes_disabled_before_prerank() -> None:
    state = SimpleNamespace(
        canary_mode_logged=False,
        last_loop_duration=0.0,
        netting_symbol_budget_cursor=0,
    )
    runtime = SimpleNamespace(tickers=["AAPL", "MSFT", "NVDA"], execution_engine=None)
    logger = _Logger()
    pre_rank_calls: list[list[str]] = []

    result = prepare_netting_cycle_inputs(
        **_prepare_kwargs(
            state=state,
            runtime=runtime,
            logger=logger,
            env={
                "AI_TRADING_SYMBOL_PRUNE_ENABLED": True,
                "AI_TRADING_SYMBOL_PRUNE_MODE": "enforce",
                "AI_TRADING_SYMBOL_PRUNE_DISABLED_SYMBOLS": "MSFT",
                "AI_TRADING_SYMBOL_PRUNE_ALLOWLIST": "AAPL,MSFT",
                "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_ENABLED": False,
            },
            pre_rank_calls=pre_rank_calls,
        )
    )

    assert result is not None
    assert pre_rank_calls == [["AAPL"]]
    assert result.symbols == ["AAPL"]


def test_universe_mismatch_alert_when_research_symbols_are_not_executable(
    tmp_path,
) -> None:
    report_path = tmp_path / "daily_research_latest.json"
    report_path.write_text(json.dumps({"symbols": "AAPL,AMZN"}), encoding="utf-8")
    state = SimpleNamespace(
        canary_mode_logged=False,
        last_loop_duration=0.0,
        netting_symbol_budget_cursor=0,
    )
    runtime = SimpleNamespace(tickers=["AAPL", "AMZN"], execution_engine=None)
    logger = _Logger()

    result = prepare_netting_cycle_inputs(
        **_prepare_kwargs(
            state=state,
            runtime=runtime,
            logger=logger,
            env={
                "AI_TRADING_CANARY_SYMBOLS": "AAPL",
                "AI_TRADING_CANARY_PERCENT": 0.0,
                "AI_TRADING_DAILY_RESEARCH_REPORT_PATH": str(report_path),
                "AI_TRADING_UNIVERSE_MISMATCH_ALERT_ENABLED": True,
                "AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS": "AMZN",
            },
        )
    )

    assert result is not None
    assert result.symbols == ["AAPL"]
    events = [event for event, _extra in logger.events]
    assert "UNIVERSE_MISMATCH_ALERT" in events
    mismatch = next(extra for event, extra in logger.events if event == "UNIVERSE_MISMATCH_ALERT")
    assert mismatch is not None
    assert mismatch["missing_executable_symbols"] == ["AMZN"]


def test_explicit_canary_symbols_extend_runtime_universe_before_prerank() -> None:
    state = SimpleNamespace(
        canary_mode_logged=False,
        last_loop_duration=0.0,
        netting_symbol_budget_cursor=0,
    )
    runtime = SimpleNamespace(tickers=["AAPL"], execution_engine=None)
    logger = _Logger()
    pre_rank_calls: list[list[str]] = []

    result = prepare_netting_cycle_inputs(
        **_prepare_kwargs(
            state=state,
            runtime=runtime,
            logger=logger,
            env={
                "AI_TRADING_CANARY_SYMBOLS": "AAPL,AMZN",
                "AI_TRADING_CANARY_PERCENT": 0.0,
                "AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS": "MSFT",
                "AI_TRADING_UNIVERSE_MISMATCH_ALERT_ENABLED": True,
                "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_ENABLED": False,
            },
            pre_rank_calls=pre_rank_calls,
        )
    )

    assert result is not None
    assert pre_rank_calls == [["AAPL", "AMZN"]]
    assert result.symbols == ["AAPL", "AMZN"]
    assert any(
        event == "CANARY_SYMBOLS_ADDED_TO_RUNTIME_UNIVERSE"
        and extra == {
            "added": ["AMZN"],
            "runtime_symbols_before": 1,
            "runtime_symbols_after": 2,
        }
        for event, extra in logger.events
    )
    assert not any(
        event == "UNIVERSE_MISMATCH_ALERT"
        and "AMZN" in (extra or {}).get("missing_executable_symbols", [])
        for event, extra in logger.events
    )


def test_prepare_netting_cycle_inputs_raises_preparation_error_on_position_failure() -> None:
    state = SimpleNamespace(canary_mode_logged=False, last_loop_duration=0.0)
    runtime = SimpleNamespace(tickers=["AAPL"], execution_engine=None)
    logger = _Logger()
    handled: list[str] = []
    error_info = SimpleNamespace(reason_code="BROKER_POSITIONS_FAILED")
    breakers = SimpleNamespace(record_failure=lambda dep, err: handled.append(dep))

    with pytest.raises(NettingPreparationError) as exc_info:
        prepare_netting_cycle_inputs(
            state=state,
            runtime=runtime,
            cfg=SimpleNamespace(),
            now=datetime(2026, 4, 19, 14, 30, tzinfo=UTC),
            market_open_now=True,
            breakers=breakers,
            logger=logger,
            get_env=lambda _key, default=None, cast=None: default,
            build_sleeve_configs_func=lambda _cfg: [
                SimpleNamespace(
                    name="intraday",
                    enabled=True,
                    timeframe="1m",
                    entry_threshold=0.1,
                    exit_threshold=0.05,
                    flip_threshold=0.2,
                    reentry_threshold=0.1,
                    deadband_dollars=10.0,
                    deadband_shares=1.0,
                    cost_k=1.0,
                    edge_scale_bps=10.0,
                    turnover_cap_dollars=1000.0,
                    max_symbol_dollars=5000.0,
                    max_gross_dollars=10000.0,
                )
            ],
            resolve_runtime_sleeve_whitelist_func=lambda: set(),
            maybe_update_allocation_state_func=lambda *_args, **_kwargs: None,
            allocation_weights_for_sleeves_func=lambda _cfg, _sleeves: {"intraday": 1.0},
            load_learned_overrides_func=lambda _cfg: {},
            load_candidate_universe_func=lambda _runtime: [],
            pre_rank_execution_candidates_func=lambda items: list(items),
            ensure_data_fetcher_func=lambda _runtime: None,
            retry_idempotent_func=lambda fn, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            compute_current_positions_func=lambda _runtime: {"AAPL": 1},
            classify_exception_func=lambda _exc, **_kwargs: error_info,
            handle_error_func=lambda info, **_kwargs: handled.append(str(info.reason_code)),
            merge_managed_position_symbols_func=lambda items, positions: list(items),
            pending_orders_block_scope_func=lambda: "none",
            get_cycle_budget_context_func=lambda: None,
            resolve_adaptive_order_cap_func=lambda **_kwargs: (None, {"mode": "none", "headroom_ratio": 1.0}),
            select_symbols_with_budget_rotation_func=lambda items, positions, **_kwargs: (list(items), 0, 0),
            pending_order_blocked_symbols_attr="_pending_order_blocked_symbols",
            pending_order_sample_limit=5,
        )

    assert exc_info.value.reason_code == "BROKER_POSITIONS_FAILED"
    assert "broker_positions" in handled
    assert "BROKER_POSITIONS_FAILED" in handled
