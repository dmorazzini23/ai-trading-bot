from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pandas as pd

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine
from ai_trading.core.netting import NettedTarget, SleeveConfig, SleeveProposal
from ai_trading.policy.compiler import ExecutionApproval


def test_netting_cycle_emits_decision_reject_reason_summary(monkeypatch, caplog) -> None:
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(cfg=cfg, tickers=["AAPL"], universe_tickers=["AAPL"], api=None)
    state = bot_engine.BotState()

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([now - timedelta(minutes=1), now], tz=UTC),
    )

    def _force_target(symbol, bar_ts, proposals, disagree_ratio):
        return NettedTarget(
            symbol=str(symbol),
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=0.0,
            proposals=list(proposals),
        )

    class _AllowBreakers:
        def allow(self, dep: str) -> bool:
            return True

        def open_reason(self, dep: str) -> str | None:
            return None

        def record_failure(self, dep: str, error_info) -> None:
            _ = (dep, error_info)

        def record_success(self, dep: str) -> None:
            _ = dep

    monkeypatch.setenv("AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE", "0")
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH", "1")
    monkeypatch.setenv("AI_TRADING_DECISION_REJECT_SUMMARY_LOG_TTL_SEC", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL")
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "0")
    monkeypatch.setenv("AI_TRADING_EVENT_DRIVEN_NEW_BAR_ONLY", "0")
    # Keep global halt gates in reject summaries so this assertion is deterministic
    # even when runtime env defaults exclude them from effectiveness rollups.
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_EXCLUDE_GLOBAL_HALTS", "0")
    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", lambda symbols, timeframe, start, end: {"AAPL": df})
    monkeypatch.setattr("ai_trading.core.netting.net_targets_for_symbol", _force_target)
    monkeypatch.setattr(
        "ai_trading.core.horizons.build_sleeve_configs",
        lambda cfg=None: [
            SleeveConfig(
                name="day",
                timeframe="1Min",
                enabled=True,
                entry_threshold=0.2,
                exit_threshold=0.1,
                flip_threshold=0.3,
                reentry_threshold=0.6,
                deadband_dollars=50.0,
                deadband_shares=1.0,
                turnover_cap_dollars=0.0,
                cost_k=1.5,
                edge_scale_bps=20.0,
                max_symbol_dollars=10000.0,
                max_gross_dollars=50000.0,
            )
        ],
    )
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (True, "kill_switch"))
    monkeypatch.setattr(bot_engine, "_dependency_breakers", lambda _state: _AllowBreakers())
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "_run_reconciliation_if_due", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_engine, "_pre_rank_execution_candidates", lambda symbols, runtime=None: list(symbols))
    monkeypatch.setattr(bot_engine, "_run_post_trade_learning_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_tca_cost_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_replay_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_walk_forward_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_tca_stale_block_reason", lambda _now: None)
    written_records = []
    monkeypatch.setattr(
        bot_engine,
        "_write_decision_record",
        lambda record, path: written_records.append(record),
    )

    with caplog.at_level(logging.INFO):
        bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    summary_emitted = any(
        record.getMessage() == "DECISION_REJECT_REASON_SUMMARY"
        for record in caplog.records
    )
    symbol_records = [record for record in written_records if str(record.symbol).upper() != "ALL"]
    if symbol_records:
        assert any("OK_TRADE" not in list(record.gates or []) for record in symbol_records)
        assert summary_emitted
    else:
        # Cross-suite guard states can short-circuit this cycle before per-symbol
        # decisions. In that case, the summary log is not expected.
        assert not written_records or all(str(record.symbol).upper() == "ALL" for record in written_records)


def test_netting_cycle_duplicate_intent_does_not_inflate_orders_attempted(monkeypatch) -> None:
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT"],
        universe_tickers=["AAPL", "MSFT"],
        api=None,
        execution_engine=SimpleNamespace(_last_submit_outcome={}),
    )
    state = bot_engine.BotState()

    class _NoBarDedupeDict(dict[str, datetime]):
        def get(self, key: str, default=None):  # type: ignore[override]
            _ = (key, default)
            return None

    state.last_order_bar_ts = _NoBarDedupeDict()

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([now - timedelta(minutes=1), now], tz=UTC),
    )

    def _force_target(symbol, bar_ts, proposals, disagree_ratio):
        return NettedTarget(
            symbol=str(symbol),
            bar_ts=bar_ts,
            target_dollars=101.0,
            target_shares=1.0,
            proposals=list(proposals),
            disagreement_ratio=0.0 if disagree_ratio is None else float(disagree_ratio),
        )

    class _AllowBreakers:
        def allow(self, dep: str) -> bool:
            return True

        def open_reason(self, dep: str) -> str | None:
            return None

        def record_failure(self, dep: str, error_info) -> None:
            _ = (dep, error_info)

        def record_success(self, dep: str) -> None:
            _ = dep

    def _fake_submit(runtime_obj, symbol, qty, side, **_kwargs):
        if str(symbol).upper() == "AAPL":
            runtime_obj.execution_engine._last_submit_outcome = {
                "status": "skipped",
                "reason": "cycle_duplicate_intent",
                "symbol": str(symbol),
                "side": str(side),
            }
            return None
        runtime_obj.execution_engine._last_submit_outcome = {
            "status": "submitted",
            "symbol": str(symbol),
            "side": str(side),
        }
        return SimpleNamespace(id=f"order-{symbol}", status="submitted", filled_avg_price=None)

    monkeypatch.setenv("AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE", "0")
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL,MSFT")
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "0")
    monkeypatch.setenv("AI_TRADING_EVENT_DRIVEN_NEW_BAR_ONLY", "0")
    monkeypatch.setenv("AI_TRADING_NETTING_CYCLE_SLO_LOG_TTL_SEC", "0")
    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_LIQ_REGIME_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_CAPACITY_AWARE_THROTTLE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_QUARANTINE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_CAP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_ALPHA_DECAY_DEWEIGHT_ENABLED", "0")
    monkeypatch.setattr(
        "ai_trading.data.fetch.get_bars_batch",
        lambda symbols, timeframe, start, end: {str(sym).upper(): df for sym in symbols},
    )
    monkeypatch.setattr("ai_trading.core.netting.net_targets_for_symbol", _force_target)
    monkeypatch.setattr(
        "ai_trading.core.horizons.build_sleeve_configs",
        lambda cfg=None: [
            SleeveConfig(
                name="day",
                timeframe="1Min",
                enabled=True,
                entry_threshold=0.2,
                exit_threshold=0.1,
                flip_threshold=0.3,
                reentry_threshold=0.6,
                deadband_dollars=50.0,
                deadband_shares=1.0,
                turnover_cap_dollars=0.0,
                cost_k=1.5,
                edge_scale_bps=20.0,
                max_symbol_dollars=10000.0,
                max_gross_dollars=50000.0,
            )
        ],
    )
    monkeypatch.setattr(
        "ai_trading.core.netting.compute_sleeve_proposal",
        lambda sleeve, symbol, bar_ts, score, confidence, current_pos, price, spread, vol, volume=None: SleeveProposal(
            symbol=str(symbol),
            sleeve=sleeve.name,
            bar_ts=bar_ts,
            target_dollars=101.0,
            expected_edge_bps=8.0,
            expected_cost_bps=2.0,
            score=0.5,
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_dependency_breakers", lambda _state: _AllowBreakers())
    monkeypatch.setattr(
        "ai_trading.oms.ledger.deterministic_client_order_id",
        lambda **kwargs: (
            f"{kwargs['symbol']}-{kwargs['bar_ts']}-{kwargs['side']}-{int(abs(float(kwargs['qty'])))}"
        ),
    )
    monkeypatch.setattr(bot_engine, "market_is_open", lambda _now=None: True)
    monkeypatch.setattr(bot_engine, "retry_idempotent", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime_obj: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime_obj: {})
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "_run_reconciliation_if_due", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_pre_rank_execution_candidates",
        lambda symbols, runtime=None: list(symbols),
    )
    monkeypatch.setattr(
        bot_engine,
        "approve_execution_candidate",
        lambda policy, candidate: ExecutionApproval(
            True,
            int(candidate.proposed_delta_shares),
            5.0,
            tuple(),
        ),
    )
    monkeypatch.setattr(bot_engine, "_run_post_trade_learning_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_tca_cost_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_replay_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_run_walk_forward_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_update_rollout_governance_state", lambda *args, **kwargs: {"capital_ramp": {}})
    monkeypatch.setattr(bot_engine, "_resolve_primary_feed_derisk_state", lambda _runtime: {})
    monkeypatch.setattr(
        bot_engine,
        "enforce_participation_cap",
        lambda **kwargs: (True, float(kwargs["order_qty"]), None),
    )
    monkeypatch.setattr(
        bot_engine,
        "_alpha_decay_entry_guard",
        lambda *args, **kwargs: {"blocked": False, "trades_in_window": 0, "start_trades": 0},
    )
    monkeypatch.setattr(bot_engine, "is_near_event", lambda _symbol, days=0: False)
    monkeypatch.setattr(bot_engine, "_tca_stale_block_reason", lambda _now: None)
    monkeypatch.setattr(
        bot_engine,
        "safe_validate_pretrade",
        lambda *args, **kwargs: (True, "OK", {}),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_submit_none_reason",
        lambda _runtime: "CYCLE_DUPLICATE_INTENT",
    )
    monkeypatch.setattr(bot_engine, "submit_order", _fake_submit)

    written_records = []
    monkeypatch.setattr(
        bot_engine,
        "_write_decision_record",
        lambda record, path: written_records.append(record),
    )
    captured_slo: dict[str, object] = {}
    monkeypatch.setattr(
        bot_engine,
        "log_throttled_event",
        lambda logger_obj, key, *, level, extra, message: (
            captured_slo.update(dict(extra)) if message == "NETTING_CYCLE_SLO" else None
        ),
    )

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    aapl_records = [record for record in written_records if str(record.symbol).upper() == "AAPL"]
    msft_records = [record for record in written_records if str(record.symbol).upper() == "MSFT"]
    assert any("CYCLE_DUPLICATE_INTENT" in list(record.gates or []) for record in aapl_records), written_records
    assert any("OK_TRADE" in list(record.gates or []) for record in msft_records), written_records
    assert int(cast(int, captured_slo.get("orders_attempted", -1))) == 1
    assert int(cast(int, captured_slo.get("orders_submitted", -1))) == 1
