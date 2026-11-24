"""The rule scorer trades only as labelled paper sampling; every other mode holds.

The expected paper-sampling targets were captured from the retired
stale-model diagnostic path (model stale, ``AI_TRADING_REQUIRE_ML_MODEL=1``,
diagnostics enabled) on the same synthetic bars, so they pin that retiring the
day-sleeve model changed labels only, not decisions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine, governance_runtime
from ai_trading.core import netting as netting_mod
from ai_trading.core.netting import SleeveConfig

_NOW = datetime(2026, 11, 18, 15, 2, 30, tzinfo=UTC)  # Wednesday 10:02:30 ET
_SYMBOLS = ("AAPL", "MSFT", "AMZN", "NVDA", "SPY")
_POSITIONS = {"AAPL": 10.0, "AMZN": -5.0, "NVDA": 3.0}
_DRIFT = {"AAPL": 0.0006, "MSFT": -0.0007, "AMZN": 0.0004, "NVDA": -0.0003, "SPY": 0.00005}
_PAPER = {
    "execution_mode": "paper",
    "paper": True,
    "alpaca_base_url": "https://paper-api.alpaca.markets",
}
_LIVE = {
    "execution_mode": "live",
    "paper": False,
    "alpaca_base_url": "https://api.alpaca.markets",
}


def _bars(symbol: str) -> pd.DataFrame:
    rng = np.random.default_rng(sum(ord(char) for char in symbol))
    index: list[datetime] = []
    day = _NOW.date()
    sessions = 0
    while sessions < 12:
        if day.weekday() < 5:
            session_open = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)
            index.extend(
                ts
                for ts in (session_open + timedelta(minutes=5 * k) for k in range(78))
                if ts <= _NOW
            )
            sessions += 1
        day -= timedelta(days=1)
    index.sort()
    close = 100.0 * np.exp(np.cumsum(_DRIFT[symbol] + rng.normal(0.0, 0.002, len(index))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(5_000, 50_000, len(index)).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0008,
            "low": np.minimum(open_, close) * 0.9992,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(index, tz=UTC),
    )


class _AllowBreakers:
    def allow(self, dep: str) -> bool:
        return True

    def open_reason(self, dep: str) -> str | None:
        return None

    def record_failure(self, dep: str, error_info: Any) -> None:
        return None

    def record_success(self, dep: str) -> None:
        return None


def _run_cycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg_updates: dict[str, Any],
    sleeves: tuple[tuple[str, str], ...] = (("day", "5Min"),),
) -> list[tuple[str, Any, Any]]:
    for key, value in {
        "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
        "AI_TRADING_KILL_SWITCH": "1",
        "AI_TRADING_WARMUP_MODE": "0",
        "AI_TRADING_EVENT_DRIVEN_NEW_BAR_ONLY": "0",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = TradingConfig.from_env(allow_missing_drawdown=True).update(
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
        **cfg_updates,
    )
    runtime = SimpleNamespace(
        cfg=cfg, tickers=list(_SYMBOLS), universe_tickers=list(_SYMBOLS), api=None
    )
    frames = {symbol: _bars(symbol) for symbol in _SYMBOLS}
    captured: list[tuple[str, Any, Any]] = []
    real_net = netting_mod.net_targets_for_symbol

    def _capture(symbol, bar_ts, proposals, disagree_ratio):
        result = real_net(symbol, bar_ts, proposals, disagree_ratio)
        for proposal in proposals:
            captured.append((symbol, proposal, result))
        return result

    monkeypatch.setattr(
        governance_runtime,
        "run_netting_cycle_governance",
        lambda _state: governance_runtime.NettingGovernanceSnapshot(
            now=_NOW, market_open_now=True
        ),
    )
    monkeypatch.setattr(
        "ai_trading.data.fetch.get_bars_batch",
        lambda symbols, timeframe, start, end: {
            symbol: frames[symbol].copy() for symbol in symbols if symbol in frames
        },
    )
    monkeypatch.setattr(netting_mod, "net_targets_for_symbol", _capture)
    monkeypatch.setattr(
        "ai_trading.core.horizons.build_sleeve_configs",
        lambda cfg=None: [
            SleeveConfig(
                name=name,
                timeframe=timeframe,
                enabled=True,
                entry_threshold=0.05,
                exit_threshold=0.02,
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
            for name, timeframe in sleeves
        ],
    )
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (True, "kill_switch"))
    monkeypatch.setattr(bot_engine, "_dependency_breakers", lambda _state: _AllowBreakers())
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: dict(_POSITIONS))
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "_run_reconciliation_if_due", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine,
        "_pre_rank_execution_candidates",
        lambda symbols, runtime=None: list(symbols),
    )
    monkeypatch.setattr(bot_engine, "_tca_stale_block_reason", lambda _now: None)
    monkeypatch.setattr(bot_engine, "_write_decision_record", lambda record, path: None)
    bot_engine._run_netting_cycle(bot_engine.BotState(), runtime, "loop", 0.0)
    assert captured, "the cycle produced no proposals"
    return captured


def _last_price(symbol: str) -> float:
    return float(_bars(symbol)["close"].iloc[-2])  # the forming bar is excluded


@pytest.mark.parametrize("cfg_updates", [
    {**_LIVE, "paper_sampling_enabled": True},
    {**_PAPER, "paper_sampling_enabled": False},
])
def test_without_paper_sampling_no_sleeve_can_increase_exposure(
    monkeypatch: pytest.MonkeyPatch, cfg_updates: dict[str, Any]
) -> None:
    captured = _run_cycle(
        monkeypatch,
        cfg_updates=cfg_updates,
        sleeves=(("day", "5Min"), ("swing", "15Min")),
    )

    assert {proposal.sleeve for _symbol, proposal, _net in captured} == {"day", "swing"}
    for symbol, proposal, _net in captured:
        assert proposal.blocked is True
        assert proposal.reason_code == "NO_QUALIFIED_STRATEGY"
        assert proposal.debug["rule_paper_sampling"] is False
        assert proposal.debug["evidence_partition"] == "no_qualified_strategy_shadow"
        assert proposal.debug["research_only"] is True
        assert proposal.debug["promotion_eligible"] is False
        position = _POSITIONS.get(symbol, 0.0)
        if position == 0.0:
            assert proposal.target_dollars == 0.0
            continue
        # target = current shares x a recent close: the position is held.
        implied_price = proposal.target_dollars / position
        closes = _bars(symbol)["close"]
        assert float(closes.min()) <= implied_price <= float(closes.max())


def test_without_paper_sampling_day_targets_hold_current_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _run_cycle(monkeypatch, cfg_updates={**_PAPER, "paper_sampling_enabled": False})

    for symbol, proposal, net in captured:
        expected = _POSITIONS.get(symbol, 0.0) * _last_price(symbol)
        assert proposal.target_dollars == pytest.approx(expected, rel=1e-12, abs=1e-9)
        assert net.target_dollars == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_paper_sampling_keeps_rule_scorer_decisions_and_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _run_cycle(monkeypatch, cfg_updates={**_PAPER, "paper_sampling_enabled": True})

    # Captured from the retired stale-model diagnostic path on these bars.
    expected = {
        "AAPL": (-4167.59750977, False, None),
        "MSFT": (-4939.463884071, False, None),
        "AMZN": (-633.299175543, True, "COST_GATE"),
        "NVDA": (-10000.0, False, None),
        "SPY": (0.0, True, "COST_GATE"),
    }
    observed = {
        symbol: (proposal.target_dollars, proposal.blocked, proposal.reason_code)
        for symbol, proposal, _net in captured
    }
    assert set(observed) == set(expected)
    for symbol, (target, blocked, reason) in expected.items():
        assert observed[symbol][0] == pytest.approx(target, rel=1e-9, abs=1e-6)
        assert observed[symbol][1:] == (blocked, reason)
    for _symbol, proposal, _net in captured:
        assert proposal.debug["rule_paper_sampling"] is True
        assert proposal.debug["evidence_partition"] == "rule_paper_sampling"
        assert proposal.debug["research_only"] is True
        assert proposal.debug["promotion_eligible"] is False
        assert proposal.debug["model_authority"] is False
        assert proposal.debug["runtime_authority"] is False
