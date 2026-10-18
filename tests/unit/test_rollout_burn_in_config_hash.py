"""Burn-in config identity must survive the runtime mutations made at the open."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.config.management import TradingConfig, config_snapshot_hash
from ai_trading.core import bot_engine
from ai_trading.core import runtime as core_runtime
from ai_trading.position_sizing import _CACHE as POSITION_SIZING_CACHE

_ENV = {
    "MAX_DRAWDOWN_THRESHOLD": "0.15",
    "EXECUTION_MODE": "paper",
    "AI_TRADING_CAPITAL_CAP": "0.04",
    "MAX_POSITION_MODE": "AUTO",
    "AI_TRADING_SIGNAL_MAX_POSITION_SIZE": "3000",
}


@pytest.fixture
def auto_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve AUTO sizing from a fixed equity instead of the broker."""

    monkeypatch.setattr(
        core_runtime,
        "resolve_max_position_size",
        lambda *_args, **_kwargs: (3266.0, {"mode": "AUTO", "source": "alpaca"}),
    )
    monkeypatch.setattr(POSITION_SIZING_CACHE, "value", POSITION_SIZING_CACHE.value)
    monkeypatch.setattr(POSITION_SIZING_CACHE, "ts", POSITION_SIZING_CACHE.ts)
    monkeypatch.setattr(bot_engine, "MAX_POSITION_SIZE", bot_engine.MAX_POSITION_SIZE)


def _rebuilt_cfg() -> TradingConfig:
    """Return a cfg after a runtime rebuild plus an in-place ``_values`` write.

    ``build_runtime`` no longer writes the resolved size into cfg, but other
    runtime code still writes ``_values`` directly; the hash must ignore both.
    """

    cfg = TradingConfig.from_env(_ENV)
    runtime = core_runtime.build_runtime(cfg)
    assert runtime.params["MAX_POSITION_SIZE"] == 3266.0
    cfg._values["max_position_size"] = 3266.0
    return cfg


def test_config_hash_unchanged_by_runtime_rebuild(auto_sizing: None) -> None:
    fresh = TradingConfig.from_env(_ENV)
    expected = config_snapshot_hash(fresh)

    rebuilt = _rebuilt_cfg()

    assert config_snapshot_hash(rebuilt) == expected
    assert config_snapshot_hash(copy.deepcopy(rebuilt)) == expected
    # A cache-hit cycle hashes a fresh, unmutated cfg; it must agree.
    assert config_snapshot_hash(TradingConfig.from_env(_ENV)) == expected


def test_config_hash_tracks_explicit_update_not_runtime_value(auto_sizing: None) -> None:
    rebuilt = _rebuilt_cfg()
    baseline = config_snapshot_hash(rebuilt)

    updated = rebuilt.update(capital_cap=0.05)

    assert config_snapshot_hash(updated) != baseline
    # The runtime-resolved size must not leak into the derived config.
    assert config_snapshot_hash(updated) == config_snapshot_hash(
        TradingConfig.from_env(_ENV).update(capital_cap=0.05)
    )


def _run_rollout_cycle(cfg: TradingConfig, now: datetime) -> dict:
    return bot_engine._update_rollout_governance_state(
        state=bot_engine.BotState(),
        cfg=cfg,
        effective_policy=SimpleNamespace(policy_hash="policy-a"),
        slo_derisk_details={},
        now=now,
    )


def test_burn_in_does_not_reset_between_first_and_steady_cycles(
    auto_sizing: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "rollout_state.json"
    monkeypatch.setattr(bot_engine, "_resolve_rollout_state_path", lambda: state_path)
    now = datetime(2026, 10, 16, 13, 30, tzinfo=UTC)

    # First cycle after the open rebuilds the runtime; later cycles reuse it
    # and see a fresh cfg. A restart with unchanged settings does the same.
    _run_rollout_cycle(_rebuilt_cfg(), now)
    _run_rollout_cycle(TradingConfig.from_env(_ENV), now)
    _run_rollout_cycle(_rebuilt_cfg(), now)

    persisted = json.loads(state_path.read_text())
    assert persisted["burn_in_reset_count"] == 0
    assert persisted["burn_in_paper_cycles"] == 3


def _raise_rollout_env() -> dict:
    raise OSError("rollout state unavailable")


@pytest.mark.parametrize("mode", ["live", "shadow"])
def test_rollout_failure_fails_closed_outside_paper(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(bot_engine, "_rollout_env_snapshot", _raise_rollout_env)
    state = bot_engine.BotState()

    summary = bot_engine._update_rollout_governance_state(
        state=state,
        cfg=SimpleNamespace(execution_mode=mode),
        effective_policy=SimpleNamespace(policy_hash="policy-a"),
        slo_derisk_details={},
        now=datetime(2026, 10, 16, 13, 30, tzinfo=UTC),
    )

    assert state.burn_in_ready is False
    assert state.burn_in_block_reason == "ROLLOUT_GOVERNANCE_UNAVAILABLE"
    assert state.capital_ramp_multiplier == pytest.approx(0.05)
    assert summary["burn_in_ready"] is False
    assert summary["capital_ramp"]["multiplier"] == pytest.approx(0.05)
    assert summary["capital_ramp"]["advance_eligible"] is False


@pytest.mark.parametrize("mode", ["paper", "sim"])
def test_rollout_failure_stays_open_in_paper(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(bot_engine, "_rollout_env_snapshot", _raise_rollout_env)
    state = bot_engine.BotState()

    summary = bot_engine._update_rollout_governance_state(
        state=state,
        cfg=SimpleNamespace(execution_mode=mode),
        effective_policy=SimpleNamespace(policy_hash="policy-a"),
        slo_derisk_details={},
        now=datetime(2026, 10, 16, 13, 30, tzinfo=UTC),
    )

    assert state.burn_in_ready is True
    assert state.capital_ramp_multiplier == pytest.approx(1.0)
    assert summary["burn_in_ready"] is True
