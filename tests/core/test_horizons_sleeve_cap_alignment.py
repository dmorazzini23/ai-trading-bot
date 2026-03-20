from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core.horizons import build_sleeve_configs


def _base_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        horizons=("day:1Min",),
        sleeve_day_enabled=True,
        sleeve_day_entry_threshold=0.01,
        sleeve_day_exit_threshold=0.1,
        sleeve_day_flip_threshold=0.35,
        sleeve_day_reentry_threshold=0.6,
        sleeve_day_deadband_dollars=1.0,
        sleeve_day_deadband_shares=1.0,
        sleeve_day_turnover_cap_dollars=100000.0,
        sleeve_day_cost_k=0.03,
        sleeve_day_edge_scale_bps=40.0,
        sleeve_day_max_symbol_dollars=9000.0,
        sleeve_day_max_gross_dollars=60000.0,
        global_max_symbol_dollars=3000.0,
    )


def test_build_sleeve_configs_aligns_to_global_cap_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_ENABLED", raising=False)
    monkeypatch.delenv("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_MULTIPLIER", raising=False)
    cfg = _base_cfg()

    sleeves = build_sleeve_configs(cfg=cfg)

    assert len(sleeves) == 1
    assert sleeves[0].max_symbol_dollars == 3000.0


def test_build_sleeve_configs_can_disable_cap_alignment(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_ENABLED", "0")
    cfg = _base_cfg()

    sleeves = build_sleeve_configs(cfg=cfg)

    assert len(sleeves) == 1
    assert sleeves[0].max_symbol_dollars == 9000.0


def test_build_sleeve_configs_applies_alignment_multiplier(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_MULTIPLIER", "1.5")
    cfg = _base_cfg()

    sleeves = build_sleeve_configs(cfg=cfg)

    assert len(sleeves) == 1
    assert sleeves[0].max_symbol_dollars == 4500.0
