from __future__ import annotations

import logging
from types import SimpleNamespace

from ai_trading import main


def test_static_mode_nonpositive_is_autofixed(caplog):
    cfg = SimpleNamespace(
        trading_mode="balanced",
        alpaca_base_url="https://paper-api.alpaca.markets",
        paper=True,
        default_max_position_size=9000.0,
    )
    tcfg = SimpleNamespace(
        capital_cap=0.04,
        dollar_risk_limit=0.05,
        max_position_mode="STATIC",
        max_position_size=0.0,
    )

    with caplog.at_level(logging.INFO, logger="ai_trading.position_sizing"):
        main._validate_runtime_config(cfg, tcfg)

    assert getattr(tcfg, "max_position_size", 0.0) == 8000.0

    assert any(
        r.__dict__.get("field") == "max_position_size"
        and r.__dict__.get("reason") == "derived_equity_cap"
        for r in caplog.records
    )


def test_auto_mode_nonpositive_is_permitted(caplog):
    cfg = SimpleNamespace(
        trading_mode="balanced",
        alpaca_base_url="https://paper-api.alpaca.markets",
        paper=True,
    )
    tcfg = SimpleNamespace(
        capital_cap=0.04,
        dollar_risk_limit=0.05,
        max_position_mode="AUTO",
        max_position_size=0.0,
    )

    with caplog.at_level(logging.WARNING):
        main._validate_runtime_config(cfg, tcfg)

    assert not any(r.getMessage() == "CONFIG_AUTOFIX" for r in caplog.records)
