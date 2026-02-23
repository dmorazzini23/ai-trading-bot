from __future__ import annotations

import logging

from ai_trading.config.management import TradingConfig, config_snapshot_hash
from ai_trading.main import _log_config_effective_summary


def test_config_snapshot_hash_is_stable_for_identical_config() -> None:
    env = {
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
        "APP_ENV": "test",
        "EXECUTION_MODE": "sim",
        "CAPITAL_CAP": "0.25",
    }
    first = TradingConfig.from_env(env)
    second = TradingConfig.from_env(dict(reversed(list(env.items()))))

    first_hash = config_snapshot_hash(first)
    second_hash = config_snapshot_hash(second)
    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_config_snapshot_hash_changes_when_effective_config_changes() -> None:
    baseline = TradingConfig.from_env(
        {
            "MAX_DRAWDOWN_THRESHOLD": "0.15",
            "CAPITAL_CAP": "0.25",
        }
    )
    changed = TradingConfig.from_env(
        {
            "MAX_DRAWDOWN_THRESHOLD": "0.15",
            "CAPITAL_CAP": "0.30",
        }
    )

    assert config_snapshot_hash(baseline) != config_snapshot_hash(changed)


def test_log_config_effective_summary_emits_hash(caplog) -> None:
    caplog.set_level(logging.INFO, logger="ai_trading.main")
    cfg = TradingConfig.from_env(
        {
            "MAX_DRAWDOWN_THRESHOLD": "0.15",
            "CAPITAL_CAP": "0.25",
        }
    )

    _log_config_effective_summary(cfg)

    records = [record for record in caplog.records if record.msg == "CONFIG_EFFECTIVE_SUMMARY"]
    assert records
    summary = records[0]
    assert isinstance(getattr(summary, "config_snapshot_hash", None), str)
    assert len(summary.config_snapshot_hash) == 64
