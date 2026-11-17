from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.meta_learning import core


pd = pytest.importorskip("pandas")




def test_meta_learning_conversion_and_synthetic_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    core._import_pandas()
    audit_df = pd.DataFrame(
        [
            [
                "550e8400-e29b-41d4-a716-446655440000",
                "2026-01-01T14:30:00Z",
                "SPY",
                "buy",
                "10",
                "100.00",
                "live",
                "filled",
            ],
            [
                "550e8400-e29b-41d4-a716-446655440001",
                "2026-01-01T15:30:00Z",
                "SPY",
                "sell",
                "10",
                "102.00",
                "live",
                "filled",
            ],
            [
                "550e8400-e29b-41d4-a716-446655440002",
                "2026-01-01T15:31:00Z",
                "SPY",
                "buy",
                "10",
                "0",
                "live",
                "rejected",
            ],
        ]
    )

    converted = core._convert_audit_to_meta_format(audit_df)

    assert list(converted["symbol"]) == ["SPY"]
    assert converted.iloc[0]["exit_price"] == 102.0
    assert converted.iloc[0]["reward"] == 20.0
    assert converted.iloc[0]["signal_tags"].startswith("audit_order_")

    monkeypatch.delenv("AI_TRADING_META_LEARNING_ALLOW_SYNTHETIC_BOOTSTRAP", raising=False)
    filtered, removed = core._exclude_synthetic_training_rows(
        pd.DataFrame(
            [
                {"signal_tags": "real+momentum", "strategy": "live", "classification": ""},
                {"signal_tags": "synthetic_bootstrap_data", "strategy": "live", "classification": ""},
                {"signal_tags": "real", "strategy": "bootstrap_generated", "classification": ""},
            ]
        )
    )

    assert removed == 2
    assert filtered["signal_tags"].tolist() == ["real+momentum"]


