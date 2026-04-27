from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.meta_learning import core


pd = pytest.importorskip("pandas")


def test_validate_trade_data_quality_detects_mixed_valid_rows(tmp_path: Path) -> None:
    trade_log = tmp_path / "trades.csv"
    trade_log.write_text(
        "\n".join(
            [
                "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward",
                "SPY,2026-01-01T14:30:00Z,100.00,2026-01-01T15:00:00Z,101.50,5,buy,mean,core,tag_a,0.8,7.5",
                "550e8400-e29b-41d4-a716-446655440000,2026-01-01T16:00:00Z,QQQ,buy,3,300.25,live,filled",
                "BAD,not,a,recognized,row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = core.validate_trade_data_quality(trade_log)

    assert report["file_exists"] is True
    assert report["file_readable"] is True
    assert report["has_valid_format"] is True
    assert report["mixed_format_detected"] is True
    assert report["audit_format_rows"] == 1
    assert report["meta_format_rows"] == 1
    assert report["valid_price_rows"] == 2
    assert core.has_mixed_format(trade_log) is True


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

    assert list(converted["symbol"]) == ["SPY", "SPY"]
    assert converted.iloc[0]["exit_price"] == 102.0
    assert converted.iloc[0]["reward"] == 20.0
    assert converted.iloc[1]["signal_tags"].startswith("audit_order_")

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


def test_signal_weight_and_optimizer_fallback_paths() -> None:
    assert core.normalize_score("2.0", cap=1.2) == 1.2
    assert core.normalize_score("bad") == 0.0
    assert core.adjust_confidence(0.8, 2.0) == 0.4
    assert core.update_signal_weights({"a": 0.5, "b": 0.5}, {"a": 3.0, "b": 1.0}) == {
        "a": 0.75,
        "b": 0.25,
    }
    assert core.update_signal_weights({"a": 1.0}, {"a": 0.0}) == {"a": 1.0}

    class BadModel:
        def predict(self, _data):
            raise ValueError("bad model")

    original = [1.5, -2.0]
    assert core.optimize_signals(original, model=BadModel()) is original

    class GoodModel:
        def predict(self, _data):
            return [2.0, -2.0, 0.5]

    assert core.optimize_signals([0.0], model=GoodModel(), volatility=2.0) == [2.0, -2.0, 0.5]
