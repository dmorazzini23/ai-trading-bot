from pathlib import Path
from types import SimpleNamespace

from ai_trading.core.netting_execution_runtime import _prepare_oms_ledger
from ai_trading.oms.ledger import OrderLedger


def test_prepare_oms_ledger_rebuilds_when_runtime_path_changes(tmp_path, monkeypatch):
    old_path = tmp_path / "old_ledger.jsonl"
    new_path = tmp_path / "new_ledger.jsonl"
    old_ledger = OrderLedger(str(old_path), 1.0)
    setattr(old_ledger, "_configured_lookback_hours", 1.0)
    state = SimpleNamespace(_oms_ledger=old_ledger)
    cfg = SimpleNamespace(
        execution_mode="live",
        ledger_enabled=True,
        ledger_path=str(new_path),
        ledger_lookback_hours=24.0,
    )
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "0")

    ledger = _prepare_oms_ledger(state, cfg)

    assert ledger is not None
    assert ledger is not old_ledger
    assert getattr(ledger, "_path", Path()) == new_path
    assert getattr(ledger, "_configured_lookback_hours", None) == 24.0
    assert getattr(state, "_oms_ledger") is ledger


def test_prepare_oms_ledger_rebuilds_when_cached_lookback_metadata_is_invalid(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "ledger.jsonl"
    old_ledger = OrderLedger(str(ledger_path), 1.0)
    setattr(old_ledger, "_configured_lookback_hours", "bad-metadata")
    state = SimpleNamespace(_oms_ledger=old_ledger)
    cfg = SimpleNamespace(
        execution_mode="live",
        ledger_enabled=True,
        ledger_path=str(ledger_path),
        ledger_lookback_hours=24.0,
    )
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "0")

    ledger = _prepare_oms_ledger(state, cfg)

    assert ledger is not None
    assert ledger is not old_ledger
    assert getattr(ledger, "_configured_lookback_hours", None) == 24.0
    assert getattr(state, "_oms_ledger") is ledger
