from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from ai_trading import main


def test_collect_live_kpi_snapshot_marks_zero_sample_metrics_as_insufficient(
    monkeypatch,
) -> None:
    class _FakeSLOMonitor:
        def get_slo_status(self, _metric_name: str | None = None):
            return {"current_value": 0.0, "sample_count": 0}

    import ai_trading.monitoring.slo as slo_mod

    monkeypatch.setattr(slo_mod, "get_slo_monitor", lambda: _FakeSLOMonitor())
    monkeypatch.setitem(
        main.sys.modules,
        "ai_trading.core.bot_engine",
        types.SimpleNamespace(_current_drawdown=lambda: 0.0),
    )

    live_kpis, diagnostics = main._collect_live_kpi_snapshot()

    assert float(live_kpis["execution_drift_bps"]) == 0.0
    assert diagnostics["execution_drift_bps"] is None
    assert "execution_drift_bps" in diagnostics["insufficient_data_metrics"]


def test_live_kpi_guard_evaluates_and_can_trigger_rollback_alert(monkeypatch) -> None:
    main._LAST_PROMOTION_KPI_GUARD_TS = 0.0
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_GUARD_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_GUARD_INTERVAL_SEC", "0")
    monkeypatch.setenv("AI_TRADING_PROMOTION_RUNTIME_STRATEGIES", "ml_edge")

    class _FakeSLOMonitor:
        def get_slo_status(self, metric_name: str | None = None):
            if metric_name == "order_reject_rate_pct":
                return {"current_value": 6.0, "sample_count": 9}
            if metric_name == "execution_drift_bps":
                return {"current_value": 42.0, "sample_count": 9}
            return {}

    import ai_trading.monitoring.slo as slo_mod

    monkeypatch.setattr(slo_mod, "get_slo_monitor", lambda: _FakeSLOMonitor())
    monkeypatch.setitem(
        main.sys.modules,
        "ai_trading.core.bot_engine",
        types.SimpleNamespace(_current_drawdown=lambda: 0.12),
    )

    eval_calls: list[dict[str, object]] = []

    class _FakePromotionManager:
        registry = types.SimpleNamespace(model_index={})

        def evaluate_live_kpis_and_maybe_rollback(
            self,
            *,
            strategy: str,
            live_kpis: dict[str, float],
            force: bool = True,
        ) -> dict[str, object]:
            eval_calls.append(
                {"strategy": strategy, "live_kpis": dict(live_kpis), "force": force}
            )
            return {"strategy": strategy, "breached": True, "triggered": True}

    import ai_trading.governance.promotion as promotion_mod

    monkeypatch.setattr(
        promotion_mod,
        "get_promotion_manager",
        lambda: _FakePromotionManager(),
    )

    emitted: list[str] = []
    monkeypatch.setattr(
        main,
        "emit_runtime_alert",
        lambda event, **_kwargs: emitted.append(str(event)),
    )

    main._maybe_evaluate_live_kpi_control_band_rollbacks(cycle_index=7)

    assert len(eval_calls) == 1
    assert eval_calls[0]["strategy"] == "ml_edge"
    live_kpis = eval_calls[0]["live_kpis"]
    assert isinstance(live_kpis, dict)
    assert float(live_kpis["max_drawdown"]) == pytest.approx(0.12)
    assert float(live_kpis["reject_rate"]) == pytest.approx(0.06)
    assert float(live_kpis["execution_drift_bps"]) == pytest.approx(42.0)
    assert "ALERT_PROMOTION_CONTROL_BAND_ROLLBACK_TRIGGERED" in emitted


def test_bad_session_replay_auto_builds_from_tca_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    main._LAST_BAD_SESSION_REPLAY_TS = 0.0

    tca_path = tmp_path / "tca_records.jsonl"
    tca_row = {
        "ts": "2026-03-02T23:42:00Z",
        "symbol": "AAPL",
        "fill_price": 191.75,
        "qty": 10,
    }
    tca_path.write_text(json.dumps(tca_row) + "\n", encoding="utf-8")
    output_dir = tmp_path / "replay_bad_session"

    monkeypatch.setenv("AI_TRADING_BAD_SESSION_REPLAY_ON_CRITICAL_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BAD_SESSION_REPLAY_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_BAD_SESSION_REPLAY_SOURCE_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_BAD_SESSION_REPLAY_OUTPUT_DIR", str(output_dir))

    main._maybe_build_bad_session_replay_dataset(trigger="unit_test")

    assert (output_dir / "replay_manifest.json").exists()
    assert (output_dir / "AAPL.csv").exists()
    assert (output_dir / "incident_intents.jsonl").exists()
    assert (output_dir / "incident_broker.jsonl").exists()
