from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools import replay_governance as tool


def test_replay_governance_tool_invokes_engine_with_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(
        state,
        *,
        now: datetime,
        market_open_now: bool,
        force: bool = False,
    ) -> None:
        captured["force"] = bool(force)
        captured["market_open_now"] = bool(market_open_now)
        state.last_replay_run_date = now.date()

    monkeypatch.setattr(tool.bot_engine, "_run_replay_governance", _fake_run)
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    payload = tool.run_replay_governance(
        [
            "--force",
            "--replay-output-dir",
            str(tmp_path / "replay_outputs"),
        ]
    )

    assert captured["force"] is True
    assert captured["market_open_now"] is False
    assert payload["status"] == "ok"
    assert payload["replay"]["exists"] is False


def test_replay_governance_tool_writes_summary_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "summary.json"

    def _fake_run(
        state,
        *,
        now: datetime,
        market_open_now: bool,
        force: bool = False,
    ) -> None:
        _ = market_open_now, force
        output_dir_raw = str(
            get_env("AI_TRADING_REPLAY_OUTPUT_DIR", "runtime/replay_outputs", cast=str)
            or ""
        ).strip()
        output_dir = resolve_runtime_artifact_path(
            output_dir_raw or "runtime/replay_outputs",
            default_relative="runtime/replay_outputs",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": now.isoformat(),
            "rows": 12,
            "orders_submitted": 8,
            "fill_events": 7,
            "cap_adjustments_count": 3,
            "violations": [],
            "violations_by_code": {},
            "counterfactual": {"passed": True},
        }
        out_path = output_dir / f"replay_hash_{now.strftime('%Y%m%d')}.json"
        out_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        state.last_replay_run_date = now.date()

    monkeypatch.setattr(tool.bot_engine, "_run_replay_governance", _fake_run)
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    payload = tool.run_replay_governance(
        [
            "--force",
            "--replay-output-dir",
            str(tmp_path / "replay_outputs"),
            "--summary-json",
            str(summary_path),
            "--no-enforce-oms-gates",
            "--no-require-non-regression",
            "--clip-intents-to-caps",
        ]
    )

    assert payload["status"] == "ok"
    assert payload["replay"]["exists"] is True
    assert payload["replay"]["rows"] == 12
    assert payload["replay"]["cap_adjustments_count"] == 3
    assert payload["replay"]["violations_count"] == 0
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["replay"]["exists"] is True
    assert saved["replay"]["counterfactual_passed"] is True
