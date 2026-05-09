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
    assert payload["status"] == "failed"
    assert payload["reason"] == "missing_fresh_replay_artifact"
    assert payload["replay"]["exists"] is False
    assert payload["fresh_artifact"] is False


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


def test_replay_governance_main_returns_nonzero_without_fresh_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool.bot_engine,
        "_run_replay_governance",
        lambda state, **_kwargs: setattr(state, "last_replay_run_date", _kwargs["now"].date()),
    )
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    exit_code = tool.main(
        [
            "--force",
            "--replay-output-dir",
            str(tmp_path / "replay_outputs"),
        ],
    )

    assert exit_code == 1


def test_collect_replay_snapshot_does_not_default_missing_counterfactual_to_passed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay_hash_20260505.json"
    path.write_text(
        json.dumps(
            {
                "ts": "2026-05-05T20:00:00Z",
                "rows": 1,
                "orders_submitted": 1,
                "fill_events": 1,
                "violations": [],
                "live_cost_alignment": {
                    "summary": {"alignment_counts": {"optimism": 1}},
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = tool._collect_replay_snapshot(path)

    assert snapshot["counterfactual_passed"] is False
    assert snapshot["counterfactual_available"] is False
    assert snapshot["live_cost_alignment"]["summary"]["alignment_counts"] == {
        "optimism": 1,
    }


def test_replay_governance_policy_regression_writes_blocked_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "summary.json"

    def _fake_run(state, **_kwargs) -> None:
        state.last_replay_run_date = _kwargs["now"].date()
        raise RuntimeError("REPLAY_POLICY_NON_REGRESSION_FAILED")

    monkeypatch.setattr(tool.bot_engine, "_run_replay_governance", _fake_run)
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    payload = tool.run_replay_governance(
        [
            "--force",
            "--replay-output-dir",
            str(tmp_path / "replay_outputs"),
            "--summary-json",
            str(summary_path),
        ],
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "REPLAY_POLICY_NON_REGRESSION_FAILED"
    assert payload["fresh_artifact"] is False
    assert payload["summary_path"] == str(summary_path)
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["status"] == "blocked"


def test_replay_governance_main_returns_two_for_policy_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool.bot_engine,
        "_run_replay_governance",
        lambda state, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("REPLAY_POLICY_NON_REGRESSION_FAILED")
        ),
    )
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    exit_code = tool.main(
        [
            "--force",
            "--replay-output-dir",
            str(tmp_path / "replay_outputs"),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ],
    )

    assert exit_code == 2


def test_replay_governance_blocks_no_baseline_counterfactual(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(state, *, now: datetime, **_kwargs) -> None:
        output_dir = tmp_path / "replay_outputs"
        output_dir.mkdir(parents=True)
        (output_dir / f"replay_hash_{now.strftime('%Y%m%d')}.json").write_text(
            json.dumps(
                {
                    "ts": now.isoformat(),
                    "rows": 12,
                    "orders_submitted": 8,
                    "fill_events": 7,
                    "violations": [],
                    "counterfactual": {"passed": True, "reason": "no_baseline_summary"},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        state.last_replay_run_date = now.date()

    monkeypatch.setattr(tool.bot_engine, "_run_replay_governance", _fake_run)
    monkeypatch.setattr(tool, "ensure_dotenv_loaded", lambda: None)

    payload = tool.run_replay_governance(
        ["--force", "--replay-output-dir", str(tmp_path / "replay_outputs")]
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "counterfactual_no_baseline"


def test_replay_governance_timer_treats_blocked_as_success() -> None:
    unit = Path("packaging/systemd/ai-trading-replay-governance.service").read_text(
        encoding="utf-8"
    )

    assert "SuccessExitStatus=2" in unit
