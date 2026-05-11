from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_trading.tools import pretrade_risk_control_verifier as verifier


@dataclass(frozen=True)
class _RuntimeCfg:
    kill_switch: bool = False
    kill_switch_path: str = ""
    max_position_size: float = 3000.0
    daily_loss_limit: float = 0.05
    ledger_enabled: bool = True


@dataclass(frozen=True)
class _LaunchProfile:
    name: str = "paper_trade"
    max_notional_per_order: float | None = None
    max_symbol_exposure: float | None = 0.35
    max_daily_loss: float | None = None


def _passing_controls() -> dict[str, dict[str, object]]:
    return {
        control: {"enabled": True, "fail_closed": True, "status": "passed"}
        for control in verifier.DEFAULT_REQUIRED_CONTROLS
    }


def test_pretrade_verifier_fails_closed_for_missing_and_unknown_controls() -> None:
    controls = _passing_controls()
    controls.pop("kill_switch")
    controls["mystery_override"] = {"enabled": True, "fail_closed": True, "status": "passed"}

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config={"controls": controls},
    )

    assert payload["status"] == "fail_closed"
    assert payload["fail_closed"] is True
    assert {item["kind"] for item in payload["violations"]} >= {
        "missing_required_control",
        "unknown_control",
    }


def test_pretrade_verifier_requires_intent_control_evidence() -> None:
    controls = _passing_controls()
    intents = [
        {
            "intent_id": "intent-1",
            "controls": {
                "kill_switch": "passed",
                "buying_power": "passed",
            },
        }
    ]

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config={"controls": controls},
        intents=intents,
    )

    assert payload["status"] == "fail_closed"
    assert "intent_required_control_missing" in {item["kind"] for item in payload["violations"]}


def test_pretrade_verifier_cli_writes_artifacts_and_returns_failure(tmp_path: Path) -> None:
    controls_path = tmp_path / "controls.json"
    output_path = tmp_path / "pretrade.json"
    latest_path = tmp_path / "latest.json"
    controls_path.write_text(json.dumps({"controls": {"kill_switch": {"status": "unknown"}}}), encoding="utf-8")

    rc = verifier.main(
        [
            "--report-date",
            "2026-05-05",
            "--controls-json",
            str(controls_path),
            "--output-json",
            str(output_path),
            "--latest-json",
            str(latest_path),
        ]
    )

    assert rc == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "fail_closed"
    assert latest_path.is_file()


def test_pretrade_verifier_auto_detects_runtime_controls(monkeypatch) -> None:
    env_values = {
        "MAX_ORDER_DOLLARS": 15_000.0,
        "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER": 250.0,
    }

    def fake_get_env(key: str, default: Any = None, *, cast: Any = None, **_: Any) -> Any:
        value = env_values.get(key, default)
        return cast(value) if cast is not None and value is not None else value

    monkeypatch.setattr(verifier, "get_trading_config", lambda: _RuntimeCfg())
    monkeypatch.setattr(verifier, "resolve_launch_profile", lambda: _LaunchProfile())
    monkeypatch.setattr(verifier, "get_env", fake_get_env)

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config=None,
    )

    assert payload["status"] == "passed"
    assert payload["summary"]["runtime_controls_auto_detected"] is True
    assert payload["summary"]["configured_controls"] == len(verifier.DEFAULT_REQUIRED_CONTROLS)
    assert payload["control_source"] == "runtime_config"
    assert payload["control_statuses"]["max_order_notional"]["status"] == "passed"
    assert payload["promotion_authority"] is False
    assert payload["live_money_authority"] is False


def test_pretrade_verifier_auto_detection_fails_closed_when_limits_missing(monkeypatch) -> None:
    def fake_get_env(key: str, default: Any = None, *, cast: Any = None, **_: Any) -> Any:
        return default

    monkeypatch.setattr(
        verifier,
        "get_trading_config",
        lambda: _RuntimeCfg(max_position_size=0.0, daily_loss_limit=0.0),
    )
    monkeypatch.setattr(
        verifier,
        "resolve_launch_profile",
        lambda: _LaunchProfile(max_notional_per_order=None, max_symbol_exposure=0.0),
    )
    monkeypatch.setattr(verifier, "get_env", fake_get_env)

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config=None,
    )

    assert payload["status"] == "fail_closed"
    assert payload["summary"]["runtime_controls_auto_detected"] is True
    assert {
        (item["kind"], item["control"], item.get("status"))
        for item in payload["violations"]
    } >= {
        ("control_status_not_passed", "max_order_notional", "unknown"),
        ("control_status_not_passed", "max_daily_loss", "unknown"),
        ("control_status_not_passed", "max_position_notional", "unknown"),
    }
