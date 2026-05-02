from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType


def _load_gate_module() -> ModuleType:
    repo_dir = Path(__file__).resolve().parents[2]
    module_path = repo_dir / "scripts" / "pre_open_acceptance_gate.py"
    spec = importlib.util.spec_from_file_location(
        "pre_open_acceptance_gate_under_test",
        module_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _health_step(module: ModuleType, payload: Mapping[str, object]):
    return module.Step(
        name="healthz",
        status="pass",
        summary="health endpoint is healthy",
        details={"payload": dict(payload)},
    )


def _pass_step(module: ModuleType, name: str):
    return module.Step(name=name, status="pass", summary=f"{name} ok", details={})


def test_canonical_env_map_normalizes_runtime_quotes(tmp_path: Path) -> None:
    module = _load_gate_module()
    env_path = tmp_path / ".env.runtime"
    env_path.write_text(
        "\n".join(
            [
                'EMPTY_VALUE=""',
                'SYMBOLS="*"',
                'BUCKETS="tight=1,normal=2"',
                'JSONISH={"AAPL":{"buy":{"12":30.0}}}',
            ]
        ),
        encoding="utf-8",
    )

    assert module._canonical_env_map(env_path) == {
        "EMPTY_VALUE": "",
        "SYMBOLS": "*",
        "BUCKETS": "tight=1,normal=2",
        "JSONISH": '{"AAPL":{"buy":{"12":30.0}}}',
    }


def test_health_port_prefers_healthcheck_env(tmp_path: Path, monkeypatch) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("HEALTHCHECK_PORT", "18081")
    monkeypatch.setenv("API_PORT", "9001")
    (tmp_path / ".env.runtime").write_text("HEALTHCHECK_PORT=9001\n", encoding="utf-8")

    assert module._health_port_from_env(tmp_path) == 18081


def test_health_port_uses_packaged_api_port_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_gate_module()
    monkeypatch.delenv("HEALTHCHECK_PORT", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert module._health_port_from_env(tmp_path) == 9001


def test_health_port_falls_back_to_runtime_api_port(tmp_path: Path, monkeypatch) -> None:
    module = _load_gate_module()
    monkeypatch.delenv("HEALTHCHECK_PORT", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    (tmp_path / ".env.runtime").write_text("API_PORT=19001\n", encoding="utf-8")

    assert module._health_port_from_env(tmp_path) == 19001


def test_preopen_operator_drill_fails_when_flat_required_and_broker_not_flat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_gate_module()
    monkeypatch.delenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", raising=False)
    monkeypatch.delenv("AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS", raising=False)
    monkeypatch.delenv("AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY", raising=False)
    (tmp_path / ".env.runtime").write_text(
        "\n".join(
            [
                "AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START=1",
                "AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS=1",
                "AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY=1",
                "AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS=AAPL,MSFT",
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "broker": {"open_orders_count": 1, "positions_count": 2},
        "attention_flags": [
            "market_closed_open_orders",
            "market_closed_non_flat_positions",
        ],
        "readiness_failures": ["oms_invariants_failed"],
        "readiness_gates": {
            "oms_invariants": {"required": True, "status": "required_failed"},
            "oms_lifecycle_parity": {"required": True, "status": "ok"},
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "fail"
    assert "flat_start_required=true" in step.summary
    assert step.details["flat_start"]["blockers"] == [
        "preopen_open_orders",
        "preopen_non_flat_positions",
    ]
    assert "oms_invariants_failed" in step.details["oms"]["blockers"]
    assert (
        step.details["config"]["AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS"]["raw"]
        == "AAPL,MSFT"
    )


def test_preopen_operator_drill_warns_on_nonflat_when_flat_start_observe_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "0")
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 1},
        "attention_flags": ["market_closed_non_flat_positions"],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {"required": False, "status": "ok"},
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "warn"
    assert "preopen_non_flat_positions" in step.details["warnings"]
    assert step.details["flat_start"]["required"] is False


def test_preopen_operator_drill_warns_on_observed_oms_failure(tmp_path: Path) -> None:
    module = _load_gate_module()
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 0},
        "attention_flags": ["oms_lifecycle_parity_failed"],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {
                "required": False,
                "status": "observed_failure",
                "detail": "missing_submit_ack",
            },
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "warn"
    assert "oms_lifecycle_parity_observed_failure" in step.details["oms"]["warnings"]
    assert "oms_state=oms_lifecycle_parity_observed_failure" in step.summary


def test_preopen_operator_drill_passes_when_required_checks_are_clean(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY", "1")
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 0},
        "attention_flags": [],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": True, "status": "ok"},
            "oms_lifecycle_parity": {"required": True, "status": "ok"},
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "pass"
    assert step.details["blockers"] == []
    assert "flat_state=clean" in step.summary
    assert "oms_state=clean" in step.summary


def test_preopen_operator_drill_trusts_runtime_flat_start_expected_swing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "1")
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 1},
        "attention_flags": ["market_closed_non_flat_positions"],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {"required": False, "status": "ok"},
        },
        "preopen_readiness": {
            "flat_start": {
                "enabled": True,
                "flat": True,
                "reason": "ok",
                "open_orders_count": 0,
                "unexpected_positions_count": 0,
                "expected_positions_count": 1,
                "expected_positions": [{"symbol": "AAPL", "qty": 5.0}],
            },
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "pass"
    assert step.details["flat_start"]["blockers"] == []
    assert step.details["flat_start"]["runtime_context"]["expected_positions_count"] == 1
    assert "flat_state=clean" in step.summary


def test_preopen_operator_drill_falls_back_when_runtime_flat_start_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "1")
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 1},
        "attention_flags": [],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {"required": False, "status": "ok"},
        },
        "preopen_readiness": {
            "flat_start": {
                "enabled": False,
                "reason": "disabled",
            },
        },
    }

    step = module._check_preopen_operator_drill(
        tmp_path,
        health_step=_health_step(module, payload),
    )

    assert step.status == "fail"
    assert step.details["flat_start"]["blockers"] == ["preopen_non_flat_positions"]


def test_preopen_operator_drill_warns_without_health_payload(tmp_path: Path) -> None:
    module = _load_gate_module()
    health_step = module.Step(
        name="healthz",
        status="warn",
        summary="health check skipped by flag",
        details={},
    )

    step = module._check_preopen_operator_drill(tmp_path, health_step=health_step)

    assert step.status == "warn"
    assert "health payload unavailable" in step.summary
    assert step.details["health_step_status"] == "warn"


def test_main_json_includes_preopen_operator_drill_and_fails_on_blocker(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "1")
    payload = {
        "broker": {"open_orders_count": 1, "positions_count": 0},
        "attention_flags": [],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {"required": False, "status": "ok"},
        },
    }
    monkeypatch.setattr(module, "_check_env_sync", lambda repo_dir, *, sync: _pass_step(module, "env_sync"))
    monkeypatch.setattr(
        module,
        "_refresh_runtime_reports",
        lambda repo_dir, *, refresh: _pass_step(module, "refresh_runtime_reports"),
    )
    monkeypatch.setattr(
        module,
        "_evaluate_runtime_gonogo",
        lambda: _pass_step(module, "runtime_gonogo"),
    )
    monkeypatch.setattr(
        module,
        "_check_health",
        lambda repo_dir, *, port, timeout_seconds, skip: _health_step(module, payload),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pre_open_acceptance_gate.py", "--repo-dir", str(tmp_path), "--json"],
    )

    exit_code = module.main()
    report = json.loads(capsys.readouterr().out)
    steps = {step["name"]: step for step in report["steps"]}

    assert exit_code == 1
    assert report["summary"]["fail"] == 1
    assert "preopen_operator_drill" in steps
    assert steps["preopen_operator_drill"]["status"] == "fail"
    assert steps["preopen_operator_drill"]["details"]["blockers"] == ["preopen_open_orders"]


def test_main_json_returns_warning_exit_when_preopen_drill_warns_only(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_gate_module()
    monkeypatch.setenv("AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START", "0")
    payload = {
        "broker": {"open_orders_count": 0, "positions_count": 1},
        "attention_flags": [],
        "readiness_failures": [],
        "readiness_gates": {
            "oms_invariants": {"required": False, "status": "ok"},
            "oms_lifecycle_parity": {"required": False, "status": "ok"},
        },
    }
    monkeypatch.setattr(module, "_check_env_sync", lambda repo_dir, *, sync: _pass_step(module, "env_sync"))
    monkeypatch.setattr(
        module,
        "_refresh_runtime_reports",
        lambda repo_dir, *, refresh: _pass_step(module, "refresh_runtime_reports"),
    )
    monkeypatch.setattr(
        module,
        "_evaluate_runtime_gonogo",
        lambda: _pass_step(module, "runtime_gonogo"),
    )
    monkeypatch.setattr(
        module,
        "_check_health",
        lambda repo_dir, *, port, timeout_seconds, skip: _health_step(module, payload),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pre_open_acceptance_gate.py", "--repo-dir", str(tmp_path), "--json"],
    )

    exit_code = module.main()
    report = json.loads(capsys.readouterr().out)
    steps = {step["name"]: step for step in report["steps"]}

    assert exit_code == 2
    assert report["summary"]["warn"] == 1
    assert report["summary"]["fail"] == 0
    assert steps["preopen_operator_drill"]["status"] == "warn"
    assert steps["preopen_operator_drill"]["details"]["warnings"] == [
        "preopen_non_flat_positions"
    ]
