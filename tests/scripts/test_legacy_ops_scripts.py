from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_health_check() -> ModuleType:
    return _load_script_module("legacy_health_check_under_test", "scripts/health_check.py")


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        _repo_root() / relative_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_health_check_exits_nonzero_on_critical(monkeypatch) -> None:
    health_check = _load_health_check()
    monkeypatch.setattr(
        health_check,
        "get_health_status",
        lambda: {"overall_status": "critical", "checks": {}},
    )

    assert health_check.main() == 1


def test_legacy_health_check_exits_zero_when_not_critical(monkeypatch) -> None:
    health_check = _load_health_check()
    monkeypatch.setattr(
        health_check,
        "get_health_status",
        lambda: {"overall_status": "warning", "checks": {}},
    )

    assert health_check.main() == 0


def test_legacy_health_check_labels_itself_as_noncanonical() -> None:
    health_check = _load_health_check()

    assert "legacy local diagnostic" in health_check.LEGACY_HEALTH_SCRIPT_NOTICE
    assert "/healthz" in health_check.LEGACY_HEALTH_SCRIPT_NOTICE


def test_ab_pacing_arm_defaults_to_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    ab_pacing = _load_script_module("ab_pacing_compare_under_test", "scripts/ab_pacing_compare.py")
    env_file = tmp_path / ".env"
    original = "EXECUTION_MAX_NEW_ORDERS_PER_CYCLE=2\n"
    env_file.write_text(original, encoding="utf-8")
    restarts: list[str] = []
    monkeypatch.setattr(ab_pacing, "_restart_service", lambda service: restarts.append(service))

    result = ab_pacing._cmd_arm(
        SimpleNamespace(
            env_file=str(env_file),
            state_dir=str(tmp_path / "state"),
            service="ai-trading.service",
            arm="baseline",
            max_new_orders=4,
            no_restart=False,
            apply=False,
        )
    )

    assert result == 0
    assert env_file.read_text(encoding="utf-8") == original
    assert not (tmp_path / "state" / "baseline_start_utc.txt").exists()
    assert restarts == []
    assert '"status": "dry_run"' in capsys.readouterr().out


def test_ab_pacing_arm_apply_writes_env_stamp_and_restarts(monkeypatch, tmp_path: Path) -> None:
    ab_pacing = _load_script_module("ab_pacing_compare_apply_under_test", "scripts/ab_pacing_compare.py")
    env_file = tmp_path / ".env"
    env_file.write_text("EXECUTION_MAX_NEW_ORDERS_PER_CYCLE=2\n", encoding="utf-8")
    restarts: list[str] = []
    monkeypatch.setattr(ab_pacing, "_restart_service", lambda service: restarts.append(service))

    result = ab_pacing._cmd_arm(
        SimpleNamespace(
            env_file=str(env_file),
            state_dir=str(tmp_path / "state"),
            service="ai-trading.service",
            arm="variant",
            max_new_orders=5,
            no_restart=False,
            apply=True,
        )
    )

    assert result == 0
    assert env_file.read_text(encoding="utf-8") == "EXECUTION_MAX_NEW_ORDERS_PER_CYCLE=5\n"
    assert (tmp_path / "state" / "variant_start_utc.txt").exists()
    assert restarts == ["ai-trading.service"]
    assert list(tmp_path.glob(".env.bak.*"))


def test_guarded_legacy_scripts_seed_only_canonical_env_names() -> None:
    for relative_path in (
        "scripts/demo_enhanced_debugging.py",
        "scripts/final_validation.py",
        "scripts/validate_critical_fix.py",
        "scripts/validate_standalone.py",
    ):
        text = (_repo_root() / relative_path).read_text(encoding="utf-8")

        assert "ALPACA_BASE_URL" not in text
        assert "FLASK_PORT" not in text
        assert "ALPACA_TRADING_BASE_URL" in text
        assert "API_PORT" in text


def test_runtime_prune_script_covers_retention_planner_artifacts() -> None:
    script = Path("scripts/prune_runtime_jsonl.sh").read_text(encoding="utf-8")

    for filename in (
        "decision_records.jsonl",
        "config_snapshots.jsonl",
        "gate_effectiveness.jsonl",
        "ml_shadow_predictions.jsonl",
        "order_events.jsonl",
        "fill_events.jsonl",
        "tca_records.jsonl",
        "oms_events.jsonl",
        "memory_samples.jsonl",
    ):
        assert filename in script


def test_debug_cli_does_not_seed_dummy_runtime_env() -> None:
    script = Path("scripts/debug_cli.py").read_text(encoding="utf-8")

    assert "os.environ.setdefault" not in script
    assert "cli_key" not in script
    assert "cli_secret" not in script


def test_runtime_artifacts_reset_defaults_to_dry_run(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "AI_TRADING_DATA_DIR": str(tmp_path),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/runtime_artifacts_reset.sh", "--skip-cycle"],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "dry run: no files will be changed" in proc.stdout
    assert not (tmp_path / "runtime").exists()


def test_rollout_advanced_gates_defaults_to_dry_run(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    original = "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_ENABLED=0\n"
    env_file.write_text(original, encoding="utf-8")

    proc = subprocess.run(
        [
            "bash",
            "scripts/rollout_advanced_gates.sh",
            "geometric",
            "--env-file",
            str(env_file),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Dry run for stage 'geometric'" in proc.stdout
    assert "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_ENABLED=1" in proc.stdout
    assert env_file.read_text(encoding="utf-8") == original
