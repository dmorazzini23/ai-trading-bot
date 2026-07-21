from __future__ import annotations

from pathlib import Path


SCRIPT = Path("scripts/agent_validate_changed.sh")


def test_agent_validate_changed_script_contains_required_modes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--targeted" in text
    assert "--full" in text
    assert "--market-hours" in text
    assert "--force-broad" in text
    assert "refusing full validation during market hours" in text


def test_agent_validate_changed_script_runs_core_validation_commands() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "./venv/bin/pytest -q" in text
    assert "./venv/bin/ruff check" in text
    assert "./venv/bin/mypy" in text
    assert "scripts/typecheck_strict.sh" in text
    assert "py_compile" in text


def test_agent_validate_changed_script_keeps_runtime_smoke_checks() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "http://127.0.0.1:9001/healthz" in text
    assert "tool_runtime_incident_snapshot" in text
    assert "isinstance(payload.get('should_alert'), bool)" in text
    assert "payload.get('should_alert') is False" not in text
    assert "--skip-runtime-smoke" in text


def test_agent_validate_changed_script_verifies_existing_systemd_units_only() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SYSTEMD_VERIFY_FILES" in text
    assert 'path_exists "$file"' in text
    assert 'systemd-analyze verify "${SYSTEMD_VERIFY_FILES[@]}"' in text


def test_agent_validate_changed_script_keeps_danger_pattern_guards() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "except (Exception|BaseException)" in text
    assert "direct os environment access" in text
    assert "pytz reference" in text
    assert "legacy Alpaca SDK reference" in text
    assert "subprocess shell=True" in text
