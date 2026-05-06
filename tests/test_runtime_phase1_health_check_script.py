from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _health_check_script() -> Path:
    return _repo_root() / "scripts" / "runtime_phase1_health_check.sh"


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _seed_runtime(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, dict[str, str]]:
    runtime_dir = tmp_path / "runtime"
    report_dir = runtime_dir / "research_reports"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    decision_file = runtime_dir / "decision_records.jsonl"
    gate_effectiveness_jsonl = runtime_dir / "gate_effectiveness.jsonl"
    gate_effectiveness_summary = runtime_dir / "gate_effectiveness_summary.json"
    report_file = report_dir / "after_hours_training_20260305.json"

    _write_jsonl(decision_file, rows)
    gate_effectiveness_jsonl.write_text(json.dumps({"gate": "OK_TRADE"}) + "\n", encoding="utf-8")
    gate_effectiveness_summary.write_text(json.dumps({"gate_totals": {"OK_TRADE": 1}}), encoding="utf-8")
    report_file.write_text(json.dumps({"ts": _now_utc_iso()}), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "N": "5000",
            "RUNTIME_DIR": str(runtime_dir),
            "REPORT_DIR": str(report_dir),
            "DECISION_FILE": str(decision_file),
            "GATE_EFFECTIVENESS_JSONL": str(gate_effectiveness_jsonl),
            "GATE_EFFECTIVENESS_SUMMARY": str(gate_effectiveness_summary),
            "SHADOW_PREDICTIONS_FILE": str(runtime_dir / "ml_shadow_predictions.jsonl"),
            "ENV_RUNTIME_FILE": str(tmp_path / "runtime" / "ai-trading-runtime.env"),
            "AI_TRADING_ML_SHADOW_ENABLED": "0",
            "RATE_ALERT_MIN_ROWS": "100",
            # Keep rate checks active in tests even when Python time is frozen.
            "DECISION_STALE_MAX_AGE_MINUTES": "2000000",
            # Avoid report-freshness flakes from clock skew between Python and shell date.
            "REPORT_MAX_AGE_MINUTES": "2000000",
            "AUTH_HALT_MAX_RATE": "0.35",
            "AUTH_BROKER_HALT_FORBIDDEN_MAX_RATE": "0.35",
            "OK_TRADE_MIN_RATE": "0.005",
            "CYCLE_DUPLICATE_INTENT_MAX_RATE": "0.70",
            "SKIP_HEALTHZ_PROBE": "1",
            "SKIP_SERVICE_LIVENESS": "1",
        }
    )
    return decision_file, env


def _rows_for_gate_rates(
    *,
    total: int,
    duplicate_rows: int,
    auth_forbidden_rows: int = 0,
    ok_rows: int,
) -> list[dict[str, object]]:
    now_iso = _now_utc_iso()
    rows: list[dict[str, object]] = []
    for idx in range(total):
        gates: list[str] = ["VOL_TARGET_SCALE"]
        if idx < auth_forbidden_rows:
            gates.append("AUTH_BROKER_HALT_FORBIDDEN")
        if idx < duplicate_rows:
            gates.append("CYCLE_DUPLICATE_INTENT")
        if idx < ok_rows:
            gates.append("OK_TRADE")
        rows.append({"bar_ts": now_iso, "symbol": f"SYM{idx % 5}", "gates": gates})
    return rows


def test_runtime_health_check_fails_on_duplicate_intent_spike(tmp_path: Path) -> None:
    _, env = _seed_runtime(
        tmp_path,
        _rows_for_gate_rates(total=160, duplicate_rows=130, ok_rows=12),
    )

    proc = subprocess.run(
        ["bash", str(_health_check_script())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "cycle_duplicate_intent_rate=" in combined
    assert "CYCLE_DUPLICATE_INTENT spike detected" in combined


def test_runtime_health_check_passes_when_duplicate_intent_rate_is_healthy(tmp_path: Path) -> None:
    _, env = _seed_runtime(
        tmp_path,
        _rows_for_gate_rates(total=160, duplicate_rows=20, ok_rows=16),
    )

    proc = subprocess.run(
        ["bash", str(_health_check_script())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cycle_duplicate_intent_rate=" in proc.stdout
    assert "OK: CYCLE_DUPLICATE_INTENT rate within threshold" in proc.stdout


def test_runtime_health_check_fails_on_auth_broker_halt_forbidden_spike(tmp_path: Path) -> None:
    _, env = _seed_runtime(
        tmp_path,
        _rows_for_gate_rates(total=160, duplicate_rows=20, auth_forbidden_rows=90, ok_rows=12),
    )

    proc = subprocess.run(
        ["bash", str(_health_check_script())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "auth_broker_halt_forbidden_rate=" in combined
    assert "AUTH_BROKER_HALT_FORBIDDEN spike detected" in combined


def test_runtime_health_check_probes_healthz_and_service_liveness(tmp_path: Path) -> None:
    _, env = _seed_runtime(
        tmp_path,
        _rows_for_gate_rates(total=160, duplicate_rows=20, ok_rows=16),
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    systemctl_log = tmp_path / "systemctl.log"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$CURL_LOG\"\n"
        "printf '%s\\n' '{\"service\":\"ai-trading\",\"ok\":true}'\n",
        encoding="utf-8",
    )
    (bin_dir / "systemctl").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$SYSTEMCTL_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    (bin_dir / "systemctl").chmod(0o755)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "SKIP_HEALTHZ_PROBE": "0",
            "SKIP_SERVICE_LIVENESS": "0",
            "HEALTHCHECK_URL": "http://127.0.0.1:9001/healthz",
            "HEALTHCHECK_SYSTEMD_UNIT": "ai-trading.service",
            "CURL_LOG": str(curl_log),
            "SYSTEMCTL_LOG": str(systemctl_log),
        }
    )

    proc = subprocess.run(
        ["bash", str(_health_check_script())],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK: service active: ai-trading.service" in proc.stdout
    assert "OK: /healthz probe healthy: http://127.0.0.1:9001/healthz" in proc.stdout
    assert "is-active\n--quiet\nai-trading.service" in systemctl_log.read_text(encoding="utf-8")
    assert "http://127.0.0.1:9001/healthz" in curl_log.read_text(encoding="utf-8")
