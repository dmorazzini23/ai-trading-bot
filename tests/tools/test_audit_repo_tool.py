import json
import subprocess
import sys
from pathlib import Path

import tools.audit_repo as audit_repo


def test_audit_repo_runs_clean():
    """AI-AGENT-REF: ensure audit script emits zero risky counts."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "audit_repo.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(result.stdout.strip())
    assert metrics["exec_eval_count"] == 0
    assert metrics["py_compile_failures"] == 0


def test_scan_repo_skips_sensitive_checks(tmp_path):
    """SAFE_PREFIXES bypass compilation and exec/eval metrics."""
    safe_prefix = ("tools", "ci")
    assert safe_prefix in audit_repo.SAFE_PREFIXES
    safe_dir = tmp_path.joinpath(*safe_prefix)
    safe_dir.mkdir(parents=True)

    safe_file = safe_dir / "uses_exec.py"
    safe_file.write_text("return 1\nexec('danger')\n")

    regular_file = tmp_path / "regular.py"
    regular_file.write_text("return 1\nexec('ok')\n")

    metrics = audit_repo.scan_repo(tmp_path)

    assert metrics["py_compile_failures"] == 1
    assert metrics["exec_eval_count"] == 1
    assert not list(tmp_path.rglob("*.pyc"))


def test_scan_repo_reports_zero_exec_eval_for_repo_root():
    """Direct scan of the repository should report zero exec/eval usage."""
    repo_root = Path(__file__).resolve().parents[2]
    metrics = audit_repo.scan_repo(repo_root)
    assert metrics["exec_eval_count"] == 0
