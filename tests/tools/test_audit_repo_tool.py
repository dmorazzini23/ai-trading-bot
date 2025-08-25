import json
import subprocess
import sys
from pathlib import Path


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

