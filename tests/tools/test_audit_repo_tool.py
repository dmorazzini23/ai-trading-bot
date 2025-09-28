import json
import subprocess
import sys
from pathlib import Path

from tools import audit_repo


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


def test_scan_repo_skips_sensitive_checks(tmp_path, monkeypatch):
    """SAFE_PREFIXES should bypass py_compile and exec/eval metrics."""
    safe_prefix = ("tools", "ci")
    assert safe_prefix in audit_repo.SAFE_PREFIXES
    safe_dir = tmp_path.joinpath(*safe_prefix)
    safe_dir.mkdir(parents=True)

    safe_file = safe_dir / "uses_exec.py"
    safe_file.write_text("exec('danger')\n")

    regular_file = tmp_path / "regular.py"
    regular_file.write_text("exec('ok')\n")

    compiled_paths: list[Path] = []

    def fake_compile(filename: str, *args, **kwargs) -> None:
        compiled_paths.append(Path(filename))

    monkeypatch.setattr(audit_repo.py_compile, "compile", fake_compile)

    metrics = audit_repo.scan_repo(tmp_path)

    compiled_paths = [p.resolve() for p in compiled_paths]
    assert regular_file.resolve() in compiled_paths
    assert safe_file.resolve() not in compiled_paths
    assert metrics["exec_eval_count"] == 1
    assert metrics["py_compile_failures"] == 0


def test_scan_repo_reports_zero_exec_eval_for_repo_root():
    """Direct scan of the repository should report zero exec/eval usage."""
    repo_root = Path(__file__).resolve().parents[2]
    metrics = audit_repo.scan_repo(repo_root)
    assert metrics["exec_eval_count"] == 0

