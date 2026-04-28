from __future__ import annotations
# ruff: noqa

import json
import subprocess
import sys

def test_auditor_runs():
    p = subprocess.run([sys.executable, "tools/audit_exceptions.py", "--paths", "ai_trading"], capture_output=True, text=True)
    assert p.returncode == 0
    data = json.loads(p.stdout.splitlines()[0])
    assert "total" in data and isinstance(data["total"], int)


def test_auditor_reports_parse_errors(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")

    p = subprocess.run(
        [sys.executable, "tools/audit_exceptions.py", "--paths", str(bad_file)],
        capture_output=True,
        text=True,
    )

    assert p.returncode == 3
    data = json.loads(p.stdout.splitlines()[0])
    assert data["parse_error_total"] == 1
    assert data["parse_errors"][0]["file"] == str(bad_file)
    assert data["parse_errors"][0]["error"] == "SyntaxError"
