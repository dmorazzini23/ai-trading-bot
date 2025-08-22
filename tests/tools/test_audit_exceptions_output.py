import json
import subprocess
import sys


def test_audit_exceptions_first_line_is_json():
    p = subprocess.run(
        [sys.executable, "tools/audit_exceptions.py", "--paths", "ai_trading"],
        check=False, capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    first = p.stdout.splitlines()[0]
    data = json.loads(first)
    assert isinstance(data, dict)

