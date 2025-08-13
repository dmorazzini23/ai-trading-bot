from __future__ import annotations

import pathlib
import subprocess
import sys

# AI-AGENT-REF: validate import wiring scripts
ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def test_import_contract() -> None:
    code, out = run([sys.executable, "tools/import_contract.py"])
    assert code == 0, f"Import contract failed:\n{out}"


def test_package_health() -> None:
    code, out = run([sys.executable, "tools/package_health.py", "--strict"])
    assert code == 0, f"Package health failed:\n{out}"
