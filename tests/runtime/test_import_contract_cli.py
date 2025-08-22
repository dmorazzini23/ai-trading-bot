import os
import subprocess
import sys
from pathlib import Path


def _script_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "import_contract.py")


def test_import_contract_ok():
    cp = subprocess.run(
        [sys.executable, _script_path(), "--ci", "--timeout", "2", "--modules", "sys"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr


def test_import_contract_timeout_simulated(monkeypatch):
    env = os.environ.copy()
    env["IMPORT_CONTRACT_SIMULATE_HANG"] = "1"
    cp = subprocess.run(
        [sys.executable, _script_path(), "--ci", "--timeout", "0.1", "--modules", "sys"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    assert cp.returncode != 0
    assert "TIMEOUT" in (cp.stderr or "")

