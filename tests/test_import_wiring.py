from __future__ import annotations

import pathlib
import subprocess
import sys

# AI-AGENT-REF: validate import wiring scripts
ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        check=False, cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def test_import_contract() -> None:
    code, out = run([sys.executable, "tools/import_contract.py"])
    assert code == 0, f"Import contract failed:\n{out}"


def test_package_health() -> None:
    import pytest

    pytest.importorskip("psutil")
    code, out = run([sys.executable, "tools/package_health.py", "--strict"])
    assert code == 0, f"Package health failed:\n{out}"


def test_package_health_skips_async_when_anyio_missing(monkeypatch) -> None:
    import builtins
    import importlib

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anyio":
            raise ModuleNotFoundError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "tools.package_health", raising=False)
    ph = importlib.import_module("tools.package_health")
    assert ph._probe_async_testing() is True
