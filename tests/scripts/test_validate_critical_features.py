from __future__ import annotations

import ast
import shlex
import subprocess
import sys
from pathlib import Path


def test_money_math_check_runs_from_script_working_directory():
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_critical_features.py"
    module = ast.parse(script.read_text(encoding="utf-8"))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    checks = next(
        ast.literal_eval(node.value)
        for node in main.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "checks" for target in node.targets
        )
    )
    command = next(command for command, description in checks if description == "Money math determinism")
    argv = shlex.split(command)
    argv[0] = sys.executable

    result = subprocess.run(argv, cwd=script.parent, capture_output=True, text=True, timeout=15, check=False)

    assert result.returncode == 0, result.stderr
