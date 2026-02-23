from __future__ import annotations

import ast
from pathlib import Path


def test_simulate_market_execution_avoids_builtin_hash() -> None:
    engine_path = (
        Path(__file__).resolve().parents[2] / "ai_trading" / "execution" / "engine.py"
    )
    source = engine_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(engine_path))

    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_simulate_market_execution":
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "hash":
                    violations.append(call.lineno)

    assert violations == [], (
        "builtin hash() is not allowed in deterministic simulation paths; "
        f"found in _simulate_market_execution at lines {violations}"
    )
