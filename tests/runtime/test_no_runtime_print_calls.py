from __future__ import annotations

import ast
from pathlib import Path


def _print_call_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            lines.append(node.lineno)
    return lines


def test_no_print_calls_in_runtime_package() -> None:
    offenders: dict[str, list[int]] = {}
    for path in Path("ai_trading").rglob("*.py"):
        lines = _print_call_lines(path)
        if lines:
            offenders[path.as_posix()] = lines
    assert offenders == {}, f"print() calls found in runtime package: {offenders}"
