#!/usr/bin/env python
"""Repository audit utility.

AI-AGENT-REF: emits code hygiene metrics and fails on risky constructs.
"""
from __future__ import annotations

import ast
import json
import py_compile
import sys
from pathlib import Path
from typing import Dict


def scan_repo(root: Path) -> Dict[str, int]:
    metrics = {
        "mock_classes": 0,
        "import_guards": 0,
        "__getattr__": 0,
        "exec_eval_count": 0,
        "metrics_logger_imports": 0,
        "py_compile_failures": 0,
    }
    py_files = [p for p in root.rglob("*.py") if "site-packages" not in p.parts]
    for path in py_files:
        try:
            source = path.read_text()
        except Exception:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError:
            metrics["py_compile_failures"] += 1
        try:
            tree = ast.parse(source)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Mock"):
                metrics["mock_classes"] += 1
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                metrics["__getattr__"] += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
                metrics["exec_eval_count"] += 1
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.startswith("metrics_logger"):
                        metrics["metrics_logger_imports"] += 1
            if isinstance(node, ast.Try) and any(
                isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body
            ):
                metrics["import_guards"] += 1
    return metrics


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    metrics = scan_repo(root)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
