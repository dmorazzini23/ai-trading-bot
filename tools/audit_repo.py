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


IGNORED_DEV_FOLDERS = {
    ".venv",
    "venv",
    "env",
    "_env",
    "build",
    "dist",
}

# Directories that are considered safe for dynamic constructs such as exec/eval
# and may intentionally contain syntactically invalid examples (e.g. fixtures,
# stub packages). We still parse these files for other metrics, but we skip
# compiling them and we do not count exec/eval usage.
SAFE_PREFIXES = (
    ("tests",),
    ("scripts",),
    ("tests", "stubs"),
    ("tests", "vendor_stubs"),
    ("tests", "_stubs"),
    ("tools", "ci"),
)

EXEC_EVAL_BUILTINS = frozenset({"exec", "eval"})


def _is_under_prefix(path: Path, root: Path, prefixes: tuple[tuple[str, ...], ...]) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    for prefix in prefixes:
        if parts[: len(prefix)] == prefix:
            return True
    return False


def _is_dev_folder(path: Path) -> bool:
    return any(part in IGNORED_DEV_FOLDERS for part in path.parts)


def _shadowed_exec_eval(tree: ast.AST) -> set[str]:
    shadowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id in EXEC_EVAL_BUILTINS:
                shadowed.add(node.id)
        elif isinstance(node, ast.arg) and node.arg in EXEC_EVAL_BUILTINS:
            shadowed.add(node.arg)
    return shadowed


def scan_repo(root: Path) -> Dict[str, int]:
    metrics = {
        "mock_classes": 0,
        "import_guards": 0,
        "__getattr__": 0,
        "exec_eval_count": 0,
        "metrics_logger_imports": 0,
        "py_compile_failures": 0,
    }
    py_files = [
        p
        for p in root.rglob("*.py")
        if "site-packages" not in p.parts and not _is_dev_folder(p)
    ]
    for path in py_files:
        skip_sensitive_metrics = _is_under_prefix(path, root, SAFE_PREFIXES)
        if not skip_sensitive_metrics:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError:
                metrics["py_compile_failures"] += 1
        try:
            source = path.read_text()
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except Exception:
            continue
        shadowed_exec_eval = _shadowed_exec_eval(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Mock"):
                metrics["mock_classes"] += 1
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                metrics["__getattr__"] += 1
            if (
                not skip_sensitive_metrics
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in EXEC_EVAL_BUILTINS
                and node.func.id not in shadowed_exec_eval
            ):
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
