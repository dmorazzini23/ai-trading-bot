"""AST-based auditor for broad exception handlers.

Usage:
  python tools/audit_exceptions.py --paths ai_trading [more/paths] [--fail-over N]

Outputs JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections.abc import Iterable

_BROAD_TYPES = frozenset({"Exception", "BaseException", "bare"})


def _handler_type(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "bare"
    if isinstance(handler.type, ast.Name):
        return handler.type.id
    try:
        return ast.unparse(handler.type)
    except Exception:
        return "unknown"


def _body_style(handler: ast.ExceptHandler) -> str:
    if any(isinstance(stmt, ast.Pass) for stmt in handler.body):
        return "pass"
    if not handler.body:
        return "empty"
    first = handler.body[0]
    if isinstance(first, ast.Return):
        if isinstance(first.value, ast.Constant):
            return "return_constant"
        return "return"
    if isinstance(first, ast.Raise):
        return "raise"
    if isinstance(first, ast.Assign):
        return "assign"
    if isinstance(first, ast.Expr):
        return "expr"
    return type(first).__name__.lower()


def _iter_files(paths: Iterable[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        elif path.is_file():
            files.append(path)
    return files


def find_broad_handlers(path: pathlib.Path, suppress_suffixes: tuple[str, ...]) -> list[dict[str, object]]:
    path_posix = path.as_posix()
    if any(path_posix.endswith(suffix) for suffix in suppress_suffixes):
        return []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    lines = source.splitlines(keepends=True)
    hits: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        handler_type = _handler_type(node)
        if handler_type not in _BROAD_TYPES:
            continue
        line = node.lineno
        snippet = "".join(lines[line - 1 : line + 1])
        body_style = _body_style(node)
        hits.append(
            {
                "file": str(path),
                "line": line,
                "type": handler_type,
                "body_style": body_style,
                "silent": body_style in {"pass", "return_constant", "return"},
                "snippet": snippet,
            }
        )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument(
        "--suppress-path-suffix",
        nargs="*",
        default=(),
        help="Optional path suffixes to suppress during audit.",
    )
    parser.add_argument("--fail-over", type=int, default=None, help="Exit nonzero if total > N")
    args = parser.parse_args()

    files = _iter_files(args.paths)
    suppressed = tuple(str(value) for value in args.suppress_path_suffix if str(value).strip())

    all_hits: list[dict[str, object]] = []
    for file_path in files:
        all_hits.extend(find_broad_handlers(file_path, suppressed))

    report: dict[str, object] = {
        "total": len(all_hits),
        "silent_total": 0,
        "by_type": {},
        "by_file": {},
    }
    by_type = report["by_type"]
    by_file = report["by_file"]
    assert isinstance(by_type, dict)
    assert isinstance(by_file, dict)
    silent_total = 0

    for hit in all_hits:
        file_name = str(hit["file"])
        type_name = str(hit["type"])
        by_file.setdefault(file_name, []).append(hit)
        by_type[type_name] = int(by_type.get(type_name, 0)) + 1
        if bool(hit.get("silent")):
            silent_total += 1
    report["silent_total"] = silent_total

    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    if args.fail_over is not None and len(all_hits) > args.fail_over:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
