#!/usr/bin/env python3
# ruff: noqa
"""AST-based auditor for broad `except Exception:`.

Usage:
  python tools/audit_exceptions.py --paths ai_trading [more/paths] [--fail-over N]

Outputs JSON summary to stdout and a human-readable table.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import pathlib
import sys
import textwrap

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

def find_broad_handlers(path: pathlib.Path) -> list[dict]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    hits: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            # match `except Exception:` (not bare except)
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                line = node.lineno
                src = "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[line-1:line+1])
                hits.append({"file": str(path), "line": line, "snippet": src})
    return hits

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--fail-over", type=int, default=None, help="Exit nonzero if total > N")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        pp = pathlib.Path(p)
        if pp.is_dir():
            files.extend(pp.rglob("*.py"))
        elif pp.is_file():
            files.append(pp)

    all_hits = []
    for f in files:
        all_hits.extend(find_broad_handlers(f))

    report = {
        "total": len(all_hits),
        "by_file": {},
    }
    for h in all_hits:
        report["by_file"].setdefault(h["file"], []).append(h)

    sys.stdout.write(
        json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n"
    )
    logging.warning("Top offenders (first 10):")
    for i, h in enumerate(all_hits[:10], 1):
        logging.warning(
            "%2d. %s:%s\n%s",
            i,
            h["file"],
            h["line"],
            textwrap.indent(h["snippet"], "    "),
        )

    if args.fail_over is not None and len(all_hits) > args.fail_over:
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())

