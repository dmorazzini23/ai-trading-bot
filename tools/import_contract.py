from __future__ import annotations

import ast
import logging
import pathlib
import sys

# AI-AGENT-REF: enforce repo-wide import style
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = "ai_trading"
# Explicitly forbid legacy top-level names that caused regressions
DENY_BARE = {
    "data_client",
    "utils",
    "helpers",
    "config",     # legacy shim removed
    "sentiment",  # legacy shim removed
    "settings",
    "models",
    "engine",
    "telemetry",
}


def main() -> int:
    bad: list[tuple[str, int, str]] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        p = str(path)
        if any(
            s in p
            for s in (
                "/.venv/",
                "/venv/",
                "/.git/",
                "/build/",
                "/dist/",
                "/tests/",
                "/scripts/",
            )
        ):
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src, filename=p)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith(PKG) or node.level > 0:
                    continue
                head = mod.split(".", 1)[0]
                if head in DENY_BARE:
                    bad.append((p, node.lineno, f"Suspicious bare import '{mod}'. Use '{PKG}.' prefix."))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.startswith(PKG):
                        continue
                    if mod.startswith("."):
                        bad.append((p, node.lineno, f"Top-level import must be absolute: {mod}"))
                        continue
                    head = mod.split(".", 1)[0]
                    if head in DENY_BARE:
                        bad.append((p, node.lineno, f"Suspicious bare import '{mod}'. Use '{PKG}.' prefix."))
    if bad:
        for f, ln, msg in bad:
            logger.error("%s:%s: %s", f, ln, msg)
        return 1
    logger.info("IMPORT CONTRACT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
