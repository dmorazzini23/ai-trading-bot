"""Codemod to replace equality checks against None.

AI-AGENT-REF: conservative None comparison fixer.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOTS = ["ai_trading", "trade_execution"]
EXCLUDE_DIRS = {"tests", ".venv", "venv", "build", "dist", "__pycache__"}

# Very conservative regexes; we only touch code under ROOTS and skip tests/.
EQ_NONE = re.compile(r"(?P<lhs>\S+)\s*==\s*None\b")
NEQ_NONE = re.compile(r"(?P<lhs>\S+)\s*!=\s*None\b")


def fix_text(text: str) -> str:
    """Return text with `== None` and `!= None` replaced safely."""
    text = EQ_NONE.sub(r"\g<lhs> is None", text)
    text = NEQ_NONE.sub(r"\g<lhs> is not None", text)
    return text


def main() -> None:
    for root in ROOTS:
        for path in Path(root).rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            if "tests" in path.parts:
                continue
            original = path.read_text(encoding="utf-8")
            fixed = fix_text(original)
            if fixed != original:
                path.write_text(fixed, encoding="utf-8")


if __name__ == "__main__":
    main()

