"""Codemod to replace equality checks against None.

AI-AGENT-REF: conservative None comparison fixer using libcst.
"""
from __future__ import annotations

from pathlib import Path

import libcst as cst

EXCLUDE_DIRS = {"tests", ".venv", "venv", "build", "dist", "__pycache__"}


class FixNoneComparisons(cst.CSTTransformer):
    """Rewrite ``== None`` and ``!= None`` to ``is None`` / ``is not None``."""

    def leave_Comparison(self, original: cst.Compare, updated: cst.Compare) -> cst.BaseExpression:
        if len(updated.comparisons) != 1:
            return updated
        comp = updated.comparisons[0]
        left = updated.left
        op = comp.operator
        right = comp.comparator
        if isinstance(op, cst.Equal) and isinstance(right, cst.Name) and right.value == "None":
            new_comp = comp.with_changes(operator=cst.Is())
            return updated.with_changes(comparisons=[new_comp])
        if isinstance(op, cst.NotEqual) and isinstance(right, cst.Name) and right.value == "None":
            new_comp = comp.with_changes(operator=cst.IsNot())
            return updated.with_changes(comparisons=[new_comp])
        if isinstance(left, cst.Name) and left.value == "None":
            if isinstance(op, cst.Equal) and isinstance(right, cst.BaseExpression):
                new_comp = cst.Comparison(operator=cst.Is(), comparator=cst.Name("None"))
                return cst.Compare(left=right, comparisons=[new_comp])
            if isinstance(op, cst.NotEqual) and isinstance(right, cst.BaseExpression):
                new_comp = cst.Comparison(operator=cst.IsNot(), comparator=cst.Name("None"))
                return cst.Compare(left=right, comparisons=[new_comp])
        return updated


def process_file(path: Path) -> None:
    module = cst.parse_module(path.read_text())
    updated = module.visit(FixNoneComparisons())
    if updated.code != module.code:
        path.write_text(updated.code)


def main() -> None:
    for path in Path(".").rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        process_file(path)


if __name__ == "__main__":
    main()
