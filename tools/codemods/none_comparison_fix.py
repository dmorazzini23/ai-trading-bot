"""Rewrite equality comparisons with None using libcst."""
from __future__ import annotations
from pathlib import Path
import libcst as cst
EXCLUDE = {'tests', '.venv', 'venv', 'build', 'dist', '__pycache__'}

class _FixNone(cst.CSTTransformer):
    """Convert ``== None`` and ``!= None`` to identity checks."""

    def leave_Comparison(self, original: cst.Compare, updated: cst.Compare) -> cst.BaseExpression:
        if len(updated.comparisons) != 1:
            return updated
        comp = updated.comparisons[0]
        left = updated.left
        op = comp.operator
        right = comp.comparator
        if isinstance(op, cst.Equal) and isinstance(right, cst.Name) and (right.value == 'None'):
            return updated.with_changes(comparisons=[comp.with_changes(operator=cst.Is())])
        if isinstance(op, cst.NotEqual) and isinstance(right, cst.Name) and (right.value == 'None'):
            return updated.with_changes(comparisons=[comp.with_changes(operator=cst.IsNot())])
        if isinstance(left, cst.Name) and left.value == 'None':
            if isinstance(op, cst.Equal):
                return cst.Compare(left=right, comparisons=[cst.Comparison(operator=cst.Is(), comparator=cst.Name('None'))])
            if isinstance(op, cst.NotEqual):
                return cst.Compare(left=right, comparisons=[cst.Comparison(operator=cst.IsNot(), comparator=cst.Name('None'))])
        return updated

def _process(path: Path) -> None:
    module = cst.parse_module(path.read_text())
    updated = module.visit(_FixNone())
    if updated.code != module.code:
        path.write_text(updated.code)

def main() -> None:
    for path in Path('.').rglob('*.py'):
        if any((part in EXCLUDE for part in path.parts)):
            continue
        _process(path)
if __name__ == '__main__':
    main()