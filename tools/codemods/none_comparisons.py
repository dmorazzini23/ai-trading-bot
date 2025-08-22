"""Codemod to rewrite None comparisons using identity checks."""

from __future__ import annotations

import pathlib

import libcst as cst
import libcst.matchers as m

# AI-AGENT-REF: replace equality with identity for None comparisons


class _NoneCompareTransformer(cst.CSTTransformer):
    def leave_Comparison(self, original: cst.Comparison, updated: cst.Comparison) -> cst.Comparison:
        targets: list[cst.ComparisonTarget] = []
        changed = False
        for target in updated.comparisons:
            op, comp = target.operator, target.comparator
            if isinstance(op, cst.Equal) and m.matches(comp, m.Name("None")):
                op = cst.Is()
                changed = True
            elif isinstance(op, cst.NotEqual) and m.matches(comp, m.Name("None")):
                op = cst.IsNot()
                changed = True
            targets.append(target.with_changes(operator=op, comparator=comp))
        if changed:
            return updated.with_changes(comparisons=targets)
        return updated


def _should_skip(path: pathlib.Path) -> bool:
    parts = path.parts
    if "artifacts" in parts:
        return True
    if "tests" in parts and "fixtures" in parts:
        return True
    return False


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    for py in repo.rglob("*.py"):
        if _should_skip(py):
            continue
        src = py.read_text()
        module = cst.parse_module(src)
        updated = module.visit(_NoneCompareTransformer())
        if module.code != updated.code:
            py.write_text(updated.code)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

