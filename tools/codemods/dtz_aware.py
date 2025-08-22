"""Ensure datetime usage is timezone aware."""

from __future__ import annotations

import pathlib

import libcst as cst
import libcst.matchers as m
from libcst.helpers import get_full_name_for_node

# AI-AGENT-REF: replace naive datetime calls with utcnow helper


class _DTZTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.modified = False
        self.has_import = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module and get_full_name_for_node(node.module) == "ai_trading.utils.time":
            for alias in node.names:
                if isinstance(alias, cst.ImportAlias) and alias.name.value == "utcnow":
                    self.has_import = True

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        if m.matches(original.func, m.Attribute(value=m.Name("datetime"), attr=m.Name("utcnow"))) and not original.args:
            self.modified = True
            return cst.parse_expression("utcnow()")
        if m.matches(original.func, m.Attribute(value=m.Name("datetime"), attr=m.Name("now"))) and not original.args:
            self.modified = True
            return cst.parse_expression("utcnow()")
        return updated

    def leave_Module(self, original: cst.Module, updated: cst.Module) -> cst.Module:
        if self.modified and not self.has_import:
            import_stmt = cst.parse_statement("from ai_trading.utils.time import utcnow\n")
            return updated.with_changes(body=[import_stmt] + list(updated.body))
        return updated


def _iter_targets(root: pathlib.Path):
    for base in (root / "ai_trading", root / "trade_execution"):
        if base.exists():
            yield from base.rglob("*.py")


def _should_skip(path: pathlib.Path) -> bool:
    parts = path.parts
    return "tests" in parts or "artifacts" in parts


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    for py in _iter_targets(repo):
        if _should_skip(py):
            continue
        src = py.read_text()
        module = cst.parse_module(src)
        transformer = _DTZTransformer()
        updated = module.visit(transformer)
        if transformer.modified and module.code != updated.code:
            py.write_text(updated.code)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

