"""Utility to rename obviously unused local variables.

AI-AGENT-REF: conservative unused local renamer.
"""
from __future__ import annotations
import ast
from pathlib import Path
ROOTS = ['ai_trading', 'trade_execution']
EXCLUDE_DIRS = {'tests', '.venv', 'venv', 'build', 'dist', '__pycache__'}

class UnusedLocalRenamer(ast.NodeTransformer):

    def visit_FunctionDef(self, node: ast.FunctionDef):
        assigned: set[str] = set()
        reads: set[str] = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    for name in _names_in_target(target):
                        assigned.add(name)
            elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                reads.add(inner.id)
        unused = {n for n in assigned if n not in reads and (not n.startswith('_'))}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store) and (inner.id in unused):
                inner.id = f'_unused_{inner.id}'
        return node

def _names_in_target(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    names: list[str] = []
    for child in ast.iter_child_nodes(target):
        names.extend(_names_in_target(child))
    return names

def main() -> None:
    for root in ROOTS:
        for path in Path(root).rglob('*.py'):
            if any((part in EXCLUDE_DIRS for part in path.parts)):
                continue
            source = path.read_text(encoding='utf-8')
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            new_tree = UnusedLocalRenamer().visit(tree)
            ast.fix_missing_locations(new_tree)
if __name__ == '__main__':
    main()