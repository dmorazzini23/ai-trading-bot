import argparse
import ast
import importlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import libcst as cst
from libcst import helpers as cst_helpers


@dataclass
class BrokenImport:
    file: Path
    lineno: int
    import_type: str  # 'import' or 'from'
    module: str
    names: List[str]
    alias: Optional[str] = None  # for 'import' statements


@dataclass
class Rewrite:
    file: Path
    old: str
    new: str


@dataclass
class Unresolved:
    file: Path
    lineno: int
    stmt: str


def build_symbol_index(pkg_root: Path) -> Dict[str, Set[str]]:
    """Build mapping of symbol name to modules exporting it."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for py in pkg_root.rglob("*.py"):
        if any(part in {"tests", "vendor", "__pycache__"} for part in py.parts):
            continue
        module = ".".join(py.with_suffix("").relative_to(pkg_root.parent).parts)
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        exports: Set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(node.name)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.add(elt.value)
        for name in exports:
            index[name].add(module)
    return index


def _gather_attr_uses(tree: ast.AST, alias: str) -> Set[str]:
    uses: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == alias:
            uses.add(node.attr)
    return uses


def find_broken_imports(test_path: Path) -> List[BrokenImport]:
    broken: List[BrokenImport] = []
    for py in test_path.rglob("*.py"):
        rel = py.relative_to(test_path)
        if rel.parts and rel.parts[0] in {"integration", "slow"}:
            continue
        try:
            source = py.read_text()
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("ai_trading"):
                    names = [a.name for a in node.names if a.name != "*"]
                    try:
                        mod = importlib.import_module(node.module)
                        missing = []
                        for n in names:
                            if hasattr(mod, n):
                                continue
                            try:
                                importlib.import_module(f"{node.module}.{n}")
                            except ImportError:
                                missing.append(n)
                        if missing:
                            broken.append(BrokenImport(py, node.lineno, "from", node.module, missing))
                    except ImportError:
                        broken.append(BrokenImport(py, node.lineno, "from", node.module, names))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("ai_trading"):
                        try:
                            importlib.import_module(alias.name)
                        except ImportError:
                            broken.append(BrokenImport(py, node.lineno, "import", alias.name, [], alias.asname or alias.name.split('.')[-1]))
    return broken


def _rewrite_import_from(file_path: Path, old_module: str, name: str, new_module: str) -> None:
    mod = cst.parse_module(file_path.read_text())

    class Transformer(cst.CSTTransformer):
        def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom):
            module_str = cst_helpers.get_full_name_for_node(original.module)
            if module_str == old_module and len(original.names) == 1:
                alias = original.names[0]
                if isinstance(alias, cst.ImportAlias) and cst_helpers.get_full_name_for_node(alias.name) == name:
                    return updated.with_changes(module=cst.parse_expression(new_module))
            return updated

    new_mod = mod.visit(Transformer())
    file_path.write_text(new_mod.code)


def _rewrite_import(file_path: Path, old_module: str, alias_name: str, new_module: str) -> None:
    mod = cst.parse_module(file_path.read_text())

    class Transformer(cst.CSTTransformer):
        def leave_Import(self, original: cst.Import, updated: cst.Import):
            new_names = []
            changed = False
            for import_alias in original.names:
                name_str = cst_helpers.get_full_name_for_node(import_alias.name)
                asname = import_alias.asname.name.value if import_alias.asname else None
                if name_str == old_module and asname == alias_name:
                    new_names.append(import_alias.with_changes(name=cst.parse_expression(new_module)))
                    changed = True
                else:
                    new_names.append(import_alias)
            if changed:
                return updated.with_changes(names=new_names)
            return updated

    new_mod = mod.visit(Transformer())
    file_path.write_text(new_mod.code)


def resolve_and_rewrite(index: Dict[str, Set[str]], broken: List[BrokenImport], write: bool = False) -> Tuple[List[Rewrite], List[Unresolved]]:
    rewrites: List[Rewrite] = []
    unresolved: List[Unresolved] = []

    for bi in broken:
        if bi.import_type == "from":
            for name in bi.names:
                candidates = index.get(name, set())
                if len(candidates) == 1:
                    new_module = next(iter(candidates))
                    old_stmt = f"from {bi.module} import {name}"
                    new_stmt = f"from {new_module} import {name}"
                    rewrites.append(Rewrite(bi.file, old_stmt, new_stmt))
                    if write:
                        _rewrite_import_from(bi.file, bi.module, name, new_module)
                else:
                    unresolved.append(Unresolved(bi.file, bi.lineno, f"from {bi.module} import {name}"))
        else:  # import
            assert bi.alias
            try:
                source = bi.file.read_text()
                tree = ast.parse(source)
            except Exception:
                unresolved.append(Unresolved(bi.file, bi.lineno, f"import {bi.module} as {bi.alias}"))
                continue
            attrs = _gather_attr_uses(tree, bi.alias)
            if not attrs:
                unresolved.append(Unresolved(bi.file, bi.lineno, f"import {bi.module} as {bi.alias}"))
                continue
            candidates: Optional[Set[str]] = None
            for attr in attrs:
                mods = index.get(attr, set())
                if len(mods) != 1:
                    candidates = None
                    break
                mods = set(mods)
                if candidates is None:
                    candidates = mods
                else:
                    candidates &= mods
            if candidates and len(candidates) == 1:
                new_module = next(iter(candidates))
                old_stmt = f"import {bi.module} as {bi.alias}"
                new_stmt = f"import {new_module} as {bi.alias}"
                rewrites.append(Rewrite(bi.file, old_stmt, new_stmt))
                if write:
                    _rewrite_import(bi.file, bi.module, bi.alias, new_module)
            else:
                unresolved.append(Unresolved(bi.file, bi.lineno, f"import {bi.module} as {bi.alias}"))
    return rewrites, unresolved


def write_report(report_path: Path, rewrites: List[Rewrite], unresolved: List[Unresolved]) -> None:
    lines = ["# Import Repair Report", ""]
    lines.append("## Rewritten")
    if rewrites:
        for r in rewrites:
            lines.append(f"- {r.file}: `{r.old}` → `{r.new}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Unresolved")
    if unresolved:
        for u in unresolved:
            lines.append(f"- {u.file}:{u.lineno}: `{u.stmt}`")
    else:
        lines.append("- None")
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair stale ai_trading imports in tests.")
    parser.add_argument("--pkg", required=True, help="Package root (e.g., ai_trading)")
    parser.add_argument("--tests", required=True, help="Path to tests directory")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed changes without writing")
    parser.add_argument("--write", action="store_true", help="Apply changes to files")
    parser.add_argument("--report", help="Path to write markdown report")
    args = parser.parse_args()

    pkg_root = Path(args.pkg).resolve()
    tests_root = Path(args.tests).resolve()

    index = build_symbol_index(pkg_root)
    broken = find_broken_imports(tests_root)
    rewrites, unresolved = resolve_and_rewrite(index, broken, write=args.write and not args.dry_run)

    for r in rewrites:
        print(f"REWRITE: {r.old} -> {r.new} ({r.file})")
    for u in unresolved:
        print(f"UNRESOLVED: {u.stmt} ({u.file}:{u.lineno})")

    if args.report:
        write_report(Path(args.report), rewrites, unresolved)


if __name__ == "__main__":
    main()
