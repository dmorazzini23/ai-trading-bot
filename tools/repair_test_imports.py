#!/usr/bin/env python3
"""
Scan tests for stale internal imports and rewrite them safely using libcst.
Focus: known breakages like `ai_trading.position.core` → `ai_trading.position`.
Usage:
  python tools/repair_test_imports.py --pkg ai_trading --tests tests --dry-run
  python tools/repair_test_imports.py --pkg ai_trading --tests tests --write --report artifacts/import-repair-report.md
"""
# AI-AGENT-REF: AST/CST based import repair utility
from __future__ import annotations
import argparse, pathlib, sys, importlib.util
import libcst as cst
import libcst.matchers as m
import libcst.helpers as cst_helpers
from typing import Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # AI-AGENT-REF: ensure project importable

# Static, hand-curated rewrites for known legacy paths that moved to a public API.
# These are applied before any dynamic package scans.
STATIC_REWRITES: Dict[str, str] = {
    # Position package cleanup: import from the public surface
    "ai_trading.position.core": "ai_trading.position",
    "ai_trading.position.core.MarketRegime": "ai_trading.position.MarketRegime",
}

def _module_exists(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

class Rewriter(cst.CSTTransformer):
    def __init__(self, changes: dict[str, str], log: list[str]) -> None:
        self.changes = changes
        self.log = log

    def leave_Import(self, node: cst.Import, updated: cst.Import) -> cst.Import:
        names = []
        for alias in updated.names:
            full = cst_helpers.get_full_name_for_node(alias.name)
            new = self.changes.get(full)
            if new:
                self.log.append(f"Import: {full} -> {new}")
                names.append(alias.with_changes(name=cst.parse_expression(new)))
            else:
                names.append(alias)
        return updated.with_changes(names=tuple(names))

    def leave_ImportFrom(self, node: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if updated.module is None:
            return updated
        full_mod = cst_helpers.get_full_name_for_node(updated.module)
        module_new = self.changes.get(full_mod)
        names = []
        for alias in updated.names:
            name_str = cst_helpers.get_full_name_for_node(alias.name)
            fq = f"{full_mod}.{name_str}"
            new_fq = self.changes.get(fq)
            if new_fq:
                new_mod, new_name = new_fq.rsplit(".", 1)
                if module_new is None:
                    module_new = new_mod
                self.log.append(f"ImportFrom: {fq} -> {new_fq}")
                names.append(alias.with_changes(name=cst.parse_expression(new_name)))
            else:
                names.append(alias)
        if module_new:
            return updated.with_changes(module=cst.parse_expression(module_new), names=tuple(names))
        return updated.with_changes(names=tuple(names))

def rewrite_file(path: pathlib.Path, mapping: dict[str, str]) -> tuple[bool, list[str], str]:
    src = path.read_text(encoding="utf-8")
    mod = cst.parse_module(src)
    log: list[str] = []
    updated = mod.visit(Rewriter(mapping, log))
    changed = updated.code != src
    return changed, log, updated.code

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="ai_trading")
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args(argv)

    # Only keep mappings that point to real, resolvable modules
    def _target_exists(target: str) -> bool:
        mod = target.rsplit(".", 1)[0]
        return _module_exists(mod)

    mapping = {k: v for k, v in STATIC_REWRITES.items() if _target_exists(v)}
    test_root = pathlib.Path(args.tests)
    report_lines = []
    changed_any = False

    for py in test_root.rglob("*.py"):
        changed, log, new_code = rewrite_file(py, mapping)
        if log:
            report_lines.append(f"### {py}\n" + "\n".join(f"- {line}" for line in log) + "\n")
        if changed:
            changed_any = True
            if args.write and not args.dry_run:
                py.write_text(new_code, encoding="utf-8")

    if args.report:
        out = pathlib.Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        hdr = "# Import repair report\n\n" if report_lines else "# Import repair report\n\n_No changes needed._\n"
        out.write_text(hdr + "\n".join(report_lines), encoding="utf-8")

    print("Done. Changes applied:", changed_any)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

