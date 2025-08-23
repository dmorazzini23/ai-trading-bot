#!/usr/bin/env python3
"""
Scan tests for stale internal imports and rewrite them safely using libcst.
Focus: known breakages like `ai_trading.position.core` → `ai_trading.position`.
Usage:
  python tools/repair_test_imports.py --pkg ai_trading --tests tests --dry-run
  python tools/repair_test_imports.py --pkg ai_trading --tests tests --write --report artifacts/import-repair-report.md
"""
# AI-AGENT-REF: AST/CST based import repair utility with reporting
from __future__ import annotations
import argparse, pathlib, sys, importlib.util, subprocess, os, re, difflib
from datetime import datetime, UTC
import libcst as cst
import libcst.helpers as cst_helpers
from typing import Dict, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # AI-AGENT-REF: ensure project importable

# Static, hand-curated rewrites for known legacy paths that moved to a public API.
# These are applied before any dynamic package scans. Use module:symbol notation
# to keep intent clear.
REWRITES_STATIC: Dict[str, str] = {
    # position
    "ai_trading.position.core:MarketRegime": "ai_trading.position:MarketRegime",
    "ai_trading.position.core:RegimeState": "ai_trading.position:RegimeState",
    "ai_trading.position.core": "ai_trading.position",

    # (leave room for future hand-curated entries; do not add shims)
}

def _expand_static_map(raw: Dict[str, str]) -> Dict[str, str]:
    """Convert colon-separated mapping to dotted form for CST rewrites."""
    out: Dict[str, str] = {}
    for k, v in raw.items():
        out[k.replace(":", ".")] = v.replace(":", ".")
    return out

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

def rewrite_file(path: pathlib.Path, mapping: dict[str, str]) -> tuple[bool, list[str], str, str]:
    src = path.read_text(encoding="utf-8")
    mod = cst.parse_module(src)
    log: list[str] = []
    updated = mod.visit(Rewriter(mapping, log))
    new_code = updated.code
    changed = new_code != src
    diff = ""
    if changed:
        diff = "\n".join(
            difflib.unified_diff(
                src.splitlines(),
                new_code.splitlines(),
                fromfile=str(path),
                tofile=str(path),
            )
        )
    return changed, log, new_code, diff


def collect_import_errors(test_root: pathlib.Path, pkg: str) -> tuple[list[str], str]:
    """Run pytest collection and parse out missing modules."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--collect-only",
        str(test_root),
        "-p",
        "xdist",
        "-p",
        "pytest_timeout",
        "-p",
        "pytest_asyncio",
    ]
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    output = proc.stdout + "\n" + proc.stderr
    missing: list[str] = []
    pattern = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
    for line in output.splitlines():
        m = pattern.search(line)
        if m:
            missing.append(m.group(1))
    return missing, output


def fill_block(template: str, start: str, end: str, lines: Iterable[str]) -> str:
    """Replace placeholder blocks in the report template."""
    start_tag = f"<!-- {start} -->"
    end_tag = f"<!-- {end} -->"
    body = "\n".join(lines) if lines else "*(none)*  "
    return re.sub(
        rf"{start_tag}.*?{end_tag}",
        f"{start_tag}\n{body}\n{end_tag}",
        template,
        flags=re.DOTALL,
    )

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="ai_trading")
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args(argv)

    mapping_all = _expand_static_map(REWRITES_STATIC)
    # Only keep mappings that point to real, resolvable modules
    def _target_exists(target: str) -> bool:
        mod = target.rsplit(".", 1)[0]
        return _module_exists(mod)

    mapping = {k: v for k, v in mapping_all.items() if _target_exists(v)}
    test_root = pathlib.Path(args.tests)

    rewrites_log: list[str] = []
    diffs: list[str] = []
    files_rewritten = 0

    for py in test_root.rglob("*.py"):
        changed, log, new_code, diff = rewrite_file(py, mapping)
        if log:
            rewrites_log.extend(log)
        if changed:
            files_rewritten += 1
            if args.write and not args.dry_run:
                py.write_text(new_code, encoding="utf-8")
            diffs.append(diff)

    missing, _ = collect_import_errors(test_root, args.pkg)
    internal_missing = sorted({m for m in missing if m.startswith(args.pkg)})
    third_party_missing = sorted({m for m in missing if not m.startswith(args.pkg)})

    static_fixed = len(rewrites_log)
    internal_unresolved = len(internal_missing)
    internal_total = internal_unresolved + static_fixed
    total_errors = len(missing)

    if args.report:
        tpl_path = pathlib.Path(args.report)
        tpl = tpl_path.read_text(encoding="utf-8")
        replacements = {
            "CMD": " ".join(sys.argv),
            "DATE_UTC": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "TOTAL_ERRORS": str(total_errors),
            "INTERNAL_TOTAL": str(internal_total),
            "INTERNAL_FIXED": str(static_fixed),
            "STATIC_FIXED": str(static_fixed),
            "INTERNAL_UNRESOLVED": str(internal_unresolved),
            "THIRD_PARTY_TOTAL": str(len(third_party_missing)),
            "FILES_REWRITTEN": str(files_rewritten),
            "REEXPORTS_TOUCHED": str(static_fixed),
        }
        for k, v in replacements.items():
            tpl = tpl.replace(f"{{{{{k}}}}}", v)
        tpl = fill_block(
            tpl,
            "UNRESOLVED_INTERNAL_BEGIN",
            "UNRESOLVED_INTERNAL_END",
            [f"- {m}" for m in internal_missing],
        )
        tpl = fill_block(
            tpl,
            "THIRD_PARTY_MISSING_BEGIN",
            "THIRD_PARTY_MISSING_END",
            [f"- {m}" for m in third_party_missing],
        )
        diff_lines = ["```diff"] + (diffs if diffs else ["# none"]) + ["```"]
        tpl = fill_block(
            tpl,
            "REWRITES_DIFF_BEGIN",
            "REWRITES_DIFF_END",
            diff_lines,
        )
        tpl_path.parent.mkdir(parents=True, exist_ok=True)
        tpl_path.write_text(tpl, encoding="utf-8")

    print("Done. Files rewritten:", files_rewritten)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

