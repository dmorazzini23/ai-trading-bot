# tools/codemods/phase3_canonical_imports.py
"""
Phase-3 codemod: rewrite legacy and relative imports to canonical ai_trading.*.
Usage:
  python tools/codemods/phase3_canonical_imports.py
"""
from __future__ import annotations
from pathlib import Path
import libcst as cst
from libcst.metadata import PositionProvider, QualifiedNameProvider, ScopeProvider

# Map legacy module names (previous root-level duplicates) to canonical targets.
LEGACY_TO_CANON = {
    "bot_engine":       "ai_trading.bot_engine",
    "data_fetcher":     "ai_trading.data_fetcher",
    "data_validation":  "ai_trading.data_validation",
    "indicators":       "ai_trading.indicators",
    "rebalancer":       "ai_trading.rebalancer",
    "runner":           "ai_trading.runner",
    "signals":          "ai_trading.signals",
}

ALLOWED_TOP = ("ai_trading",)  # preserve canonical package

class CanonicalizeImports(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider, ScopeProvider)

    def __init__(self, file_path: Path):
        self.file_path = file_path
        # In tests we allow more flexibility, but still prefer canonical.
        self.is_test = "tests" in str(file_path.parts)

    def _rewrite_name(self, name: str) -> str:
        # "yfinance" and third-party modules are left untouched.
        return LEGACY_TO_CANON.get(name, name)

    def leave_Import_alias(self, node: cst.ImportAlias, updated: cst.ImportAlias):
        if isinstance(updated.name, cst.Name):
            new = self._rewrite_name(updated.name.value)
            if new != updated.name.value:
                return updated.with_changes(name=cst.parse_module(new).body[0].body[0].names[0].name)
        elif isinstance(updated.name, cst.Attribute):
            # import bot_engine.something -> ai_trading.bot_engine.something
            base = updated.name.value
            code = cst.Module([]).code_for_node(base)
            root = code.split(".")[0]
            new_root = self._rewrite_name(root)
            if new_root != root:
                suffix = code[len(root):]
                new_full = new_root + suffix
                return updated.with_changes(name=cst.parse_expression(new_full))
        return updated

    def leave_ImportFrom(self, node: cst.ImportFrom, updated: cst.ImportFrom):
        # Handle "from X import Y" and relative "from . import Z" / "from ..module import Z"
        if updated.module is None:
            return updated  # "from  import" (malformed) — ignore
        
        # Handle absolute imports like "from bot_engine import ..."
        if isinstance(updated.module, cst.Name):
            root = updated.module.value
            new_root = self._rewrite_name(root)
            if new_root != root:
                return updated.with_changes(module=cst.parse_expression(new_root))
        elif isinstance(updated.module, cst.Attribute):
            # Handle "from bot_engine.something import ..."
            code = cst.Module([]).code_for_node(updated.module)
            root = code.split(".")[0]
            new_root = self._rewrite_name(root)
            if new_root != root:
                suffix = code[len(root):]
                new_full = new_root + suffix
                return updated.with_changes(module=cst.parse_expression(new_full))
        elif isinstance(updated.module, cst.RelativeImport):
            # For relative imports, only convert if we're NOT inside ai_trading/
            # and only for specific patterns that make sense to convert
            parts = self.file_path.parts
            
            # Skip conversion of relative imports within ai_trading/ - they should stay relative
            if "ai_trading" in parts:
                return updated
            
            # For files outside ai_trading/, convert relative imports if they reference legacy modules
            if updated.module.module is not None:
                module_name = cst.Module([]).code_for_node(updated.module.module)
                if module_name in LEGACY_TO_CANON:
                    new_module = LEGACY_TO_CANON[module_name]
                    return updated.with_changes(module=cst.parse_expression(new_module))
        
        return updated

def rewrite_file(path: Path):
    src = path.read_text(encoding="utf-8", errors="ignore")
    mod = cst.parse_module(src)
    wrapper = cst.MetadataWrapper(mod)
    new_mod = wrapper.visit(CanonicalizeImports(path))
    new_src = new_mod.code
    if new_src != src:
        path.write_text(new_src, encoding="utf-8")

def should_visit(path: Path) -> bool:
    if not path.suffix == ".py":
        return False
    rel = str(path)
    # Visit production code and tests; skip venv, build, .git, artifacts
    skip_dirs = (".git/", "venv/", ".venv/", "build/", "dist/", "artifacts/", "models/", "__pycache__/")
    return not any(s in rel for s in skip_dirs)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    for p in root.rglob("*.py"):
        if should_visit(p):
            rewrite_file(p)