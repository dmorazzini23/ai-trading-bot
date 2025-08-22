from __future__ import annotations

from pathlib import Path

import libcst as cst


class NoneComparisonTransformer(cst.CSTTransformer):
    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison):
        # Rewrite "== None" / "!= None" to "is None" / "is not None"
        comps = []
        for comp in updated_node.comparisons:
            if isinstance(comp.operator, cst.Equal | cst.NotEqual) and isinstance(comp.comparator, cst.Name) and comp.comparator.value == "None":
                op = cst.Is() if isinstance(comp.operator, cst.Equal) else cst.IsNot()
                comps.append(comp.with_changes(operator=op))
            else:
                comps.append(comp)
        return updated_node.with_changes(comparisons=comps)

def rewrite_code(code: str) -> str:
    mod = cst.parse_module(code)
    mod = mod.visit(NoneComparisonTransformer())
    return mod.code

def run_path(p: Path) -> None:
    code = p.read_text(encoding="utf-8")
    new = rewrite_code(code)
    if new != code:
        p.write_text(new, encoding="utf-8")

if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    for p in root.rglob("*.py"):
        s = str(p)
        if any(part in s for part in ("/.venv/", "/venv/", "/.git/", "/artifacts/", "/build/", "/dist/")):
            continue
        if "/tests/snapshots/" in s:
            continue
        run_path(p)
