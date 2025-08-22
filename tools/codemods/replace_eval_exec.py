from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("ai_trading")

DISPATCH_HINT = """# TODO: replace dynamic eval/exec with explicit dispatch
# Example:
# DISPATCH = {'momentum': run_momentum, 'mean_reversion': run_mean_reversion}
# fn = DISPATCH.get(name) or (_ for _ in ()).throw(ValueError(f"Unknown strategy {name}"))
# return fn(*args, **kwargs)
"""

def transform_text(text: str) -> tuple[str, bool]:
    # naive but safe: comment-out raw exec/eval and insert a dispatch hint just above
    out = []
    lines = text.splitlines()
    i = 0
    changed = False
    while i < len(lines):
        L = lines[i]
        if re.search(r"\bexec\s*\(", L) or re.search(r"(^|[^.])\beval\s*\(", L):
            # Check if this is actually .eval() method call (like PyTorch model.eval())
            if ".eval(" in L or "model.eval()" in L or "_MODEL.eval()" in L:
                # This is a method call, not raw eval - keep it
                out.append(L)
            else:
                out.append(DISPATCH_HINT)
                out.append("# " + L)
                changed = True
        else:
            out.append(L)
        i += 1
    return ("\n".join(out), changed)

def main():
    changed_files = 0
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            new, changed = transform_text(t)
            if changed:
                p.write_text(new, encoding="utf-8")
                changed_files += 1
        except Exception:
            pass

if __name__ == "__main__":
    main()
