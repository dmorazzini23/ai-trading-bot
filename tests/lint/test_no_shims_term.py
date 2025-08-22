import pathlib
import re

ROOT = pathlib.Path("ai_trading")
BAD = []
for p in ROOT.rglob("*.py"):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"\bNoOp[A-Za-z_]*\b", txt) or re.search(r"\bshim\b", txt, flags=re.I):
        BAD.append(str(p))
assert not BAD, f"Remove shim artifacts from: {BAD}"
