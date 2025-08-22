# tools/scan_import_time.py
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = {"ai_trading", "tests"}
EXCLUDE = {".venv", ".git", "__pycache__", "tools"}

PATS = {
    "MODULE_CONST": re.compile(r"^\s*[A-Z0-9_]+\s*=\s*get_settings\(\)\.\w+", re.M),
    "MODULE_CFG":   re.compile(r"^\s*CFG\s*=\s*get_settings\(\)", re.M),
    "DIRECT_ATTR":  re.compile(r"\bget_settings\(\)\.\w+"),
    "BARE_CALL":    re.compile(r"\bget_settings\("),
}

def scan_file(p: pathlib.Path):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    hits = []
    if PATS["MODULE_CONST"].search(txt): hits.append("MODULE_CONST_FROM_SETTINGS")
    if PATS["MODULE_CFG"].search(txt):   hits.append("MODULE_CFG_FROM_SETTINGS")
    if PATS["DIRECT_ATTR"].search(txt):  hits.append("DIRECT_SETTINGS_ATTR")
    if PATS["BARE_CALL"].search(txt):    hits.append("BARE_GET_SETTINGS")
    return hits

def main():
    any_hits = False
    for py in ROOT.rglob("*.py"):
        if any(part in EXCLUDE for part in py.parts): continue
        if py.parts[0] not in INCLUDE: continue
        hits = scan_file(py)
        if hits:
            any_hits = True
    sys.exit(1 if any_hits else 0)

if __name__ == "__main__":
    main()
