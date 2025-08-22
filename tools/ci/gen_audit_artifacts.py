# tools/ci/gen_audit_artifacts.py
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = ROOT / "tools" / "out"
OUT.mkdir(parents=True, exist_ok=True)

PATTERNS = {
    "import_guard_block": re.compile(r"(?ms)try:\s*(?:\n\s*(?:from\s+[.\w]+\s+import\s+[^\n]+|import\s+[^\n]+))+?\n\s*except\s+ImportError\s*:\s*"),
    "mock_class": re.compile(r"(^|\n)\s*class\s+Mock[A-Za-z0-9_]+\s*:", re.MULTILINE),
    "config_dunder_getattr": re.compile(r"def\s+__getattr__\s*\("),
    "uppercase_alias_property": re.compile(r"@property\s+def\s+[A-Z0-9_]+\s*\("),
    "raw_exec": re.compile(r"\bexec\s*\("),
    "raw_eval": re.compile(r"(^|[^.])\beval\s*\(", re.MULTILINE),
    "root_metrics_logger": re.compile(r"\b(from|import)\s+metrics_logger\b"),
}

def scan():
    rows = []
    for p in (ROOT).rglob("*.py"):
        # limit to project files (skip venvs, builds, etc.)
        rel = p.relative_to(ROOT)
        srel = str(rel)
        if any(part in srel for part in ("/venv/", "/.venv/", "/site-packages/", "/build/", "/dist/", "/__pycache__/")):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                ln = text.count("\n", 0, m.start()) + 1
                snippet = text.splitlines()[ln-1][:240]
                rows.append({"file": srel, "line": ln, "marker": name, "snippet": snippet})
    return rows

def pycompile_errors():
    errs = []
    for p in (ROOT).rglob("*.py"):
        rel = p.relative_to(ROOT)
        srel = str(rel)
        if any(part in srel for part in ("/venv/", "/.venv/", "/site-packages/", "/build/", "/dist/", "/__pycache__/")):
            continue
        try:
            compile(p.read_text(encoding="utf-8", errors="ignore"), srel, "exec")
        except Exception as e:
            errs.append({"file": srel, "error": repr(e)})
    return errs

def write_csv(path: Path, rows, headers):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    markers = scan()
    guards = [r for r in markers if r["marker"] == "import_guard_block"]
    mocks = [r for r in markers if r["marker"] == "mock_class"]
    comp = pycompile_errors()

    write_csv(OUT / "markers_all.csv", markers, ["file","line","marker","snippet"])
    write_csv(OUT / "import_guards.csv", guards, ["file","line","marker","snippet"])
    write_csv(OUT / "runtime_mocks.csv", mocks, ["file","line","marker","snippet"])
    write_csv(OUT / "compile_errors.csv", comp, ["file","error"])

    # summary
    counts = {}
    for r in markers:
        counts[r["marker"]] = counts.get(r["marker"], 0) + 1
    write_csv(OUT / "summary.csv", [{"marker":k,"count":v} for k,v in sorted(counts.items(), key=lambda x:-x[1])], ["marker","count"])

    for fn in ("markers_all.csv","import_guards.csv","runtime_mocks.csv","compile_errors.csv","summary.csv"):
        pass
