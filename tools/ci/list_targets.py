# tools/ci/list_targets.py
import re
from pathlib import Path

ROOT = Path("ai_trading")
patterns = {
    "bare_except": re.compile(r"^\s*except\s*:\s*(#.*)?$", re.MULTILINE),
    "requests_no_timeout": re.compile(
        r"requests\.(get|post|put|delete|patch)\s*\((?![^)]*timeout\s*=)"
    ),
    "subprocess_unsafe": re.compile(
        r"subprocess\.(run|Popen|call|check_call|check_output)\s*\((?![^)]*timeout\s*=)"
    ),
    "eval_exec_raw": re.compile(r"(^|[^.])\beval\s*\(|\bexec\s*\("),
    "yaml_unsafe_load": re.compile(r"yaml\.load\s*\((?![^)]*Loader\s*=)"),
    "datetime_naive_now": re.compile(r"datetime\.now\s*\(\s*\)"),
    "empty_except": re.compile(
        r"^\s*except\s+(?:[A-Za-z_][\w\.]*\s+as\s+\w+|[A-Za-z_][\w\.]*)?\s*:\s*(?:\n\s+(pass|return\s*(?:None)?|#.*\n))",
        re.MULTILINE,
    ),
    "mutable_default": re.compile(r"def\s+\w+\s*\([^)]*(\[\]|\{\}|set\(\))", re.DOTALL),
    "time_sleep_in_async": re.compile(
        r"^\s*async\s+def[\s\S]*?^\s*time\.sleep\(", re.MULTILINE
    ),
}
rows = []
for p in ROOT.rglob("*.py"):
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    lines = text.splitlines()
    for issue, rx in patterns.items():
        for m in rx.finditer(text):
            pos = m.start()
            line = text.count("\n", 0, pos) + 1
            snippet = lines[line - 2 : line + 1]
            rows.append(
                (
                    issue,
                    str(p),
                    line,
                    " | ".join(s.strip() for s in snippet if s.strip()),
                )
            )
rows.sort()
print("issue,file,line,snippet")
for issue, file, line, snip in rows:
    print(f"{issue},{file},{line},\"{snip.replace('\"','\"\"')}\"")
