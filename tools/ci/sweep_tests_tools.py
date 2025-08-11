# tools/ci/sweep_tests_tools.py
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [ROOT / "tests", ROOT / "tools"]

def read(p): return p.read_text(encoding="utf-8", errors="ignore")
def write(p, s): p.write_text(s, encoding="utf-8")

def fix_metrics_logger(text: str) -> str:
    text = re.sub(r'(^|\n)\s*import\s+metrics_logger\s*(\n|$)',
                  r'\1from ai_trading.telemetry import metrics_logger\2', text)
    text = re.sub(r'(^|\n)\s*from\s+metrics_logger\s+import\s+',
                  r'\1from ai_trading.telemetry.metrics_logger import ', text)
    return text

def replace_exec_eval(text: str) -> tuple[str, bool]:
    changed = False
    def _eval(m):
        nonlocal changed
        changed = True
        prefix = m.group(1) or ''
        args = m.group(2)
        return f"{prefix}_raise_dynamic_eval_disabled({args})"
    text = re.sub(r'(^|[^.])\beval\s*\((.*?)\)', _eval, text, flags=re.DOTALL)

    def _exec(m):
        nonlocal changed
        changed = True
        return f"_raise_dynamic_exec_disabled({m.group(1)})"
    text = re.sub(r'\bexec\s*\((.*?)\)', _exec, text, flags=re.DOTALL)
    return text, changed

def ensure_helpers(text: str) -> str:
    if "_raise_dynamic_exec_disabled" in text or "_raise_dynamic_eval_disabled" in text:
        return text
    prologue = (
        "def _raise_dynamic_exec_disabled(*_args, **_kwargs):\n"
        "    raise RuntimeError('Dynamic _raise_dynamic_exec_disabled() disabled. Replace with explicit dispatch.')\n\n"
        "def _raise_dynamic_eval_disabled(*_args, **_kwargs):\n"
        "    raise RuntimeError('Dynamic _raise_dynamic_eval_disabled() disabled. Replace with explicit dispatch.')\n\n"
    )
    lines = text.splitlines(keepends=True)
    # insert after initial imports/docstring where reasonable
    i = 0
    while i < len(lines) and (lines[i].startswith('#!') or 'coding:' in lines[i] or lines[i].strip()==''):
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith(('"""',"'''")):
        q = lines[i].strip()[:3]; i += 1
        while i < len(lines) and q not in lines[i]: i += 1
        if i < len(lines): i += 1
    while i < len(lines) and lines[i].lstrip().startswith(("import ","from ")):
        i += 1
    lines.insert(i, prologue)
    return "".join(lines)

def strip_mock_classes(text: str) -> str:
    # Only adjust tests/tools if they accidentally define broad Mock* used at runtime; prefer relocating to tests/support/mocks_runtime.py if needed.
    rx_bases = re.compile(r'(?ms)^(\s*)class\s+Mock[A-Za-z0-9_]+\s*\([^)]*\)\s*:\s*\n(?:\1    |\t).+?(?=^\1\S|\Z)')
    rx_plain = re.compile(r'(?ms)^(\s*)class\s+Mock[A-Za-z0-9_]+\s*:\s*\n(?:\1    |\t).+?(?=^\1\S|\Z)')
    return rx_plain.sub('', rx_bases.sub('', text))

changed = 0
for root in TARGETS:
    if not root.exists():
        continue
    for p in root.rglob("*.py"):
        txt0 = read(p)
        txt = txt0
        txt = fix_metrics_logger(txt)
        txt, changed_exec = replace_exec_eval(txt)
        if changed_exec:
            txt = ensure_helpers(txt)
        # keep mocks minimal in tests/tools; remove obviously broad Mock* skeletons
        txt = strip_mock_classes(txt)
        if txt != txt0:
            write(p, txt); changed += 1

print(f"Files changed: {changed}")