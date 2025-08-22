import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "ai_trading"

def read(p): return p.read_text(encoding="utf-8", errors="ignore")
def write(p, s): p.write_text(s, encoding="utf-8")

def fix_metrics_logger(text: str) -> str:
    text = re.sub(r'(^|\n)\s*import\s+metrics_logger\s*(\n|$)',
                  r'\1from ai_trading.telemetry import metrics_logger\2', text)
    text = re.sub(r'(^|\n)\s*from\s+metrics_logger\s+import\s+',
                  r'\1from ai_trading.telemetry.metrics_logger import ', text)
    return text

def strip_mock_classes(text: str) -> str:
    rx_bases = re.compile(r'(?ms)^(\s*)class\s+Mock[A-Za-z0-9_]+\s*\([^)]*\)\s*:\s*\n(?:\1    |\t).+?(?=^\1\S|\Z)')
    rx_plain = re.compile(r'(?ms)^(\s*)class\s+Mock[A-Za-z0-9_]+\s*:\s*\n(?:\1    |\t).+?(?=^\1\S|\Z)')
    text = rx_bases.sub('', text)
    text = rx_plain.sub('', text)
    return text

def strip_dunder_getattr(text: str) -> str:
    rx = re.compile(r'(?ms)^def\s+__getattr__\s*\([^)]*\)\s*:\s*\n(?:\s{4}|\t).+?(?=^\S|\Z)')
    return rx.sub('', text)

def replace_exec_eval(text: str):
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

UPPER = re.compile(r'\bconfig\.([A-Z][A-Z0-9_]+)\b')
def migrate_config_callsites(text: str):
    names = set(m.group(1) for m in UPPER.finditer(text))
    if not names: return text, set()
    new = UPPER.sub(lambda m: f"S.{m.group(1).lower()}", text)
    if "from ai_trading.config import get_settings" not in new:
        new = "from ai_trading.config import get_settings\n" + new
    if re.search(r'^\s*S\s*=\s*get_settings\(\)\s*$', new, flags=re.MULTILINE) is None:
        lines = new.splitlines(keepends=True)
        j = 0
        while j < len(lines) and lines[j].lstrip().startswith(("import ","from ")):
            j += 1
        lines.insert(j, "S = get_settings()\n")
        new = "".join(lines)
    return new, names

def ensure_settings_fields(settings_py: Path, needed: set):
    if settings_py.exists():
        txt = read(settings_py)
    else:
        settings_py.parent.mkdir(parents=True, exist_ok=True)
        txt = ""
    if "BaseSettings" not in txt:
        fields = "".join([f"    {n.lower()}: str | None = Field(default=None, env='{n}')\n" for n in sorted(needed)])
        txt = (
            "from pydantic import BaseSettings, Field\n\n"
            "class Settings(BaseSettings):\n" + fields +
            "\n"
            "def get_settings() -> 'Settings':\n"
            "    global _S\n"
            "    try:\n"
            "        return _S\n"
            "    except NameError:\n"
            "        _S = Settings()  # type: ignore\n"
            "        return _S\n"
        )
    else:
        existing = set(re.findall(r'^\s*([a-z_][a-z0-9_]*)\s*:\s*', txt, flags=re.MULTILINE))
        additions = [f"    {n.lower()}: str | None = Field(default=None, env='{n}')\n"
                     for n in sorted(needed) if n.lower() not in existing]
        if additions and "class Settings" in txt:
            txt = re.sub(r'(class\s+Settings\s*\(.*?\)\s*:\s*\n)', r'\1' + "".join(additions), txt, count=1, flags=re.DOTALL)
        if "def get_settings" not in txt:
            txt += (
                "\n\ndef get_settings() -> 'Settings':\n"
                "    global _S\n"
                "    try:\n"
                "        return _S\n"
                "    except NameError:\n"
                "        _S = Settings()  # type: ignore\n"
                "        return _S\n"
            )
    write(settings_py, txt)

def main():
    upper_needed = set()
    changed = 0
    for p in PKG.rglob("*.py"):
        txt0 = read(p)
        txt = txt0
        txt = fix_metrics_logger(txt)
        txt = strip_mock_classes(txt)
        txt, changed_exec = replace_exec_eval(txt)
        if changed_exec:
            txt = ensure_helpers(txt)
        if p.as_posix().startswith("ai_trading/config/"):
            txt = strip_dunder_getattr(txt)
        txt, names = migrate_config_callsites(txt)
        upper_needed |= names
        if txt != txt0:
            write(p, txt)
            changed += 1
    ensure_settings_fields(PKG / "config" / "settings.py", upper_needed)

if __name__ == "__main__":
    main()
