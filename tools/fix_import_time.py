# tools/fix_import_time.py
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE = {".venv", ".git", "__pycache__", "tools"}

KNOWN_GETTERS = {
    "rebalance_interval_min": "get_rebalance_interval_min",
    "news_api_key": "get_news_api_key",
    "disaster_dd_limit": "get_disaster_dd_limit",
    "capital_cap": "get_capital_cap",
    "dollar_risk_limit": "get_dollar_risk_limit",
    "buy_threshold": "get_buy_threshold",
    "conf_threshold": "get_conf_threshold",
    "trade_cooldown_min": "get_trade_cooldown_min",
    "portfolio_drift_threshold": "get_portfolio_drift_threshold",
    "max_drawdown_threshold": "get_max_drawdown_threshold",
    "daily_loss_limit": "get_daily_loss_limit",
    "max_portfolio_positions": "get_max_portfolio_positions",
}

RE_MODULE_CONST = re.compile(r"^(\s*)([A-Z0-9_]+)\s*=\s*get_settings\(\)\.(\w+)\s*$", re.M)
RE_MODULE_CFG   = re.compile(r"^(\s*)CFG\s*=\s*get_settings\(\)\s*$", re.M)
RE_DIRECT_ATTR  = re.compile(r"\bget_settings\(\)\.(\w+)")

IMPORT_GETTERS_TMPL = "from ai_trading.settings import ({})"
LOCAL_RUNTIME_IMPORT = "from ai_trading.settings import get_settings as get_runtime_settings"

def ensure_getter_imports(lines, getters):
    if not getters: return False
    imp_line = IMPORT_GETTERS_TMPL.format(", ".join(sorted(getters)))
    joined = "".join(lines)
    if imp_line in joined: return False
    idx = 0
    while idx < len(lines) and lines[idx].startswith(("#!", "# -*-", "from __future__")):
        idx += 1
    lines.insert(idx, imp_line + "\n")
    return True

def add_function_local_runtime_import(lines):
    j = "".join(lines)
    if "get_runtime_settings()" not in j:
        return False
    # naive insertion: put a local import after the first def that uses it
    func = re.compile(r"^(\s*)def\s+\w+\s*\(.*?\):", re.M)
    pos = 0
    while True:
        m = func.search(j, pos)
        if not m: break
        start = m.end()
        head_line = j[:m.start()].count("\n")
        body = j[start:start+2000]
        if "get_runtime_settings()" in body:
            lines.insert(head_line+1, m.group(1) + LOCAL_RUNTIME_IMPORT + "\n")
            return True
        pos = start
    return False

def patch_text(text):
    original = text
    # remove CFG at module scope
    text = RE_MODULE_CFG.sub(r"\1# REMOVED: module-scope CFG = get_settings()", text)

    # remove module constants, replace uses later
    const_map = {}
    def _const_sub(m):
        indent, const, field = m.groups()
        repl = KNOWN_GETTERS.get(field, None)
        const_map[const] = f"{KNOWN_GETTERS[field]}()" if repl else f"get_runtime_settings().{field}"
        return indent + f"# REMOVED: module-scope {const} = get_settings().{field}\n"
    text = RE_MODULE_CONST.sub(_const_sub, text)

    # replace removed const names
    for const, repl in const_map.items():
        text = re.sub(rf"\b{re.escape(const)}\b", repl, text)

    # replace direct get_settings().field everywhere
    def _attr_sub(m):
        field = m.group(1)
        return f"{KNOWN_GETTERS[field]}()" if field in KNOWN_GETTERS else f"get_runtime_settings().{field}"
    text = RE_DIRECT_ATTR.sub(_attr_sub, text)

    if text == original:
        return None  # no change
    # manage imports
    lines = text.splitlines(True)
    getters_used = {g for g in KNOWN_GETTERS.values() if f"{g}()" in text}
    ensure_getter_imports(lines, getters_used)
    add_function_local_runtime_import(lines)
    return "".join(lines)

def patch_file(p: pathlib.Path) -> bool:
    src = p.read_text(encoding="utf-8", errors="ignore")
    new = patch_text(src)
    if new and new != src:
        p.write_text(new, encoding="utf-8")
        print(f"UPDATED {p}")
        return True
    return False

def main():
    changed = 0
    for py in ROOT.rglob("*.py"):
        if any(part in EXCLUDE for part in py.parts): continue
        if py.parts[0] not in {"ai_trading", "tests"}: continue
        if py.name == "settings.py" and py.parts[-2] == "ai_trading": continue
        try:
            if patch_file(py): changed += 1
        except Exception as e:
            print(f"PATCH_FAIL {py}: {e}")
    print(f"Done. Files changed: {changed}")

if __name__ == "__main__":
    main()
