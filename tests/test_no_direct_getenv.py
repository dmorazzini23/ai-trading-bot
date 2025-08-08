# tests/test_no_direct_getenv.py
import os, pathlib, re, pytest

@pytest.mark.skipif(os.getenv("ENFORCE_SETTINGS_ONLY") not in {"1","true","True"}, reason="Enable after config consolidation PR merges")
def test_no_direct_getenv_outside_settings():
    root = pathlib.Path(__file__).resolve().parents[1]  # project root
    offenders = []
    for p in root.rglob("*.py"):
        # allow in settings.py only
        if str(p).endswith("ai_trading/config/settings.py"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\bos\.getenv\s*\(", text):
            offenders.append(str(p))
    assert not offenders, f"os.getenv found outside settings.py: {offenders}"