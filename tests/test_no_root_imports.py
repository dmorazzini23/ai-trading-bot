import pathlib
import re


def test_no_root_level_imports_of_migrated_modules():
    root = pathlib.Path(__file__).resolve().parents[1]
    banned = {
        r"\bfrom\s+signals\s+import\b",
        r"\bfrom\s+data_fetcher\s+import\b",
        r"\bfrom\s+pipeline\s+import\b",
        r"\bfrom\s+indicators\s+import\b",
        r"\bfrom\s+portfolio\s+import\b",
        r"\bfrom\s+rebalancer\s+import\b",
        r"^\s*import\s+signals\b",
        r"^\s*import\s+data_fetcher\b",
        r"^\s*import\s+pipeline\b",
        r"^\s*import\s+indicators\b",
        r"^\s*import\s+portfolio\b",
        r"^\s*import\s+rebalancer\b",
    }
    offenders = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in banned:
            if re.search(pat, text, re.MULTILINE):
                offenders.append(f"{p}:{pat}")
                break
    assert not offenders, f"Root imports are no longer supported. Offenders: {offenders}"
