from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECT_ENV_PATTERN = re.compile(r"\bos\.getenv\b|\bos\.environ\b")

# Minimal allowlist for modules that intentionally inspect raw environment
# snapshots or implement config/env bootstrap mechanics.
_ALLOWLIST_PREFIXES = (
    "ai_trading/config/",
    "ai_trading/env/",
    "ai_trading/scripts/",
    "ai_trading/tools/",
    "ai_trading/validation/",
)
_ALLOWLIST_FILES = {
    "ai_trading/logging_filters.py",
    "ai_trading/policy/compiler.py",
    "ai_trading/util/env_check.py",
    "ai_trading/utils/env.py",
    "ai_trading/utils/environment.py",
    "ai_trading/utils/exec.py",
}


def _is_allowlisted(rel_path: str) -> bool:
    if rel_path in _ALLOWLIST_FILES:
        return True
    return rel_path.startswith(_ALLOWLIST_PREFIXES)


def test_runtime_modules_do_not_read_env_directly() -> None:
    offenders: list[str] = []
    for file_path in sorted((REPO_ROOT / "ai_trading").rglob("*.py")):
        rel_path = file_path.relative_to(REPO_ROOT).as_posix()
        if _is_allowlisted(rel_path):
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if DIRECT_ENV_PATTERN.search(line):
                offenders.append(f"{rel_path}:{line_number}: {line.strip()}")
    assert offenders == [], "Direct env access detected:\n" + "\n".join(offenders)
