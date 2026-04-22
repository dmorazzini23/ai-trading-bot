from __future__ import annotations

from pathlib import Path


def test_direct_deserialization_calls_are_limited_to_approved_files() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {
        Path("ai_trading/models/artifacts.py"),
        Path("ai_trading/tools/migrate_pickle_artifacts.py"),
        Path("ai_trading/utils/pickle_safe.py"),
    }
    patterns = ("joblib.load(", "pickle.load(", "_pickle.load(")
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", "venv", ".venv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns) and rel not in allowed:
            offenders.append(str(rel))

    assert offenders == []
