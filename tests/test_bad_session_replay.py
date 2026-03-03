from __future__ import annotations

import json
from pathlib import Path

from ai_trading.replay.bad_session import (
    build_replay_dataset_from_bad_session,
    canonical_bad_session_events,
    deterministic_replay_fingerprint,
)


def _write_bad_session(path: Path) -> None:
    rows = [
        {"timestamp": "2026-01-02T14:31:00Z", "symbol": "AAPL", "price": 190.1, "volume": 1000},
        {"timestamp": "2026-01-02T14:32:00Z", "symbol": "AAPL", "price": 190.2, "volume": 900},
        {"timestamp": "2026-01-02T14:31:00Z", "symbol": "MSFT", "price": 410.4, "volume": 800},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_bad_session_fingerprint_is_deterministic(tmp_path: Path) -> None:
    log_path = tmp_path / "bad_session.jsonl"
    _write_bad_session(log_path)

    first = deterministic_replay_fingerprint(log_path, seed=11)
    second = deterministic_replay_fingerprint(log_path, seed=11)
    assert first == second


def test_build_replay_dataset_from_bad_session(tmp_path: Path) -> None:
    log_path = tmp_path / "bad_session.jsonl"
    _write_bad_session(log_path)

    report = build_replay_dataset_from_bad_session(
        log_path,
        output_dir=tmp_path / "replay_out",
        seed=11,
    )

    assert "fingerprint" in report
    manifest_path = Path(report["manifest"])
    assert manifest_path.exists()
    events = canonical_bad_session_events(log_path)
    assert len(events) == 3
    assert (tmp_path / "replay_out" / "AAPL.csv").exists()
