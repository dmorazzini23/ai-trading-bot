from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from ai_trading.tools.live_cutover_drill import main


def test_live_cutover_drill_paper_mode_writes_durable_intent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "drill.db"
    out_path = tmp_path / "drill.json"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")

    rc = main(
        [
            "--execution-mode",
            "paper",
            "--intent-store-path",
            str(db_path),
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "ok"
    assert payload.get("oms_drill", {}).get("durability_ok") is True
    with sqlite3.connect(db_path) as conn:
        intent_count = int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0])
        fill_count = int(conn.execute("SELECT COUNT(*) FROM intent_fills").fetchone()[0])
    assert intent_count >= 1
    assert fill_count >= 1


def test_live_cutover_drill_live_mode_requires_non_sqlite_database(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_path = tmp_path / "drill_live_fail.json"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")

    rc = main(
        [
            "--execution-mode",
            "live",
            "--intent-store-path",
            str(tmp_path / "live.db"),
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 1

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "failed"
    assert payload.get("database_url_ok") is False
