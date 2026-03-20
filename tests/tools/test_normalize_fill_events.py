from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools.normalize_fill_events import normalize_fill_events_file


def test_normalize_fill_events_backfills_canonical_fields(tmp_path: Path) -> None:
    path = tmp_path / "fill_events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-03-18T18:00:00+00:00",
                        "event": "fill_recorded",
                        "symbol": "AAPL",
                        "entry_price": "101.25",
                        "qty": "7",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-03-18T18:00:01+00:00",
                        "event": "fill_recorded",
                        "symbol": "MSFT",
                        "fill_price": 212.5,
                        "fill_qty": 3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = normalize_fill_events_file(path, backup=True)

    assert summary["ok"] is True
    assert summary["updated_rows"] >= 1
    assert summary["missing_fill_fields"] == 0
    assert (tmp_path / "fill_events.jsonl.bak").exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["fill_price"] == 101.25
    assert rows[0]["fill_qty"] == 7.0
    assert rows[1]["fill_price"] == 212.5
    assert rows[1]["fill_qty"] == 3.0
