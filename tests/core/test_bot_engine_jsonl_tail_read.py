from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from ai_trading.core import bot_engine


def _reference_read(path: Path, max_records: int) -> list[dict]:
    rows: deque[dict] = deque(maxlen=max(1, int(max_records)))
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return list(rows)


@pytest.mark.parametrize("chunk_bytes", [1, 7, 64, 1 << 20])
@pytest.mark.parametrize("trailing_newline", [True, False])
def test_read_jsonl_records_tail_matches_full_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chunk_bytes: int,
    trailing_newline: bool,
) -> None:
    monkeypatch.setattr(bot_engine, "_JSONL_TAIL_READ_CHUNK_BYTES", chunk_bytes)
    lines = [
        json.dumps({"i": 0, "symbol": "AAPL"}),
        "",
        "not json",
        json.dumps([1, 2, 3]),
        "   ",
        json.dumps({"i": 1, "note": "café — long " + "x" * 90}),
        "\r",
        json.dumps({"i": 2}) + "\r",
        "{truncated",
        json.dumps({"i": 3}),
    ]
    path = tmp_path / "records.jsonl"
    text = "\n".join(lines) + ("\n" if trailing_newline else "")
    path.write_bytes(text.encode("utf-8"))

    for max_records in (1, 2, 3, 4, 50):
        assert bot_engine._read_jsonl_records(str(path), max_records=max_records) == (
            _reference_read(path, max_records)
        )
    assert bot_engine._read_jsonl_records(str(path), max_records=2) == [
        {"i": 2},
        {"i": 3},
    ]


def test_read_jsonl_records_handles_empty_and_undecodable_lines(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert bot_engine._read_jsonl_records(str(empty)) == []

    path = tmp_path / "mixed.jsonl"
    path.write_bytes(b'{"i": 0}\n\xff\xfe broken\n{"i": 1}\n')
    assert bot_engine._read_jsonl_records(str(path), max_records=5) == [
        {"i": 0},
        {"i": 1},
    ]


def test_read_jsonl_records_reads_only_the_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(5000):
            handle.write(json.dumps({"i": index, "pad": "y" * 100}) + "\n")
    monkeypatch.setattr(bot_engine, "_JSONL_TAIL_READ_CHUNK_BYTES", 4096)

    read_sizes: list[int] = []
    real_open = Path.open

    def _tracking_open(self: Path, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def _read(size: int = -1):
            data = real_read(size)
            read_sizes.append(len(data))
            return data

        handle.read = _read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", _tracking_open)
    rows = bot_engine._read_jsonl_records(str(path), max_records=10)

    assert [row["i"] for row in rows] == list(range(4990, 5000))
    assert sum(read_sizes) <= 4096
