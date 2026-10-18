from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.tools import runtime_performance_report as rpr


def _row(ts: str, total: int, accepted: int, edge: float) -> str:
    return json.dumps(
        {
            "ts": ts,
            "records_total": total,
            "accepted_records": accepted,
            "total_expected_net_edge_bps": edge,
        }
    )


def _reference_daily(path: Path) -> list[dict]:
    buckets: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rpr._fold_gate_effectiveness_row(buckets, parsed)
    return rpr._gate_effectiveness_daily_rows(buckets)


@pytest.fixture(autouse=True)
def _clear_gate_daily_cache():
    rpr._GATE_DAILY_CACHE.clear()
    yield
    rpr._GATE_DAILY_CACHE.clear()


def test_gate_daily_summary_folds_only_appended_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate_effectiveness.jsonl"
    path.write_text(
        "\n".join(
            [
                _row("2026-10-15T14:00:00Z", 10, 4, 1.5),
                "not json",
                _row("", 3, 3, 9.0),
                _row("2026-10-15T15:00:00Z", 6, 1, -0.25),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    folded: list[bytes] = []
    real_fold = rpr._fold_gate_effectiveness_line

    def _counting_fold(buckets, raw_line):
        folded.append(raw_line)
        return real_fold(buckets, raw_line)

    monkeypatch.setattr(rpr, "_fold_gate_effectiveness_line", _counting_fold)

    first = rpr._summarize_gate_effectiveness_daily(path)
    assert first == _reference_daily(path)
    assert first[0]["total_records"] == 16
    assert len(folded) == 4

    folded.clear()
    assert rpr._summarize_gate_effectiveness_daily(path) == first
    assert folded == []

    with path.open("a", encoding="utf-8") as handle:
        handle.write(_row("2026-10-16T14:00:00Z", 5, 5, 2.0) + "\n")
        # Unterminated final line counts now and is re-read once completed.
        handle.write(_row("2026-10-16T15:00:00Z", 1, 0, 0.5))
    folded.clear()
    appended = rpr._summarize_gate_effectiveness_daily(path)
    assert appended == _reference_daily(path)
    assert [row["date"] for row in appended] == ["2026-10-15", "2026-10-16"]
    assert len(folded) == 2

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    folded.clear()
    assert rpr._summarize_gate_effectiveness_daily(path) == appended
    assert len(folded) == 1


def test_gate_daily_summary_rebuilds_after_rotation_or_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "gate_effectiveness.jsonl"
    path.write_text(_row("2026-10-15T14:00:00Z", 10, 4, 1.5) + "\n", encoding="utf-8")
    assert rpr._summarize_gate_effectiveness_daily(path)[0]["total_records"] == 10

    # Rotation: a new file (new inode) at the same path.
    rotated = tmp_path / "gate_effectiveness.new"
    rotated.write_text(_row("2026-10-17T14:00:00Z", 2, 1, 0.0) + "\n", encoding="utf-8")
    rotated.replace(path)
    assert rpr._summarize_gate_effectiveness_daily(path) == _reference_daily(path)
    assert [row["date"] for row in rpr._summarize_gate_effectiveness_daily(path)] == [
        "2026-10-17"
    ]

    # In-place rewrite (same inode) that grows past the cached offset.
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_row("2026-10-18T14:00:00Z", 7, 7, 3.0) + "\n")
        handle.write(_row("2026-10-18T15:00:00Z", 1, 1, 1.0) + "\n")
    assert rpr._summarize_gate_effectiveness_daily(path) == _reference_daily(path)
    assert rpr._summarize_gate_effectiveness_daily(path)[0]["total_records"] == 8

    # Truncation below the cached offset.
    path.write_text("", encoding="utf-8")
    assert rpr._summarize_gate_effectiveness_daily(path) == []


def test_gate_daily_summary_fails_without_caching_undecodable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "gate_effectiveness.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(UnicodeDecodeError):
        rpr._summarize_gate_effectiveness_daily(path)
    assert str(path) not in rpr._GATE_DAILY_CACHE
    summary = rpr.summarize_gate_effectiveness(tmp_path / "gate_effectiveness_summary.json")
    assert summary["daily_gate_stats"] == []
