from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_trading.runtime.atomic_io import atomic_write_text


def test_atomic_write_text_replaces_destination(tmp_path: Path) -> None:
    destination = tmp_path / "runtime" / "artifact.json"

    path = atomic_write_text(destination, '{"ok": true}')

    assert path == destination
    assert destination.read_text(encoding="utf-8") == '{"ok": true}'
    assert not list(destination.parent.glob(".artifact.json.*.tmp"))


def test_atomic_write_text_removes_temp_file_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"

    def _raise_fsync(_fd: int) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "fsync", _raise_fsync)

    with pytest.raises(OSError, match="disk unavailable"):
        atomic_write_text(destination, '{"ok": false}')

    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))
