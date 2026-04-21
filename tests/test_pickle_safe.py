from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ai_trading.utils import pickle_safe


def test_safe_pickle_load_works_in_pytest_context(tmp_path: Path) -> None:
    path = tmp_path / "payload.pkl"
    with path.open("wb") as handle:
        pickle.dump({"ok": True}, handle)

    loaded = pickle_safe.safe_pickle_load(path, [tmp_path])

    assert loaded == {"ok": True}


def test_safe_pickle_load_fails_closed_when_legacy_gate_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "payload.pkl"
    with path.open("wb") as handle:
        pickle.dump({"ok": True}, handle)

    monkeypatch.setattr(pickle_safe, "_pickle_deserialization_allowed", lambda: False)

    with pytest.raises(RuntimeError, match="retired generic model deserialization"):
        pickle_safe.safe_pickle_load(path, [tmp_path])
