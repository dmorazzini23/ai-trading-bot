"""Regression coverage for the production-state isolation in tests/conftest.py."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

import tests.conftest as root_conftest

_PROBE = "/var/lib/ai-trading-bot/runtime/__pytest_read_guard_probe__.json"


@pytest.fixture
def isolated_hits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Redirect the active guard's hit list so probes do not fail this test."""

    hits: list[tuple[str, str]] = []
    monkeypatch.setitem(root_conftest._PRODUCTION_ACCESS_STATE, "hits", hits)
    return hits


def test_production_roots_include_canonical_state_dirs() -> None:
    roots = root_conftest._PRODUCTION_STATE_ROOTS
    for root in ("/var/lib/ai-trading-bot", "/var/log/ai-trading-bot", "/var/cache/ai-trading-bot"):
        assert root in roots


def test_runtime_env_points_at_session_temp_dir() -> None:
    temp_root = os.path.realpath(tempfile.gettempdir())
    for key in (
        "AI_TRADING_RUNTIME_ROOT",
        "AI_TRADING_RECAP_RUNTIME_DIR",
        "AI_TRADING_RUNTIME_DAILY_REPORT_PATH",
        "AI_TRADING_GATE_AUTO_DISABLE_RUNTIME_GONOGO_PATH",
        "AI_TRADING_SLACK_INCIDENT_STATE_PATH",
        "AI_TRADING_SLACK_EOD_STATE_PATH",
        "AI_TRADING_INFRA_AUDIT_PATH",
        "YF_CACHE_DIR",
        "AI_TRADING_LOG_DIR",
        "AI_TRADING_MODELS_DIR",
    ):
        value = os.environ[key]
        assert os.path.realpath(value).startswith(temp_root + os.sep), (key, value)


def test_no_env_value_references_production_state() -> None:
    leaking = {
        key: value
        for key, value in os.environ.items()
        if any(root in value for root in root_conftest._PRODUCTION_STATE_ROOTS)
    }
    assert leaking == {}


def test_relative_runtime_artifact_env_is_unset_by_default() -> None:
    for key in (
        "DATABASE_URL",
        "AI_TRADING_OMS_INTENT_STORE_PATH",
        "AI_TRADING_DECISION_LOG_PATH",
        "AI_TRADING_FILL_EVENTS_PATH",
        "AI_TRADING_RESEARCH_REPORT_ROOT",
        "AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH",
    ):
        assert key not in os.environ, key


def test_read_guard_records_open_under_production_root(isolated_hits: list[tuple[str, str]]) -> None:
    assert root_conftest._PRODUCTION_ACCESS_STATE["active"] is True
    with pytest.raises(OSError):
        with open(_PROBE, encoding="utf-8"):
            pass
    assert ("open", _PROBE) in isolated_hits


def test_read_guard_records_sqlite_and_listing(isolated_hits: list[tuple[str, str]]) -> None:
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(f"file:{_PROBE}?mode=ro", uri=True)
    with pytest.raises(OSError):
        os.listdir("/var/lib/ai-trading-bot/__pytest_read_guard_missing_dir__")
    events = {event for event, _path in isolated_hits}
    assert {"sqlite3.connect", "os.listdir"} <= events


def test_read_guard_ignores_tmp_path_and_existence_probes(
    tmp_path: Path, isolated_hits: list[tuple[str, str]]
) -> None:
    fixture = tmp_path / "runtime" / "fill_events.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}\n", encoding="utf-8")
    assert fixture.read_text(encoding="utf-8") == "{}\n"
    Path(_PROBE).exists()
    assert isolated_hits == []


def test_production_path_hit_matches_only_under_roots() -> None:
    hit = root_conftest._production_path_hit
    assert hit("/var/lib/ai-trading-bot/runtime/fill_events.jsonl") is not None
    assert hit(Path("/var/log/ai-trading-bot/bot.log")) is not None
    assert hit(b"/var/cache/ai-trading-bot/yf/x") is not None
    assert hit("/var/lib/ai-trading-bot-other/x") is None
    assert hit("/tmp/ai-trading-bot/runtime/x") is None
    assert hit(3) is None
