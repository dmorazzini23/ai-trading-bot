from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.net.http_host_limit as http_host_limit


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_FALLBACK_PEAK_PATH", str(tmp_path / "peak.json"))
    monkeypatch.setattr(http_host_limit, "_IN_FLIGHT", 0)
    monkeypatch.setattr(http_host_limit, "_PEAK", 0)
    http_host_limit._FALLBACK_SYNC_LIMITERS.clear()
    http_host_limit._FALLBACK_ASYNC_LIMITERS.clear()










def test_reload_with_peak_file_uses_disk_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peak_path = tmp_path / "peak.json"
    peak_path.write_text('{"peak": 7}')
    monkeypatch.setenv("AI_TRADING_FALLBACK_PEAK_PATH", str(peak_path))

    reloaded = importlib.reload(http_host_limit)
    try:
        assert reloaded.current_peak() == 7
    finally:
        importlib.reload(http_host_limit)
