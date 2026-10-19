from __future__ import annotations

import ast
import asyncio
import inspect
import json
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

from ai_trading import alpaca_api as api
from ai_trading.telemetry import trade_updates_stream as tus

ALL_EVENTS = (
    "new",
    "pending_new",
    "accepted",
    "partial_fill",
    "fill",
    "canceled",
    "pending_cancel",
    "expired",
    "rejected",
    "replaced",
    "pending_replace",
    "done_for_day",
)


def _message(event: str, *, order_id: str = "o-1", filled_qty: str = "0", ts: str | None = None,
             execution_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event": event,
        "timestamp": ts or f"2026-10-16T14:30:{ALL_EVENTS.index(event):02d}.000000Z",
        "order": {
            "id": order_id,
            "client_order_id": f"c-{order_id}",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "filled_qty": filled_qty,
            "filled_avg_price": "190.5" if filled_qty != "0" else None,
            "status": event,
            "created_at": "2026-10-16T14:29:59Z",
            "filled_at": "2026-10-16T14:30:04Z" if event == "fill" else None,
        },
    }
    if event in {"partial_fill", "fill"}:
        data.update({"price": "190.5", "qty": "5", "position_qty": filled_qty})
    if execution_id:
        data["execution_id"] = execution_id
    return {"stream": "trade_updates", "data": data}


class FakeStream:
    def __init__(self, *, messages: list[dict[str, Any]] | None = None, fail_connect: bool = False,
                 disconnect_after: bool = True) -> None:
        self.messages = list(messages or [])
        self.fail_connect = fail_connect
        self.disconnect_after = disconnect_after
        self.handler: Any = None
        self.stopped = asyncio.Event()
        self.closed = False

    def subscribe_trade_updates(self, handler: Any) -> None:
        assert asyncio.iscoroutinefunction(handler)
        self.handler = handler

    async def _start_ws(self) -> None:
        if self.fail_connect:
            raise ConnectionError("connect refused")

    async def _consume(self) -> None:
        for message in self.messages:
            await self.handler(message)
        if self.disconnect_after:
            raise ConnectionError("socket closed")
        await self.stopped.wait()

    async def stop_ws(self) -> None:
        self.stopped.set()

    async def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self, streams: list[FakeStream]) -> None:
        self.streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, key: str, secret: str, *, paper: bool) -> FakeStream:
        self.calls.append({"key": key, "secret": secret, "paper": paper})
        return self.streams.pop(0)


class LogRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def _add(self, level: str, msg: str, *args: Any, extra: dict[str, Any] | None = None, **_k: Any) -> None:
        self.events.append((level, msg, dict(extra or {})))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._add("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._add("warning", msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._add("debug", msg, *args, **kwargs)

    def names(self) -> list[str]:
        return [msg for _level, msg, _extra in self.events]


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    tus.reset_trade_updates_stream_state()
    monkeypatch.delenv(tus.ENABLED_ENV, raising=False)
    yield
    tus.reset_trade_updates_stream_state()


@pytest.fixture()
def logs(monkeypatch: pytest.MonkeyPatch) -> LogRecorder:
    recorder = LogRecorder()
    monkeypatch.setattr(api, "_log", recorder)
    return recorder


def _run(factory: Factory, recorder: tus.TradeUpdatesShadowRecorder, **kwargs: Any) -> None:
    asyncio.run(
        api.start_trade_updates_stream(
            "key",
            "secret",
            paper=kwargs.pop("paper", True),
            stream_factory=factory,
            recorder=recorder,
            backoff_base_seconds=0.001,
            backoff_max_seconds=0.002,
            parity_interval_seconds=0,
            **kwargs,
        )
    )


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_event_types_recorded_and_replays_deduped(tmp_path: Path, logs: LogRecorder) -> None:
    artifact = tmp_path / "runtime" / "trade_updates_stream.jsonl"
    first = [
        _message(event, filled_qty="5" if event == "partial_fill" else "10" if event == "fill" else "0")
        for event in ALL_EVENTS
    ]
    replay = [first[ALL_EVENTS.index("fill")], first[ALL_EVENTS.index("partial_fill")]]
    factory = Factory([FakeStream(messages=first), FakeStream(messages=replay)])

    _run(factory, tus.TradeUpdatesShadowRecorder(path=artifact), max_connect_attempts=2)

    rows = _read_lines(artifact)
    assert [row["event"] for row in rows] == list(ALL_EVENTS)
    fill = rows[ALL_EVENTS.index("fill")]
    assert fill["order_id"] == "o-1"
    assert fill["client_order_id"] == "c-o-1"
    assert fill["symbol"] == "AAPL"
    assert fill["side"] == "buy"
    assert fill["order_qty"] == 10.0
    assert fill["qty"] == 5.0
    assert fill["filled_qty"] == 10.0
    assert fill["price"] == 190.5
    assert fill["filled_avg_price"] == 190.5
    assert fill["position_qty"] == 10.0
    assert fill["event_timestamp"] == "2026-10-16T14:30:04.000000Z"
    assert fill["order_timestamps"]["filled_at"] == "2026-10-16T14:30:04Z"
    assert fill["raw"]["data"]["event"] == "fill"
    assert fill["received_at"].endswith("Z")

    state = tus.observe_trade_updates_stream_state()
    assert state["reconnects"] == 1
    assert state["duplicates_skipped"] == 2
    assert state["shadow_only"] is True
    assert state["connected"] is False
    names = logs.names()
    assert names.count("TRADE_UPDATES_STREAM_CONNECTED") == 2
    assert names.count("TRADE_UPDATES_STREAM_DISCONNECTED") == 2
    assert names.count("TRADE_UPDATES_STREAM_RECONNECTING") == 1
    assert names.count("TRADE_UPDATES_STREAM_EVENT") == len(ALL_EVENTS)


def test_dedupe_distinguishes_execution_id_and_filled_qty(tmp_path: Path) -> None:
    recorder = tus.TradeUpdatesShadowRecorder(path=tmp_path / "s.jsonl", max_keys=2)
    ts = "2026-10-16T14:30:00Z"
    assert recorder.record(_message("partial_fill", filled_qty="2", ts=ts, execution_id="e1"))
    assert recorder.record(_message("partial_fill", filled_qty="2", ts=ts, execution_id="e1")) is None
    assert recorder.record(_message("partial_fill", filled_qty="4", ts=ts, execution_id="e2"))
    assert recorder.record(_message("partial_fill", filled_qty="6", ts=ts))
    # Bounded: the oldest key was evicted, so it records again.
    assert recorder.record(_message("partial_fill", filled_qty="2", ts=ts, execution_id="e1"))
    assert len(_read_lines(tmp_path / "s.jsonl")) == 4


def test_connect_failure_backs_off_then_reconnects(tmp_path: Path, logs: LogRecorder) -> None:
    artifact = tmp_path / "s.jsonl"
    factory = Factory(
        [FakeStream(fail_connect=True), FakeStream(messages=[_message("new")])]
    )
    _run(factory, tus.TradeUpdatesShadowRecorder(path=artifact), max_connect_attempts=2)

    assert len(_read_lines(artifact)) == 1
    disconnects = [extra for _l, msg, extra in logs.events if msg == "TRADE_UPDATES_STREAM_DISCONNECTED"]
    assert disconnects[0]["was_connected"] is False
    assert "connect refused" in disconnects[0]["reason"]
    assert logs.names().count("TRADE_UPDATES_STREAM_CONNECTED") == 1
    assert tus.observe_trade_updates_stream_state()["reconnects"] == 1


def test_backoff_delay_is_capped_with_jitter() -> None:
    for attempt in range(12):
        delay = api._stream_backoff_delay(attempt, base=1.0, cap=60.0)
        step = min(60.0, 2.0**attempt)
        assert step * 0.5 <= delay <= step
    assert api._stream_backoff_delay(50, base=1.0, cap=60.0) <= 60.0


def test_running_event_clear_stops_stream(tmp_path: Path, logs: LogRecorder) -> None:
    running = threading.Event()
    running.set()
    stream = FakeStream(messages=[_message("new")], disconnect_after=False)
    original_handler_hook = stream.subscribe_trade_updates

    def _subscribe(handler: Any) -> None:
        async def _wrapped(message: Any) -> None:
            await handler(message)
            assert tus.observe_trade_updates_stream_state()["connected"] is True
            running.clear()

        original_handler_hook(_wrapped)

    stream.subscribe_trade_updates = _subscribe  # type: ignore[method-assign]
    _run(Factory([stream]), tus.TradeUpdatesShadowRecorder(path=tmp_path / "s.jsonl"), running=running)

    assert stream.stopped.is_set()
    assert stream.closed is True
    disconnects = [extra for _l, msg, extra in logs.events if msg == "TRADE_UPDATES_STREAM_DISCONNECTED"]
    assert disconnects == [{"reason": "stop_requested", "was_connected": True}]
    assert "TRADE_UPDATES_STREAM_RECONNECTING" not in logs.names()


def test_kill_switch_skips_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: LogRecorder) -> None:
    monkeypatch.setenv(tus.ENABLED_ENV, "0")
    factory = Factory([FakeStream()])
    _run(factory, tus.TradeUpdatesShadowRecorder(path=tmp_path / "s.jsonl"))

    assert factory.calls == []
    assert "TRADE_UPDATES_STREAM_DISABLED" in logs.names()
    state = tus.observe_trade_updates_stream_state()
    assert state["enabled"] is False
    assert state["degraded"] is False


def test_startup_shadow_thread_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading.core import startup_runtime

    monkeypatch.setenv(tus.ENABLED_ENV, "0")
    assert startup_runtime.start_shadow_trade_updates_stream_thread(threading.Event()) is None


@pytest.mark.parametrize(
    ("base_url", "execution_mode", "expected"),
    [
        ("https://paper-api.alpaca.markets", "live", True),
        ("https://api.alpaca.markets", "paper", False),
        ("", "live", False),
        ("", "paper", True),
    ],
)
def test_paper_live_derivation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logs: LogRecorder,
    base_url: str,
    execution_mode: str,
    expected: bool,
) -> None:
    real_get_env = tus.get_env
    overrides = {"ALPACA_TRADING_BASE_URL": base_url, "EXECUTION_MODE": execution_mode}

    def _get_env(key: str, default: Any = None, **kwargs: Any) -> Any:
        if key in overrides:
            return overrides[key]
        return real_get_env(key, default, **kwargs)

    # The settings layer supplies a paper base URL default, so patch the
    # lookup to exercise the EXECUTION_MODE fallback when no URL is configured.
    monkeypatch.setattr(tus, "get_env", _get_env)
    assert tus.trade_updates_stream_paper() is expected

    factory = Factory([FakeStream()])
    _run(
        factory,
        tus.TradeUpdatesShadowRecorder(path=tmp_path / "s.jsonl"),
        paper=None,
        max_connect_attempts=1,
    )
    assert factory.calls == [{"key": "key", "secret": "secret", "paper": expected}]
    assert tus.observe_trade_updates_stream_state()["paper"] is expected


def test_startup_runtime_no_longer_hardcodes_paper() -> None:
    from ai_trading.core import startup_runtime

    source = inspect.getsource(startup_runtime._run_trade_updates_stream)
    assert "paper=True" not in source
    assert "paper=None" in source


def test_write_failure_never_raises_and_logs_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logged: list[str] = []
    monkeypatch.setattr(
        tus,
        "_log",
        type("L", (), {"warning": staticmethod(lambda msg, **_k: logged.append(msg))})(),
    )
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir", encoding="utf-8")
    recorder = tus.TradeUpdatesShadowRecorder(path=blocker / "s.jsonl")
    for idx in range(3):
        assert recorder.record(_message("new", order_id=f"o-{idx}")) is not None
    assert logged == ["TRADE_UPDATES_STREAM_WRITE_FAILED"]
    assert tus.observe_trade_updates_stream_state()["write_failures"] == 3


def test_default_artifact_path_resolves_under_isolated_runtime() -> None:
    path = tus.resolve_stream_artifact_path(for_write=True)
    assert path.name == "trade_updates_stream.jsonl"
    assert not str(path).startswith("/var/lib/ai-trading-bot")


def test_health_section_connected_and_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading import health_payload

    monkeypatch.setenv(tus.GRACE_ENV, "0")
    tus.mark_stream_started(enabled=True, paper=True)
    section = health_payload._trade_updates_stream_snapshot()
    assert section["enabled"] is True
    assert section["connected"] is False
    assert section["degraded"] is True
    assert section["shadow_only"] is True
    flags = health_payload._build_runtime_attention_flags(
        provider_state={}, broker_state={}, service_state={}, trade_updates_stream=section
    )
    assert "trade_updates_stream_degraded" in flags

    tus.mark_stream_connected()
    tus.record_parity_summary(
        {"session": "2026-10-16", "matched": 3, "stream_only": 0, "polled_only": 0,
         "price_delta_bps_p95": 0.2, "mismatches": [{"order_id": "x"}]}
    )
    section = health_payload._trade_updates_stream_snapshot()
    assert section["connected"] is True
    assert section["degraded"] is False
    assert section["parity"]["matched"] == 3
    assert "mismatches" not in section["parity"]
    flags = health_payload._build_runtime_attention_flags(
        provider_state={}, broker_state={}, service_state={}, trade_updates_stream=section
    )
    assert "trade_updates_stream_degraded" not in flags


def test_health_section_within_grace_not_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading import health_payload

    monkeypatch.setenv(tus.GRACE_ENV, "3600")
    tus.mark_stream_started(enabled=True, paper=True)
    assert health_payload._trade_updates_stream_snapshot()["degraded"] is False


def test_health_section_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_trading import health_payload

    def _boom() -> dict[str, Any]:
        raise RuntimeError("state broken")

    monkeypatch.setattr(health_payload, "observe_trade_updates_stream_state", _boom)
    section = health_payload._trade_updates_stream_snapshot()
    assert section["reason"] == "trade_updates_stream_state_unavailable"
    assert section["shadow_only"] is True


def test_runtime_health_payload_includes_section() -> None:
    from ai_trading import health_payload

    payload = health_payload.build_runtime_health_payload()
    assert payload["trade_updates_stream"]["shadow_only"] is True


_STREAM_MODULES = (
    "ai_trading.telemetry.trade_updates_stream",
    "ai_trading.tools.trade_updates_parity",
)
_FORBIDDEN_IMPORT_PREFIXES = ("ai_trading.oms", "ai_trading.execution", "ai_trading.core")


def _imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module_name", _STREAM_MODULES)
def test_shadow_modules_do_not_import_oms_or_fill_writers(module_name: str) -> None:
    module = __import__(module_name, fromlist=["_"])
    imported = _imported_modules(inspect.getsource(module))
    assert not {name for name in imported if name.startswith(_FORBIDDEN_IMPORT_PREFIXES)}


def test_stream_entrypoint_never_touches_oms_or_fill_events() -> None:
    for func in (
        api.start_trade_updates_stream,
        api._refresh_trade_updates_parity_loop,
        api._default_trade_updates_stream,
    ):
        source = inspect.getsource(func)
        imported = _imported_modules(textwrap.dedent(source))
        assert not {name for name in imported if name.startswith(_FORBIDDEN_IMPORT_PREFIXES)}
        for token in ("event_store", "fill_events", "append_oms", "record_fill"):
            assert token not in source


def test_stream_run_writes_only_shadow_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: LogRecorder
) -> None:
    import ai_trading.oms.event_store as event_store

    def _forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("stream code must not write the OMS event store")

    for name, value in vars(event_store).items():
        if callable(value) and not name.startswith("_") and getattr(value, "__module__", "") == event_store.__name__:
            monkeypatch.setattr(event_store, name, _forbidden)

    data_dir = Path(tus.resolve_stream_artifact_path(for_write=True)).parent
    before = {path.name for path in data_dir.glob("*")} if data_dir.exists() else set()
    messages = [_message("partial_fill", filled_qty="5"), _message("fill", filled_qty="10")]
    _run(Factory([FakeStream(messages=messages)]), tus.TradeUpdatesShadowRecorder(), max_connect_attempts=1)

    after = {path.name for path in data_dir.glob("*")}
    assert after - before <= {"trade_updates_stream.jsonl"}
    assert not (data_dir / "fill_events.jsonl").exists()
