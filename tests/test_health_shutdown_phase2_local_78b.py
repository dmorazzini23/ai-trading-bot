from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ai_trading import health_monitor as hm
from ai_trading import shutdown_handler as sh


def _shutdown_handler(monkeypatch):
    monkeypatch.setattr(sh.signal, "signal", lambda sig, handler: f"old-{sig}")
    return sh.ShutdownHandler()


def test_health_monitor_psutil_resource_branches(monkeypatch):
    monitor = hm.HealthMonitor()
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=1: 96.0,
        virtual_memory=lambda: SimpleNamespace(percent=96.0, used=3 * 1024**3, available=2 * 1024**3),
        disk_usage=lambda path: SimpleNamespace(percent=96.0, used=10 * 1024**3, free=5 * 1024**3),
        pids=lambda: [1, 2, 3],
        Process=lambda: SimpleNamespace(num_fds=lambda: 7),
        net_connections=lambda: [object(), object()],
        NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError,
    )
    monkeypatch.setattr(hm, "_HAS_PSUTIL", True)
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    monkeypatch.setattr(hm.os, "getloadavg", lambda: (1.0, 2.0, 3.0))

    metrics = monitor._collect_system_metrics()
    assert metrics.cpu_percent == 96.0
    assert metrics.open_files == 7
    assert metrics.network_connections == 2
    assert monitor._check_cpu_usage()["status"] == "critical"
    assert monitor._check_memory_usage()["status"] == "critical"
    assert monitor._check_disk_usage()["status"] == "critical"

    fake_psutil.cpu_percent = lambda interval=1: 50.0
    fake_psutil.virtual_memory = lambda: SimpleNamespace(percent=50.0, used=1024**3, available=4 * 1024**3)
    fake_psutil.disk_usage = lambda path: SimpleNamespace(percent=50.0, used=1024**3, free=5 * 1024**3)
    assert monitor._check_cpu_usage()["status"] == "healthy"
    assert monitor._check_memory_usage()["status"] == "healthy"
    assert monitor._check_disk_usage()["status"] == "healthy"


def test_health_monitor_resource_fallbacks_and_global_wrappers(monkeypatch):
    monitor = hm.HealthMonitor()
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=1: (_ for _ in ()).throw(TypeError("bad cpu")),
        virtual_memory=lambda: SimpleNamespace(percent=81.0, used=1, available=1),
        disk_usage=lambda path: SimpleNamespace(percent=86.0, used=1, free=1),
        pids=lambda: [],
        Process=lambda: SimpleNamespace(),
        net_connections=lambda: [],
        NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError,
    )
    monkeypatch.setattr(hm, "_HAS_PSUTIL", True)
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    assert monitor._collect_system_metrics().cpu_percent == 0

    monkeypatch.setattr(hm, "_HAS_PSUTIL", False)
    monkeypatch.setattr(hm, "snapshot_basic", lambda: {"cpu_percent": 86.0, "mem_percent": 44.0})
    assert monitor._check_cpu_usage()["status"] == "warning"
    assert monitor._check_memory_usage()["status"] == "unknown"
    assert monitor._check_disk_usage()["status"] == "unknown"

    monkeypatch.setattr(hm, "_health_monitor", None)
    created = hm.get_health_monitor()
    assert hm.get_health_monitor() is created
    monkeypatch.setattr(hm, "_health_monitor", monitor)
    assert hm.get_system_health()["status"] in {"unknown", "healthy", "warning", "critical"}

    async def fake_start():
        monitor.running = True

    async def fake_stop():
        monitor.running = False

    async def fake_run_all_checks():
        return []

    monkeypatch.setattr(monitor, "start_monitoring", fake_start)
    monkeypatch.setattr(monitor, "stop_monitoring", fake_stop)
    monkeypatch.setattr(monitor, "run_all_checks", fake_run_all_checks)
    asyncio.run(hm.start_health_monitoring())
    assert monitor.running is True
    asyncio.run(hm.stop_health_monitoring())
    assert monitor.running is False
    assert hm.run_health_check() == []


def test_shutdown_forced_positions_and_failure_paths(monkeypatch):
    handler = _shutdown_handler(monkeypatch)
    events: list[str] = []
    handler.config["force_close_positions"] = True
    handler.config["save_state_on_shutdown"] = False
    handler.register_order_handler(lambda: [])
    handler.register_position_handler(lambda: [{"symbol": "AAPL"}])

    async def close_position(position):
        events.append(f"close:{position['symbol']}")

    handler._close_single_position = close_position
    assert asyncio.run(handler.shutdown(sh.ShutdownReason.USER_REQUEST)) is True
    assert events == ["close:AAPL"]
    assert handler.get_shutdown_status().positions_closed == 1

    failed = _shutdown_handler(monkeypatch)

    async def fail_gracefully():
        raise OSError("stop failed")

    failed._graceful_shutdown = fail_gracefully
    assert asyncio.run(failed.shutdown(sh.ShutdownReason.SYSTEM_ERROR)) is False
    assert failed.get_shutdown_status().phase is sh.ShutdownPhase.FAILED
    assert failed.get_shutdown_status().errors == ["stop failed"]

    emergency = _shutdown_handler(monkeypatch)

    async def fail_emergency():
        raise TimeoutError("too slow")

    emergency._emergency_shutdown = fail_emergency
    assert asyncio.run(emergency.shutdown(sh.ShutdownReason.EMERGENCY_STOP, emergency=True)) is False
    assert emergency.get_shutdown_status().phase is sh.ShutdownPhase.FAILED


def test_shutdown_hooks_state_files_and_module_wrappers(tmp_path, monkeypatch):
    handler = _shutdown_handler(monkeypatch)
    monkeypatch.chdir(tmp_path)
    handler._status.reason = sh.ShutdownReason.USER_REQUEST
    handler._status.phase = sh.ShutdownPhase.SAVING_STATE
    handler.update_system_state({"mode": "paper"})
    handler.set_active_positions([{"symbol": "AAPL"}])
    handler.set_pending_orders([{"id": "order-1"}])

    asyncio.run(handler._save_system_state())
    asyncio.run(handler._save_critical_state())
    saved = sorted((tmp_path / "logs").glob("*.json"))
    assert [path.name.split("_state_")[0] for path in saved] == ["emergency", "shutdown"]
    payloads = [json.loads(path.read_text()) for path in saved]
    assert any(payload.get("system_state") == {"mode": "paper"} for payload in payloads)
    assert any(payload.get("emergency_shutdown") is True for payload in payloads)

    async def async_error():
        raise OSError("async hook failed")

    def sync_error():
        raise OSError("sync hook failed")

    handler.register_pre_shutdown_hook(async_error)
    handler.register_cleanup_hook(sync_error)
    handler.register_post_shutdown_hook(sync_error)
    asyncio.run(handler._run_pre_shutdown_hooks())
    asyncio.run(handler._run_cleanup_hooks())
    asyncio.run(handler._run_post_shutdown_hooks())
    assert any("Pre-shutdown hook error" in err for err in handler.get_shutdown_status().errors)
    assert any("Cleanup hook error" in err for err in handler.get_shutdown_status().errors)

    asyncio.run(handler._cancel_single_order({"id": "order-1"}))
    asyncio.run(handler._close_single_position({"symbol": "AAPL"}))

    monkeypatch.setattr(sh, "_shutdown_handler", handler)
    called: list[tuple[str, bool]] = []

    async def fake_shutdown(reason=sh.ShutdownReason.USER_REQUEST, emergency=False):
        called.append((reason.value, emergency))
        return True

    handler.shutdown = fake_shutdown
    assert asyncio.run(sh.initiate_shutdown(sh.ShutdownReason.SCHEDULED_MAINTENANCE)) is True
    assert asyncio.run(sh.emergency_shutdown()) is True
    assert called == [("scheduled_maintenance", False), ("emergency_stop", True)]
