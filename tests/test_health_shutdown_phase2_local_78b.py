from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_trading import health_monitor as hm




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
    monkeypatch.setattr(hm, "snapshot_basic", lambda: {"has_psutil": True})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    assert monitor._collect_system_metrics().cpu_percent == 0

    monkeypatch.setattr(hm, "snapshot_basic", lambda: {"has_psutil": False, "cpu_percent": 86.0, "mem_percent": 44.0})
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




