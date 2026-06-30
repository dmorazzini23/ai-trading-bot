from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "openclaw_weekly_runtime_audit.py"
    spec = importlib.util.spec_from_file_location("openclaw_weekly_runtime_audit_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def test_build_audit_reports_repo_and_cron_drift(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    openclaw = tmp_path / "openclaw"
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "scripts" / "openclaw_market_close_recap.py").write_text("x", encoding="utf-8")
    (openclaw / "workspace").mkdir(parents=True)
    (openclaw / "workspace" / "MEMORY.md").write_text("memory", encoding="utf-8")
    (openclaw / "hooks" / "transforms").mkdir(parents=True)
    (openclaw / "hooks" / "transforms" / "runtime_event.mjs").write_text("x", encoding="utf-8")

    def fake_run(args, *, cwd=None, timeout=12.0):
        if args[:2] == ["git", "status"]:
            return 0, "?? scripts/openclaw_market_close_recap.py\n"
        if args[:3] == ["openclaw", "cron", "list"]:
            return 0, (
                "ID Name Status\n"
                "00cd1509 weekly-runtime-audit error\n"
                "515d67cc market-close-recap ok\n"
                "plugin not installed\n"
            )
        return 0, "ok"

    monkeypatch.setattr(audit, "ROOT_DIR", root)
    monkeypatch.setattr(audit, "OPENCLAW_DIR", openclaw)
    monkeypatch.setattr(audit, "_run_command", fake_run)
    monkeypatch.setattr(audit, "_read_health", lambda: ("health ok=True status=healthy", {"ok": True}))

    text = audit.build_audit()

    assert "repo dirty" in text
    assert "errors=1" in text
    assert "command-based" in text


def test_build_audit_never_requires_openclaw_in_dry_mode(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    openclaw = tmp_path / "openclaw"
    root.mkdir()
    openclaw.mkdir()
    monkeypatch.setattr(audit, "ROOT_DIR", root)
    monkeypatch.setattr(audit, "OPENCLAW_DIR", openclaw)
    monkeypatch.setenv("AI_TRADING_AUDIT_SKIP_OPENCLAW", "1")
    monkeypatch.setattr(audit, "_read_health", lambda: ("health skipped", None))

    text = audit.build_audit()

    assert "OpenClaw cron skipped" in text
    assert "Single best hardening step" in text
