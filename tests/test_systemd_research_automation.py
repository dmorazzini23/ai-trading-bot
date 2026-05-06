from __future__ import annotations

from pathlib import Path


SYSTEMD_DIR = Path(__file__).resolve().parents[1] / "packaging" / "systemd"


def test_research_automation_units_follow_runtime_safety_contract() -> None:
    for service_name in (
        "ai-trading-research-daily.service",
        "ai-trading-research-weekly.service",
        "ai-trading-research-monthly.service",
        "ai-trading-research-manual@.service",
    ):
        content = (SYSTEMD_DIR / service_name).read_text(encoding="utf-8")
        assert "User=aiuser" in content
        assert "Group=aiuser" in content
        assert "WorkingDirectory=/home/aiuser/ai-trading-bot" in content
        assert "Environment=AI_TRADING_ENV_SRC=/home/aiuser/ai-trading-bot/.env" in content
        assert "EnvironmentFile=-/run/ai-trading-bot/ai-trading-runtime.env" in content
        assert "Environment=AI_TRADING_DOTENV_RUNTIME_OVERRIDE=1" in content
        assert "ExecStartPre=/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh" in content
        assert "ExecStart=/home/aiuser/ai-trading-bot/scripts/run_research_automation.sh" in content
        assert "NoNewPrivileges=true" in content
        assert "ProtectSystem=strict" in content
        assert "RuntimeDirectory=ai-trading-bot" in content
        assert "ReadWritePaths=/var/lib/ai-trading-bot" in content
        env_src_idx = content.index("Environment=AI_TRADING_ENV_SRC=/home/aiuser/ai-trading-bot/.env")
        runtime_idx = content.index("EnvironmentFile=-/run/ai-trading-bot/ai-trading-runtime.env")
        override_idx = content.index("Environment=AI_TRADING_DOTENV_RUNTIME_OVERRIDE=1")
        sync_idx = content.index("ExecStartPre=/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh")
        exec_idx = content.index("ExecStart=")
        assert env_src_idx < runtime_idx
        assert runtime_idx < override_idx
        assert sync_idx < exec_idx
        assert "/etc/ai-trading-bot/ai-trading.env" not in content


def test_research_automation_timers_are_after_hours_and_persistent() -> None:
    timer_expectations = {
        "ai-trading-research-daily.timer": "OnCalendar=Mon..Fri 16:35 America/New_York",
        "ai-trading-research-weekly.timer": "OnCalendar=Sat 10:15 America/New_York",
        "ai-trading-research-monthly.timer": "OnCalendar=Sat *-*-01..07 12:00 America/New_York",
    }
    for timer_name, calendar in timer_expectations.items():
        content = (SYSTEMD_DIR / timer_name).read_text(encoding="utf-8")
        assert calendar in content
        assert "Persistent=true" in content
        assert "Timezone=" not in content
