from __future__ import annotations

import json
from pathlib import Path

from tools import mcp_secrets_manager_server as secrets_srv


def test_secrets_backend_status_reports_config(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    runtime_env_file = tmp_path / ".env.runtime"
    env_file.write_text(
        "\n".join(
            [
                "AI_TRADING_SECRETS_BACKEND=aws-secrets-manager",
                "AI_TRADING_AWS_SECRET_ID=ai-trading-bot/prod",
                "AI_TRADING_MANAGED_SECRET_KEYS=ALPACA_API_KEY,ALPACA_SECRET_KEY",
                "ALPACA_API_KEY=",
                "ALPACA_SECRET_KEY=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_env_file.write_text(
        "ALPACA_API_KEY=abc\nALPACA_SECRET_KEY=def\n",
        encoding="utf-8",
    )

    payload = secrets_srv.tool_secrets_backend_status(
        {"env_file": str(env_file), "runtime_env_file": str(runtime_env_file)}
    )
    assert payload["configured"] is True
    assert payload["secret_id"] == "ai-trading-bot/prod"
    assert payload["runtime_managed_present_count"] >= 2


def test_migrate_local_env_requires_confirm() -> None:
    payload = secrets_srv.tool_migrate_local_env_to_aws({})
    assert payload["executed"] is False
    assert "confirm" in payload["reason"]


def test_aws_secret_inventory_parses_secret_string(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_TRADING_AWS_SECRET_ID=ai-trading-bot/prod\nAI_TRADING_AWS_REGION=us-west-2\n",
        encoding="utf-8",
    )

    def _fake_run_cmd(cmd: list[str], timeout_s: int = 120):
        _ = cmd, timeout_s
        return {
            "cmd": cmd,
            "rc": 0,
            "stdout": json.dumps(
                {
                    "ARN": "arn:aws:secretsmanager:us-west-2:123:secret:ai-trading-bot/prod",
                    "SecretString": json.dumps(
                        {"ALPACA_API_KEY": "A" * 26, "ALPACA_SECRET_KEY": "B" * 44}
                    ),
                }
            ),
            "stderr": "",
        }

    monkeypatch.setattr(secrets_srv, "_run_cmd", _fake_run_cmd)
    payload = secrets_srv.tool_aws_secret_inventory({"env_file": str(env_file)})
    assert payload["key_count"] == 2
    assert payload["key_lengths"]["ALPACA_API_KEY"] == 26
    assert payload["key_lengths"]["ALPACA_SECRET_KEY"] == 44
