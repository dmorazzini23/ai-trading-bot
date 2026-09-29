"""AWS provisioning plan is scoped and tested without making API calls."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/provision_host_failure_alarm.sh"


def _run_with_fake_aws(tmp_path: Path, *, account: str) -> tuple[subprocess.CompletedProcess[str], str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws = fake_bin / "aws"
    aws.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$AWS_CALLS"\n'
        'case "$1 $2" in\n'
        '  "sts get-caller-identity") printf "%s\\n" "$FAKE_ACCOUNT" ;;\n'
        '  "sns create-topic") printf "%s\\n" "arn:aws:sns:us-east-2:399705375437:ai-trading-host-failure" ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n',
        encoding="utf-8",
    )
    aws.chmod(0o700)
    calls = tmp_path / "calls.txt"
    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:{env.get('PATH', '')}",
        AWS_CALLS=str(calls),
        FAKE_ACCOUNT=account,
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "dmorazzini23@gmail.com"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8")


def test_alarm_configuration_requires_expected_account(tmp_path: Path) -> None:
    result, calls = _run_with_fake_aws(tmp_path, account="123456789012")
    assert result.returncode != 0
    assert "sns create-topic" not in calls
    assert "cloudwatch put-metric-alarm" not in calls


def test_alarm_configuration_uses_exact_metric_and_missing_data(tmp_path: Path) -> None:
    result, calls = _run_with_fake_aws(tmp_path, account="399705375437")
    assert result.returncode == 0
    assert "sns subscribe" in calls
    assert "--notification-endpoint dmorazzini23@gmail.com" in calls
    assert "--namespace AITrading/Host" in calls
    assert "--metric-name PaperRuntimeResponding" in calls
    assert "--dimensions Name=Host,Value=ai-trading-primary" in calls
    assert "--treat-missing-data breaching" in calls
    assert "--alarm-actions arn:aws:sns:us-east-2:399705375437:ai-trading-host-failure" in calls
