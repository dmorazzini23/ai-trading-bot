"""Isolated selection checks for the optional S3 recovery-bundle uploader."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sync_runtime_backups_to_s3.sh"


def _sync_with_fake_aws(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    aws = fake_bin / "aws"
    aws.write_text(
        '#!/bin/sh\nfind "$3" -type f -printf "%P\\n" > "$FAKE_AWS_STAGE_LIST"\n',
        encoding="utf-8",
    )
    aws.chmod(0o700)
    staged_list = tmp_path / "staged.txt"
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ.get('PATH', '')}",
        AI_TRADING_RUNTIME_DIR=str(tmp_path / "runtime"),
        AI_TRADING_BACKUP_S3_SYNC_ENABLED="1",
        AI_TRADING_BACKUP_S3_BUCKET="isolated-test-bucket",
        AI_TRADING_BACKUP_S3_RETENTION_ENABLED="0",
        FAKE_AWS_STAGE_LIST=str(staged_list),
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, staged_list


@pytest.mark.parametrize("legacy_archive", [False, True])
def test_enabled_sync_requires_recovery_bundle(
    tmp_path: Path, legacy_archive: bool
) -> None:
    if legacy_archive:
        legacy = tmp_path / "runtime" / "logs" / "archive.bak.20260922.gz"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy log archive")

    result, staged_list = _sync_with_fake_aws(tmp_path)

    assert result.returncode == 3
    assert "Runtime recovery bundle missing" in result.stderr
    assert not staged_list.exists()


def test_enabled_sync_stages_recovery_bundle_for_upload(tmp_path: Path) -> None:
    bundle = (
        tmp_path / "runtime" / "recovery_backups" / "recovery.bak.20260923T230000Z-abcd1234.gz"
    )
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"isolated bundle selection fixture")

    result, staged_list = _sync_with_fake_aws(tmp_path)

    assert result.returncode == 0, result.stderr
    assert staged_list.read_text(encoding="utf-8").splitlines() == [
        f"recovery_backups/{bundle.name}"
    ]
