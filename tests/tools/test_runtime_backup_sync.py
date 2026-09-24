"""Isolated S3 uploader tests; fake AWS never writes to the network."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from ai_trading.tools.runtime_recovery_backup import create_backup


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sync_runtime_backups_to_s3.sh"


def _seed_recovery_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    runtime = source / "runtime"
    runtime.mkdir(parents=True)
    with sqlite3.connect(runtime / "oms_intents.db") as connection:
        connection.execute("CREATE TABLE intents (id TEXT PRIMARY KEY)")
    models = source / "models"
    models.mkdir()
    (models / "registry_index.json").write_text("{}", encoding="utf-8")
    return create_backup(
        source,
        destination=tmp_path / "runtime" / "recovery_backups",
        status_path=tmp_path / "backup-status.json",
        required_databases=("oms_intents.db",),
    )


def _sync_with_fake_aws(
    tmp_path: Path,
    *,
    expected_owner: str = "399705375437",
    prefix: str = "pruned/",
    sync_enabled: str = "1",
    aws_failure: str = "",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    aws = fake_bin / "aws"
    aws.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$FAKE_AWS_CALLS"\n'
        'if [ "$1 $2" = "s3api put-object" ]; then\n'
        '  [ "$FAKE_AWS_FAILURE" = "put" ] && exit 5\n'
        '  previous=""\n'
        '  for arg in "$@"; do\n'
        '    [ "$previous" = "--body" ] && cp "$arg" "$FAKE_AWS_REMOTE"\n'
        '    previous="$arg"\n'
        '  done\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1 $2" = "s3api get-object" ]; then\n'
        '  [ "$FAKE_AWS_FAILURE" = "get" ] && exit 5\n'
        '  for destination do :; done\n'
        '  if [ "$FAKE_AWS_FAILURE" = "mismatch" ]; then\n'
        '    printf "corrupt" > "$destination"\n'
        "  else\n"
        '    cp "$FAKE_AWS_REMOTE" "$destination"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exit 7\n",
        encoding="utf-8",
    )
    aws.chmod(0o700)
    calls = tmp_path / "aws-calls.txt"
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ.get('PATH', '')}",
        AI_TRADING_RUNTIME_DIR=str(tmp_path / "runtime"),
        AI_TRADING_BACKUP_S3_SYNC_ENABLED=sync_enabled,
        AI_TRADING_BACKUP_S3_BUCKET="isolated-test-bucket",
        AI_TRADING_BACKUP_S3_PREFIX=prefix,
        AI_TRADING_BACKUP_S3_EXPECTED_BUCKET_OWNER=expected_owner,
        AI_TRADING_BACKUP_S3_REGION="us-east-2",
        FAKE_AWS_CALLS=str(calls),
        FAKE_AWS_REMOTE=str(tmp_path / "remote-object.gz"),
        FAKE_AWS_FAILURE=aws_failure,
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_SCRIPT.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, calls


def test_sync_requires_recovery_bundle_even_with_legacy_archive(tmp_path: Path) -> None:
    legacy = tmp_path / "runtime" / "logs" / "archive.bak.20260922.gz"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy log archive")

    result, calls = _sync_with_fake_aws(tmp_path)

    assert result.returncode == 3
    assert "Runtime recovery bundle missing" in result.stderr
    assert not calls.exists()


def test_disabled_sync_makes_no_aws_call(tmp_path: Path) -> None:
    result, calls = _sync_with_fake_aws(tmp_path, sync_enabled="0")

    assert result.returncode == 0
    assert not calls.exists()


def test_sync_uploads_only_latest_verified_bundle_and_reads_it_back(tmp_path: Path) -> None:
    old = _seed_recovery_bundle(tmp_path)
    newer = old.with_name("recovery.bak.20260924T030000Z-12345678.gz")
    shutil.copy2(old, newer)
    os.utime(old, (1, 1))
    os.utime(newer, (2, 2))
    legacy = tmp_path / "runtime" / "logs" / "archive.bak.20260922.gz"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")

    result, calls = _sync_with_fake_aws(tmp_path)

    assert result.returncode == 0, result.stderr
    assert f"pruned/recovery_backups/{newer.name}" in result.stdout
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("s3api put-object ")
    assert f"--body {newer}" in lines[0]
    assert f"--key pruned/recovery_backups/{newer.name}" in lines[0]
    assert "--server-side-encryption AES256" in lines[0]
    assert "--checksum-algorithm SHA256" in lines[0]
    assert all("--expected-bucket-owner 399705375437" in line for line in lines)
    assert lines[1].startswith("s3api get-object ")


def test_invalid_bundle_blocks_upload(tmp_path: Path) -> None:
    bundle = _seed_recovery_bundle(tmp_path)
    bundle.write_bytes(b"not a recovery bundle")

    result, calls = _sync_with_fake_aws(tmp_path)

    assert result.returncode != 0
    assert not calls.exists()


@pytest.mark.parametrize("expected_owner", ["", "not-an-account", "123"])
def test_missing_or_invalid_expected_owner_blocks_upload(
    tmp_path: Path, expected_owner: str
) -> None:
    _seed_recovery_bundle(tmp_path)

    result, calls = _sync_with_fake_aws(tmp_path, expected_owner=expected_owner)

    assert result.returncode == 2
    assert "expected bucket owner" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("prefix", ["/", "../other/", "pruned//nested/"])
def test_invalid_prefix_blocks_upload(tmp_path: Path, prefix: str) -> None:
    _seed_recovery_bundle(tmp_path)

    result, calls = _sync_with_fake_aws(tmp_path, prefix=prefix)

    assert result.returncode == 2
    assert "prefix is invalid" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("aws_failure", ["put", "get", "mismatch"])
def test_upload_or_readback_failure_does_not_report_success(
    tmp_path: Path, aws_failure: str
) -> None:
    _seed_recovery_bundle(tmp_path)

    result, calls = _sync_with_fake_aws(tmp_path, aws_failure=aws_failure)

    assert result.returncode != 0
    assert "uploaded and read back" not in result.stdout
    assert len(calls.read_text(encoding="utf-8").splitlines()) == (
        1 if aws_failure == "put" else 2
    )
    if aws_failure == "mismatch":
        assert "read-back mismatch" in result.stderr
