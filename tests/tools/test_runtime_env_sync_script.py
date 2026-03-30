from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "runtime_env_sync.py"
_SPEC = importlib.util.spec_from_file_location("runtime_env_sync", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
runtime_env_sync = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runtime_env_sync
_SPEC.loader.exec_module(runtime_env_sync)


def test_render_runtime_env_copies_plain_env(tmp_path: Path) -> None:
    src = tmp_path / ".env"
    dst = tmp_path / ".env.runtime"
    src.write_text(
        "\n".join(
            [
                "# comment",
                "ALPACA_API_KEY=abc",
                "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO=1",
                "HEALTHCHECK_PORT=8081",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = runtime_env_sync._render_runtime_env(src, dst)
    rendered = dst.read_text(encoding="utf-8")

    assert "ALPACA_API_KEY=abc" in rendered
    assert "HEALTHCHECK_PORT=8081" in rendered
    assert "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO" not in rendered
    assert summary["secrets_backend"] == "none"


def test_render_runtime_env_applies_aws_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / ".env"
    dst = tmp_path / ".env.runtime"
    src.write_text(
        "\n".join(
            [
                "AI_TRADING_SECRETS_BACKEND=aws-secrets-manager",
                "AI_TRADING_AWS_SECRET_ID=ai-trading/prod",
                "ALPACA_API_KEY=local-key",
                "ALPACA_SECRET_KEY=",
                "HEALTHCHECK_PORT=8081",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_fetch(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
        assert secret_id == "ai-trading/prod"
        assert region == ""
        assert profile == ""
        return {
            "ALPACA_API_KEY": "remote-key",
            "ALPACA_SECRET_KEY": "remote-secret",
        }

    monkeypatch.setattr(runtime_env_sync, "_fetch_aws_secret_payload", _fake_fetch)
    summary = runtime_env_sync._render_runtime_env(src, dst)
    rendered = dst.read_text(encoding="utf-8")

    assert "ALPACA_API_KEY=remote-key" in rendered
    assert "ALPACA_SECRET_KEY=remote-secret" in rendered
    assert "HEALTHCHECK_PORT=8081" in rendered
    assert summary["secrets_backend"] == "aws-secrets-manager"
    assert summary["manager_overrides_applied"] == 2


def test_render_runtime_env_require_managed_fails_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / ".env"
    dst = tmp_path / ".env.runtime"
    src.write_text(
        "\n".join(
            [
                "AI_TRADING_SECRETS_BACKEND=aws-secrets-manager",
                "AI_TRADING_AWS_SECRET_ID=ai-trading/prod",
                "AI_TRADING_REQUIRE_MANAGED_SECRETS=1",
                "ALPACA_SECRET_KEY=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_env_sync, "_fetch_aws_secret_payload", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="managed secret key"):
        runtime_env_sync._render_runtime_env(src, dst)
