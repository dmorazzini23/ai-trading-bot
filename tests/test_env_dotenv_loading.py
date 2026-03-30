from __future__ import annotations

import importlib
import os
from pathlib import Path

import ai_trading.env as env_mod


class _DotenvStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def load_dotenv(self, *, dotenv_path: str, override: bool) -> None:
        self.calls.append((dotenv_path, override))
        path = Path(dotenv_path)
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if override or os.environ.get(key) is None:
                os.environ[key] = value


def _reset_env_module(monkeypatch) -> _DotenvStub:
    module = importlib.reload(env_mod)
    stub = _DotenvStub()
    monkeypatch.setattr(module, "_dotenv", stub)
    monkeypatch.setattr(module, "PYTHON_DOTENV_RESOLVED", True)
    monkeypatch.setattr(module, "_ENV_LOADED", False)
    monkeypatch.setattr(module, "is_test_runtime", lambda include_pytest_module=True: False)
    monkeypatch.setattr(module, "refresh_alpaca_credentials_cache", lambda: None)
    return stub


def test_load_dotenv_if_present_defaults_to_non_overriding(monkeypatch, tmp_path):
    stub = _reset_env_module(monkeypatch)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("FOO=bar\n", encoding="utf-8")

    assert env_mod.load_dotenv_if_present(str(dotenv_file))
    assert stub.calls == [(str(dotenv_file), False)]


def test_ensure_dotenv_loaded_does_not_blank_existing_secrets(monkeypatch, tmp_path):
    _reset_env_module(monkeypatch)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "ALPACA_API_KEY=\nALPACA_SECRET_KEY=\nWEBHOOK_SECRET=\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ALPACA_API_KEY", "existing_api_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "existing_secret_key")
    monkeypatch.setenv("WEBHOOK_SECRET", "existing_webhook_secret")

    env_mod.ensure_dotenv_loaded(str(dotenv_file))

    assert os.environ["ALPACA_API_KEY"] == "existing_api_key"
    assert os.environ["ALPACA_SECRET_KEY"] == "existing_secret_key"
    assert os.environ["WEBHOOK_SECRET"] == "existing_webhook_secret"


def test_ensure_dotenv_loaded_populates_missing_values(monkeypatch, tmp_path):
    _reset_env_module(monkeypatch)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "ALPACA_API_KEY=from_dotenv_api\nALPACA_SECRET_KEY=from_dotenv_secret\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    env_mod.ensure_dotenv_loaded(str(dotenv_file))

    assert os.environ["ALPACA_API_KEY"] == "from_dotenv_api"
    assert os.environ["ALPACA_SECRET_KEY"] == "from_dotenv_secret"


def test_ensure_dotenv_loaded_prefers_env_runtime_for_blank_process_values(monkeypatch, tmp_path):
    _reset_env_module(monkeypatch)
    dotenv_file = tmp_path / ".env"
    runtime_file = tmp_path / ".env.runtime"
    dotenv_file.write_text("ALPACA_API_KEY=\nALPACA_SECRET_KEY=\n", encoding="utf-8")
    runtime_file.write_text(
        "ALPACA_API_KEY=runtime_api\nALPACA_SECRET_KEY=runtime_secret\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")

    env_mod.ensure_dotenv_loaded(str(dotenv_file))

    assert os.environ["ALPACA_API_KEY"] == "runtime_api"
    assert os.environ["ALPACA_SECRET_KEY"] == "runtime_secret"
