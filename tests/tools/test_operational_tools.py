from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

from tools import check_python_version
from tools import fetch_csv


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_python_version_accepts_only_python_312() -> None:
    ok, message = check_python_version.check_version((3, 12, 3))
    assert ok is True
    assert "matches required 3.12" in message

    ok, message = check_python_version.check_version((3, 11, 9))
    assert ok is False
    assert "Python 3.12 is required" in message


def test_fetch_symbol_accepts_lowercase_close(monkeypatch, tmp_path: Path) -> None:
    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, *, period: str, interval: str) -> pd.DataFrame:
            assert period == "5d"
            assert interval == "1d"
            return pd.DataFrame({"close": [101.5, 102.0]})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch_csv, "yf", SimpleNamespace(Ticker=FakeTicker))

    path = fetch_csv.fetch_symbol("spy", period="5d", interval="1d")

    assert path == Path("data") / "SPY.csv"
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["date", "close"]
    assert saved["close"].tolist() == [101.5, 102.0]


def test_verify_config_requires_canonical_alpaca_base_url(monkeypatch, tmp_path: Path) -> None:
    verify_config = _load_script("verify_config_under_test", "scripts/verify_config.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ALPACA_API_KEY=PK" + "A" * 20,
                "ALPACA_SECRET_KEY=SK" + "B" * 32,
                "ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets",
            ]
        ),
        encoding="utf-8",
    )
    values = {
        "ALPACA_API_KEY": "PK" + "A" * 20,
        "ALPACA_SECRET_KEY": "SK" + "B" * 32,
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
    }
    monkeypatch.setattr(verify_config, "reload_env", lambda *args, **kwargs: True)
    monkeypatch.setattr(verify_config, "get_env", lambda key: values.get(key))

    env_ok, env_message = verify_config.check_env_file()
    api_ok, api_message = verify_config.check_api_keys()

    assert env_ok is True, env_message
    assert api_ok is True, api_message
    assert "Paper Trading" in api_message


def test_health_check_requires_canonical_alpaca_base_url(monkeypatch) -> None:
    health_check = _load_script("health_check_under_test", "scripts/health_check.py")
    monitor = health_check.HealthMonitor()
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)

    result = monitor._check_environment_variables()

    assert result.status is health_check.HealthStatus.CRITICAL
    assert "ALPACA_TRADING_BASE_URL" in result.message
    assert "ALPACA_BASE_URL" not in result.message
