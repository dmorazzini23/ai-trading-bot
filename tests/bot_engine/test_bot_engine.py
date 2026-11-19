import sys
import types
from pathlib import Path

from typing import Any, cast


if "numpy" not in sys.modules:  # pragma: no cover - optional dependency shim
    numpy_stub = cast(Any, types.ModuleType("numpy"))
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = numpy_stub.nan
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["portalocker"] = cast(Any, types.ModuleType("portalocker"))

if "bs4" not in sys.modules:  # pragma: no cover - optional dependency shim
    bs4_stub = cast(Any, types.ModuleType("bs4"))

    class _BeautifulSoup:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "flask" not in sys.modules:  # pragma: no cover - optional dependency shim
    flask_stub = cast(Any, types.ModuleType("flask"))

    class _Flask:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def route(self, *_a, **_k):  # noqa: ANN001 - placeholder
            def decorator(func):
                return func

            return decorator

    flask_stub.Flask = _Flask
    sys.modules["flask"] = flask_stub

from ai_trading.core import bot_engine


def test_get_trade_logger_uses_fallback_when_primary_unwritable(monkeypatch, tmp_path):
    primary_dir = tmp_path / "primary"
    fallback_dir = tmp_path / "fallback"
    primary_dir.mkdir()
    fallback_dir.mkdir()
    primary_file = primary_dir / "trades.csv"
    fallback_path = fallback_dir / "trades.csv"

    constructed_paths: list[str] = []
    primary_dir_resolved = primary_dir.resolve()

    def fake_is_dir_writable(path: str) -> bool:
        try:
            resolved = Path(path).resolve()
        except OSError:
            resolved = Path(path)
        if resolved == primary_dir_resolved:
            return False
        return True

    class _StubTradeLogger:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)
            constructed_paths.append(self.path)

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)
    monkeypatch.setattr(bot_engine, "TradeLogger", _StubTradeLogger)
    monkeypatch.setattr(
        bot_engine,
        "_compute_user_state_trade_log_path",
        lambda filename="trades.csv": str(fallback_path),
    )
    monkeypatch.setattr(bot_engine, "_TRADE_LOGGER_SINGLETON", None, raising=False)
    monkeypatch.setattr(bot_engine, "_TRADE_LOG_FALLBACK_PATH", None, raising=False)
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(primary_file), raising=False)

    logger = bot_engine.get_trade_logger()

    assert constructed_paths[0] == str(primary_file)
    assert constructed_paths[-1] == str(fallback_path)
    assert logger.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(fallback_path)


def test_get_latest_price_skips_primary_during_cooldown(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)

    def _fail_alpaca_symbols():  # pragma: no cover - should not be invoked
        raise AssertionError("Alpaca should not be queried during cooldown")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)

    price = bot_engine.get_latest_price("AAPL")

    assert price is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == bot_engine._ALPACA_DISABLED_SENTINEL


def test_get_latest_price_prefers_backup_when_primary_disabled(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

    def _fail_alpaca_symbols():  # pragma: no cover - should not be invoked
        raise AssertionError("Alpaca should not be queried when disabled")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)

    called: dict[str, int] = {"backup": 0}

    def _fake_backup_get_bars(symbol, start, end, interval):  # noqa: D401, ANN001
        called["backup"] += 1
        return object()

    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_backup_get_bars",
        _fake_backup_get_bars,
    )
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda _df: 123.45)

    price = bot_engine.get_latest_price("AAPL", prefer_backup=True)

    assert price == 123.45
    assert called["backup"] == 1
    assert bot_engine._PRICE_SOURCE["AAPL"] == "yahoo"


