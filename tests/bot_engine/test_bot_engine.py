import sys
import types

import pytest

if "numpy" not in sys.modules:  # pragma: no cover - optional dependency shim
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = numpy_stub.nan
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["portalocker"] = types.ModuleType("portalocker")

if "bs4" not in sys.modules:  # pragma: no cover - optional dependency shim
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "flask" not in sys.modules:  # pragma: no cover - optional dependency shim
    flask_stub = types.ModuleType("flask")

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


class DummyHaltManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def manual_halt_trading(self, reason: str) -> None:
        self.calls.append(reason)


class DummyExecutor:
    def submit(self, fn, symbol):  # noqa: ANN001 - test helper
        return types.SimpleNamespace(result=lambda: fn(symbol))


class DummyContext:
    def __init__(self, halt_manager):
        self.halt_manager = halt_manager


class TestProcessSymbol:
    @pytest.fixture(autouse=True)
    def _patch_env(self, monkeypatch):
        monkeypatch.setattr(bot_engine, "get_env", lambda key, default=None: default, raising=False)

    def test_process_symbol_skips_on_all_nan_close(self, monkeypatch):
        state = bot_engine.BotState()
        state.position_cache = {}
        bot_engine.state = state

        dummy_halt = DummyHaltManager()
        dummy_ctx = DummyContext(dummy_halt)
        monkeypatch.setattr(bot_engine, "get_ctx", lambda: dummy_ctx)
        monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
        monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda *_, **__: True)
        monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *_, **__: None)
        monkeypatch.setattr(
            bot_engine,
            "skipped_duplicates",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(
            bot_engine,
            "skipped_cooldown",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)
        monkeypatch.setattr(bot_engine, "prediction_executor", DummyExecutor(), raising=False)
        monkeypatch.setattr(bot_engine, "_safe_trade", lambda *_, **__: None)

        def _raise_nan(_symbol: str):
            err = bot_engine.DataFetchError("close_column_all_nan")
            setattr(err, "fetch_reason", "close_column_all_nan")
            raise err

        monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", _raise_nan)

        processed, row_counts = bot_engine._process_symbols(["AAPL"], 1000.0, None, True)

        assert processed == []
        assert row_counts == {}
        assert dummy_halt.calls == ["AAPL:empty_frame"]
