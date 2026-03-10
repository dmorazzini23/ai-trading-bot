from __future__ import annotations

import types

from ai_trading import main


class _DummyMetric:
    def labels(self, *args, **kwargs):  # noqa: D401 - test stub
        return self

    def observe(self, *args, **kwargs):  # noqa: D401 - test stub
        return None

    def inc(self, *args, **kwargs):  # noqa: D401 - test stub
        return None


class _InlineThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False


def test_coerce_int_like_accepts_float_string() -> None:
    assert main._coerce_int_like("5.0", 60) == 5
    assert main._coerce_int_like("bad", 60) == 60


def test_main_uses_float_interval_cli_value(monkeypatch) -> None:
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "1")
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "1")

    settings = types.SimpleNamespace(
        alpaca_data_feed="iex",
        alpaca_adjustment="raw",
        iterations=None,
        scheduler_iterations=0,
        interval=60,
        interval_when_closed=300,
        api_port=9001,
        api_port_wait_seconds=0,
        health_tick_seconds=60,
        max_position_mode="STATIC",
        max_position_size=1.0,
        http_connect_timeout=1.0,
        http_read_timeout=1.0,
        http_pool_maxsize=1,
        http_total_retries=1,
        http_backoff_factor=0.1,
        alpaca_base_url="https://paper-api.example.com",
        capital_cap=0.1,
        dollar_risk_limit=0.1,
        paper=True,
        trading_mode="balanced",
    )

    monkeypatch.setattr(main, "_check_alpaca_sdk", lambda: None)
    monkeypatch.setattr(main, "_fail_fast_env", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "_assert_singleton_api", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_init_http_session", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "ensure_trade_log_path", lambda: None)
    monkeypatch.setattr(main, "preflight_import_health", lambda: None)
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)
    monkeypatch.setattr(main, "optimize_memory", lambda: {})
    monkeypatch.setattr(main, "resolve_max_position_size", lambda *a, **k: (1.0, {}))
    monkeypatch.setattr(main, "get_histogram", lambda *a, **k: _DummyMetric())
    monkeypatch.setattr(main, "get_counter", lambda *a, **k: _DummyMetric())
    monkeypatch.setattr(main, "start_api_with_signal", lambda ready, _err: ready.set())
    monkeypatch.setattr(main, "Thread", _InlineThread)
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "run_cycle", lambda: None)

    sleep_calls: list[int] = []
    monkeypatch.setattr(main, "_interruptible_sleep", lambda seconds: sleep_calls.append(int(seconds)))

    main.main(["--iterations", "1", "--interval", "5.0"])

    assert 5 in sleep_calls
