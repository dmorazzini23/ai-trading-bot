import types

from ai_trading import main


class DummyCfg:
    def __init__(self, token: int) -> None:
        self.token = token

    def to_dict(self) -> dict[str, int]:
        return {"token": self.token}


def test_resolve_cached_context_reuses_instances(monkeypatch):
    monkeypatch.setattr(main, "_STATE_CACHE", None, raising=False)
    monkeypatch.setattr(main, "_RUNTIME_CACHE", None, raising=False)
    monkeypatch.setattr(main, "_RUNTIME_CFG_SNAPSHOT", None, raising=False)

    def state_factory():
        return types.SimpleNamespace(state={})

    def runtime_builder(cfg):
        return types.SimpleNamespace(cfg=cfg, state={})

    cfg1 = DummyCfg(1)
    state1, runtime1, reused1 = main._resolve_cached_context(cfg1, state_factory, runtime_builder)
    assert reused1 is False
    runtime1.state["pending"] = True

    cfg2 = DummyCfg(1)
    state2, runtime2, reused2 = main._resolve_cached_context(cfg2, state_factory, runtime_builder)
    assert reused2 is True
    assert state2 is state1
    assert runtime2 is runtime1
    assert runtime2.state["pending"] is True
    assert runtime2.cfg is cfg2

    cfg3 = DummyCfg(2)
    state3, runtime3, reused3 = main._resolve_cached_context(cfg3, state_factory, runtime_builder)
    assert reused3 is False
    assert state3 is not state2
    assert runtime3 is not runtime2
    assert runtime3.cfg is cfg3


def test_auto_sizing_rebuild_keeps_runtime_cache_stable(monkeypatch):
    """AUTO sizing must not mutate cfg, or every cycle rebuilds runtime/state."""

    import itertools

    from ai_trading.config.management import TradingConfig
    from ai_trading.core import bot_engine
    from ai_trading.core import runtime as core_runtime
    from ai_trading.position_sizing import _CACHE as position_sizing_cache

    monkeypatch.setattr(main, "_STATE_CACHE", None, raising=False)
    monkeypatch.setattr(main, "_RUNTIME_CACHE", None, raising=False)
    monkeypatch.setattr(main, "_RUNTIME_CFG_SNAPSHOT", None, raising=False)
    monkeypatch.setattr(position_sizing_cache, "value", position_sizing_cache.value)
    monkeypatch.setattr(position_sizing_cache, "ts", position_sizing_cache.ts)
    monkeypatch.setattr(bot_engine, "MAX_POSITION_SIZE", bot_engine.MAX_POSITION_SIZE)
    # Each resolution carries a new refreshed_at, as the broker-backed path does.
    ticks = itertools.count()
    monkeypatch.setattr(
        core_runtime,
        "resolve_max_position_size",
        lambda *_a, **_k: (
            3266.0,
            {"mode": "AUTO", "source": "alpaca", "refreshed_at": f"t{next(ticks)}"},
        ),
    )
    env = {
        "MAX_DRAWDOWN_THRESHOLD": "0.15",
        "EXECUTION_MODE": "paper",
        "AI_TRADING_CAPITAL_CAP": "0.04",
        "MAX_POSITION_MODE": "AUTO",
        "AI_TRADING_SIGNAL_MAX_POSITION_SIZE": "3000",
    }
    builds: list[object] = []

    def runtime_builder(cfg):
        runtime = core_runtime.build_runtime(cfg)
        builds.append(runtime)
        return runtime

    # Pre-open cycles pass the same cached cfg object each minute.
    cached_cfg = TradingConfig.from_env(env)
    state1, runtime1, reused1 = main._resolve_cached_context(
        cached_cfg, bot_engine.BotState, runtime_builder
    )
    state2, runtime2, reused2 = main._resolve_cached_context(
        cached_cfg, bot_engine.BotState, runtime_builder
    )
    # The first session cycle passes a fresh cfg built from the same env.
    state3, runtime3, reused3 = main._resolve_cached_context(
        TradingConfig.from_env(env), bot_engine.BotState, runtime_builder
    )

    assert (reused1, reused2, reused3) == (False, True, True)
    assert state3 is state2 is state1
    assert runtime3 is runtime1
    assert len(builds) == 1
    assert runtime1.params["MAX_POSITION_SIZE"] == 3266.0
    assert cached_cfg.max_position_size == 3000.0
