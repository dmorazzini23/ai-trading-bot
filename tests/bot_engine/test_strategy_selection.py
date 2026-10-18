from ai_trading.core import bot_engine


def test_get_strategies_from_env(monkeypatch):
    class Dummy:
        name = "dummy"

        def __init__(self):
            pass

    class Alt:
        name = "alt"

        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"dummy": Dummy, "alt": Alt}, raising=False)
    monkeypatch.setenv("STRATEGIES", "dummy,alt")
    strategies_list = bot_engine.get_strategies()
    assert {type(s) for s in strategies_list} == {Dummy, Alt}


def test_get_strategies_defaults_to_momentum(monkeypatch):
    class Momentum:
        name = "momentum"

        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"momentum": Momentum}, raising=False)
    monkeypatch.delenv("STRATEGIES", raising=False)
    strategies_list = bot_engine.get_strategies()
    assert len(strategies_list) == 1
    assert isinstance(strategies_list[0], Momentum)


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def _record(self, level: str, msg: str, *args, **kwargs) -> None:
        self.records.append((level, msg, dict(kwargs.get("extra") or {})))

    def info(self, msg, *args, **kwargs):
        self._record("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._record("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._record("error", msg, *args, **kwargs)

    def messages(self, level: str) -> list[str]:
        return [msg for lvl, msg, _ in self.records if lvl == level]


def _install_recorders(monkeypatch, *, netting_enabled: bool):
    from types import SimpleNamespace

    from ai_trading.config import settings as settings_module

    # Other tests mutate the cached settings object; start from a clean read.
    settings_module.get_settings.cache_clear()
    log = _RecordingLogger()
    once = _RecordingLogger()
    monkeypatch.setattr(bot_engine, "logger", log)
    monkeypatch.setattr(bot_engine, "logger_once", once)
    monkeypatch.setattr(
        bot_engine,
        "get_trading_config",
        lambda: SimpleNamespace(netting_enabled=netting_enabled),
    )
    return log, once


def test_get_strategies_errors_when_no_requested_name_resolves(monkeypatch):
    class Real:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"real": Real}, raising=False)
    monkeypatch.setenv("STRATEGIES", "bogus,also_bogus")
    log, once = _install_recorders(monkeypatch, netting_enabled=True)

    assert bot_engine.get_strategies() == []

    errors = [rec for rec in log.records if rec[0] == "error"]
    assert len(errors) == 1
    _, msg, extra = errors[0]
    assert msg == "STRATEGIES_NONE_RESOLVED"
    assert extra["unknown"] == ["bogus", "also_bogus"]
    assert extra["registry_keys"] == ["real"]
    # Nothing loaded, so no "not consumed" warning either.
    assert once.messages("warning") == []


def test_get_strategies_does_not_fall_back_to_momentum(monkeypatch):
    class Momentum:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"momentum": Momentum}, raising=False)
    monkeypatch.setenv("STRATEGIES", "bogus")
    log, _ = _install_recorders(monkeypatch, netting_enabled=False)

    assert bot_engine.get_strategies() == []
    assert log.messages("error") == ["STRATEGIES_NONE_RESOLVED"]


def test_get_strategies_all_resolved_loads_without_error(monkeypatch):
    class Alpha:
        def __init__(self):
            pass

    class Beta:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(
        strategies, "REGISTRY", {"alpha": Alpha, "beta": Beta}, raising=False
    )
    monkeypatch.setenv("STRATEGIES", "alpha,beta")
    log, once = _install_recorders(monkeypatch, netting_enabled=False)

    loaded = bot_engine.get_strategies()

    assert [type(s) for s in loaded] == [Alpha, Beta]
    assert log.messages("error") == []
    assert log.messages("warning") == []
    assert once.messages("info") == ["Loaded strategies: Alpha, Beta"]
    assert once.messages("warning") == []


def test_get_strategies_partial_resolution_keeps_resolved(monkeypatch):
    class Alpha:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"alpha": Alpha}, raising=False)
    monkeypatch.setenv("STRATEGIES", "alpha,bogus")
    log, _ = _install_recorders(monkeypatch, netting_enabled=False)

    loaded = bot_engine.get_strategies()

    assert [type(s) for s in loaded] == [Alpha]
    assert log.messages("error") == []
    assert len(log.messages("warning")) == 1


def test_get_strategies_warns_once_when_netting_enabled(monkeypatch):
    class Alpha:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"alpha": Alpha}, raising=False)
    monkeypatch.setenv("STRATEGIES", "alpha")
    log, once = _install_recorders(monkeypatch, netting_enabled=True)

    loaded = bot_engine.get_strategies()

    assert [type(s) for s in loaded] == [Alpha]
    warnings = [rec for rec in once.records if rec[0] == "warning"]
    assert len(warnings) == 1
    _, msg, extra = warnings[0]
    assert msg == "STRATEGIES_NOT_CONSUMED_UNDER_NETTING"
    assert extra["governing_flag"] == "AI_TRADING_NETTING_ENABLED"
    assert extra["strategies"] == ["Alpha"]
    assert log.messages("error") == []


def test_get_strategies_unset_env_is_quiet_when_default_unregistered(monkeypatch):
    class Alpha:
        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"alpha": Alpha}, raising=False)
    monkeypatch.delenv("STRATEGIES", raising=False)
    log, once = _install_recorders(monkeypatch, netting_enabled=True)

    assert bot_engine.get_strategies() == []
    assert log.messages("error") == []
    assert log.messages("warning") == []
    assert once.messages("warning") == []
