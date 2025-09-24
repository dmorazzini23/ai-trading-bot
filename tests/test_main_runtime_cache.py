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
