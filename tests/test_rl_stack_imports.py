"""Tests for RL stack optional dependency handling."""

from __future__ import annotations

import types

import ai_trading.rl_trading as rl


def test_is_rl_unavailable_when_torch_raises_name_error(monkeypatch):
    """Torch import errors should keep the RL facade in stub mode."""

    rl._load_rl_stack.cache_clear()

    dummy_vec_env = object()
    dummy_sb3 = types.SimpleNamespace(
        PPO=object(),
        common=types.SimpleNamespace(
            vec_env=types.SimpleNamespace(DummyVecEnv=dummy_vec_env)
        ),
    )
    dummy_gym = types.ModuleType("gymnasium")

    def fake_import(name: str, package: str | None = None):
        if name == "stable_baselines3":
            return dummy_sb3
        if name == "gymnasium":
            return dummy_gym
        if name == "torch":
            raise NameError("_C")
        raise AssertionError(f"unexpected import request: {name}")

    monkeypatch.setattr(rl.importlib, "import_module", fake_import)

    try:
        assert rl._load_rl_stack() is None
        assert rl.is_rl_available() is False
    finally:
        rl._load_rl_stack.cache_clear()
