from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import ai_trading.app as app_module
from ai_trading.core import bot_engine
from ai_trading.core import runtime_services






def test_reconcile_positions_preserves_warn_state_across_calls(monkeypatch) -> None:
    warned_values: list[bool] = []

    def _fake_reconcile(ctx, *, logger, targets_lock, warned):  # noqa: ARG001
        warned_values.append(bool(warned))
        return True

    monkeypatch.setattr(
        bot_engine,
        "_reconcile_position_targets_service",
        _fake_reconcile,
    )
    monkeypatch.setattr(runtime_services, "_reconcile_warned", False)

    bot_engine.reconcile_positions(cast(Any, SimpleNamespace()))
    bot_engine.reconcile_positions(cast(Any, SimpleNamespace()))

    assert warned_values == [False, True]


def test_legacy_healthcheck_runtime_uses_canonical_health_only_app() -> None:
    legacy_app = runtime_services.build_healthcheck_app_runtime()
    canonical_app = app_module.create_app(health_only=True, fail_fast_env=True)

    legacy_routes = getattr(legacy_app, "_routes", None)
    canonical_routes = getattr(canonical_app, "_routes", None)

    assert isinstance(legacy_routes, dict)
    assert isinstance(canonical_routes, dict)
    assert set(legacy_routes) == set(canonical_routes)
