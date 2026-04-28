from __future__ import annotations

from typing import Any, Callable

import ai_trading.health_payload as health_payload


def test_dedupe_flags_drops_blank_and_preserves_order() -> None:
    assert health_payload._dedupe_flags(
        [" service_degraded ", "", "provider_safe_mode", None, "service_degraded"]  # type: ignore[list-item]
    ) == ["service_degraded", "provider_safe_mode"]


def test_attention_flags_optional_contract_failures_are_opt_in() -> None:
    kwargs: dict[str, Any] = {
        "provider_state": {"using_backup": False, "safe_mode": False},
        "broker_state": {"open_orders_count": 0, "positions_count": 0},
        "service_state": {"status": "ready"},
        "database_readiness": {"configured": True, "ok": False},
        "oms_invariants": {"enabled": True, "ok": False},
        "oms_lifecycle_parity": {"enabled": True, "ok": False},
        "replay_live_parity_gate": {"enabled": True, "ok": False},
    }

    assert health_payload._build_runtime_attention_flags(**kwargs) == []

    flags = health_payload._build_runtime_attention_flags(
        **kwargs,
        include_optional_contract_failures=True,
    )

    assert flags == [
        "replay_live_parity_gate_failed",
        "database_unhealthy",
        "oms_invariants_failed",
        "oms_lifecycle_parity_failed",
    ]


def test_attention_flags_detect_market_closed_halt_and_trade_stream() -> None:
    flags = health_payload._build_runtime_attention_flags(
        provider_state={"reason": "market_closed", "using_backup": True, "safe_mode": True},
        broker_state={"open_orders_count": "2", "positions_count": "3"},
        service_state={"status": "halted", "reason": "trade_updates_stream_failed"},
    )

    assert flags == [
        "market_closed_non_flat_positions",
        "market_closed_open_orders",
        "provider_backup_active",
        "provider_safe_mode",
        "service_degraded",
        "service_halt_active",
        "trade_updates_stream_degraded",
    ]


def test_cached_background_snapshot_records_builder_error(monkeypatch) -> None:
    health_payload._HEALTH_SNAPSHOT_CACHE.clear()

    class _ImmediateThread:
        def __init__(self, *, target: Callable[[], None], name: str | None = None, daemon: bool | None = None):
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(health_payload, "Thread", _ImmediateThread)

    first = health_payload._cached_background_snapshot(
        name="broken_snapshot",
        ttl_seconds=30.0,
        placeholder={"enabled": True, "ok": False, "reason": "warming_up"},
        builder=lambda: (_ for _ in ()).throw(RuntimeError("database locked")),
    )
    second = health_payload._cached_background_snapshot(
        name="broken_snapshot",
        ttl_seconds=30.0,
        placeholder={"enabled": True, "ok": False, "reason": "warming_up"},
        builder=lambda: {"enabled": True, "ok": True},
    )

    assert first["refreshing"] is True
    assert second["available"] is False
    assert second["ok"] is False
    assert second["error"] == "database locked"
    assert second["refreshing"] is False


def test_build_health_json_response_handles_mapping_and_response_objects() -> None:
    payload = {"ok": False}

    assert health_payload.build_health_json_response(
        payload,
        200,
        jsonify_fn=lambda body: dict(body),
    ) == payload
    assert health_payload.build_health_json_response(
        payload,
        503,
        jsonify_fn=lambda body: dict(body),
    ) == (payload, 503)

    class _Response:
        status_code = 200

        @staticmethod
        def get_json() -> dict[str, bool]:
            return {"ok": False}

    response = health_payload.build_health_json_response(
        payload,
        503,
        jsonify_fn=lambda _body: _Response(),
    )

    assert response.status_code == 503


def test_build_health_json_response_sanitizes_unserializable_payload() -> None:
    bad_value = object()
    payload = {"ok": False, "nested": {"bad": bad_value}}

    response_payload = health_payload.build_health_json_response(
        payload,
        200,
        jsonify_fn=lambda body: dict(body),
    )

    assert response_payload == {"ok": False, "nested": {"bad": str(bad_value)}}


def test_register_healthz_routes_returns_exception_payload() -> None:
    calls: dict[str, Any] = {}

    class _App:
        def route(self, route: str, *, methods: list[str]):
            calls["route"] = route
            calls["methods"] = methods

            def _decorator(func):
                calls["handler"] = func
                return func

            return _decorator

    class _Logger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def exception(self, event: str, exc_info: BaseException | None = None) -> None:
            self.events.append(event)

    logger = _Logger()
    health_payload.register_healthz_routes(
        _App(),
        payload_builder=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        response_builder=lambda payload, status: (payload, status),
        logger=logger,
    )

    payload, status = calls["handler"]()

    assert calls["route"] == "/healthz"
    assert calls["methods"] == ["GET"]
    assert status == 200
    assert payload["status"] == "degraded"
    assert payload["ok"] is False
    assert payload["error"] == "boom"
    assert logger.events == ["HEALTH_CHECK_FAILED"]


def test_register_health_routes_response_fallback_is_non_500() -> None:
    calls: dict[str, Any] = {}

    class _App:
        def route(self, route: str, *, methods: list[str]):
            def _decorator(func):
                calls["handler"] = func
                return func

            return _decorator

    def _broken_response(_payload: dict[str, Any], _status: int) -> Any:
        raise RuntimeError("response busted")

    health_payload.register_health_routes(
        _App(),
        payload_builder=lambda: {"ok": True, "status": "healthy"},
        response_builder=_broken_response,
    )

    payload, status = calls["handler"]()

    assert status == 200
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "response busted"


def test_register_health_routes_sanitizes_payload_before_response() -> None:
    calls: dict[str, Any] = {}
    bad_value = object()

    class _App:
        def route(self, route: str, *, methods: list[str]):
            calls["route"] = route
            calls["methods"] = methods

            def _decorator(func):
                calls["handler_name"] = func.__name__
                calls["handler"] = func
                return func

            return _decorator

    health_payload.register_health_routes(
        _App(),
        payload_builder=lambda: {"ok": True, "nested": {"bad": bad_value}},
        response_builder=lambda payload, status: (payload, status),
        routes=("/health",),
    )

    payload, status = calls["handler"]()

    assert calls["route"] == "/health"
    assert calls["methods"] == ["GET"]
    assert calls["handler_name"] == "health"
    assert status == 200
    assert payload["nested"]["bad"] == str(bad_value)


def test_service_health_payload_env_error_forces_degraded() -> None:
    payload = health_payload.build_service_health_payload(
        force_ok_for_pytest=True,
        env_error="missing ALPACA_API_KEY",
        alpaca_context={
            "sdk_ok": True,
            "initialized": True,
            "client_attached": True,
            "has_key": True,
            "has_secret": True,
            "base_url": "https://paper-api.alpaca.markets",
            "paper": True,
            "shadow_mode": False,
        },
        enrich_alpaca_from_runtime_env=False,
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "missing ALPACA_API_KEY"
    assert payload["reason"]


def test_canonical_healthz_payload_env_error_forces_degraded() -> None:
    payload = health_payload.build_canonical_healthz_payload(
        force_ok_for_pytest=True,
        env_error="invalid HEALTHCHECK_PORT",
        alpaca_context={
            "sdk_ok": True,
            "initialized": True,
            "client_attached": True,
            "has_key": True,
            "has_secret": True,
            "base_url": "https://paper-api.alpaca.markets",
            "paper": True,
            "shadow_mode": False,
        },
        enrich_alpaca_from_runtime_env=False,
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "invalid HEALTHCHECK_PORT"
