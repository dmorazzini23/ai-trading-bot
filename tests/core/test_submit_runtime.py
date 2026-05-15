from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.paper_sampling import reserve_paper_sampling_order


class _DummyExecEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.skips: list[dict[str, object]] = []
        self._last_submit_outcome: dict[str, object] = {}

    def execute_order(
        self,
        symbol: str,
        side: object,
        qty: int,
        *,
        price: float | None = None,
        **kwargs: object,
    ) -> object:
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "kwargs": dict(kwargs),
            }
        )
        return SimpleNamespace(id="broker-order-1")

    def _skip_submit(self, **kwargs: object) -> None:
        payload = dict(kwargs)
        self.skips.append(payload)
        self._last_submit_outcome = {
            "status": "skipped",
            "reason": payload.get("reason"),
            "detail": payload.get("detail"),
        }


def _reset_submit_state(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine.state, "running", False, raising=False)
    monkeypatch.setattr(bot_engine.state, "cycle_submit_compaction", set(), raising=False)
    monkeypatch.setattr(bot_engine.state, "pretrade_rate_limiter", None, raising=False)
    monkeypatch.setattr(bot_engine.state, "dependency_breakers", None, raising=False)
    monkeypatch.setattr(bot_engine.state, "auth_forbidden_cooldowns", {}, raising=False)
    setattr(bot_engine.state, "_oms_ledger", None)


def _base_cfg(**overrides: object) -> SimpleNamespace:
    payload = {
        "seed": "phase2-seed",
        "rth_only": False,
        "allow_extended": True,
        "quote_max_age_ms": 0,
        "execution_require_realtime_nbbo": False,
        "ledger_enabled": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_submit_order_non_netting_path_uses_shared_pretrade_block(monkeypatch) -> None:
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: []))
    stale_quote_ts = datetime.now(UTC) - timedelta(seconds=30)

    _reset_submit_state(monkeypatch)
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(
        bot_engine,
        "_resolve_trading_config",
        lambda _ctx: _base_cfg(quote_max_age_ms=5),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            99.0,
            101.0,
            100.0,
            100.0,
            stale_quote_ts,
        ),
    )

    order = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)

    assert order is None
    assert engine.calls == []
    assert engine.skips
    assert engine.skips[-1]["reason"] == "STALE_QUOTE_BLOCK"


def test_submit_order_records_auth_forbidden_cooldown_on_none_result(monkeypatch) -> None:
    class _ForbiddenExecEngine(_DummyExecEngine):
        def execute_order(
            self,
            symbol: str,
            side: object,
            qty: int,
            *,
            price: float | None = None,
            **kwargs: object,
        ) -> object | None:
            self.calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "kwargs": dict(kwargs),
                }
            )
            self._last_submit_outcome = {
                "status": "skipped",
                "reason": "AUTH_BROKER_HALT_FORBIDDEN_PROVIDER",
            }
            return None

    engine = _ForbiddenExecEngine()
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: []))
    fresh_quote_ts = datetime.now(UTC)

    _reset_submit_state(monkeypatch)
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: _base_cfg())
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            99.99,
            100.01,
            100.0,
            100.0,
            fresh_quote_ts,
        ),
    )

    first = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)
    second = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)

    assert first is None
    assert second is None
    assert len(engine.calls) == 1
    assert engine.skips
    assert engine.skips[-1]["reason"] == "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN"
    cooldowns = getattr(bot_engine.state, "auth_forbidden_cooldowns", {})
    assert ("AAPL", "buy") in cooldowns


def test_submit_order_does_not_record_rejected_broker_object(monkeypatch) -> None:
    class _RejectedExecEngine(_DummyExecEngine):
        def execute_order(
            self,
            symbol: str,
            side: object,
            qty: int,
            *,
            price: float | None = None,
            **kwargs: object,
        ) -> object:
            self.calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(id="broker-order-rejected", status="rejected")

    recorded: list[object] = []

    class _Ledger:
        def record(self, entry: object) -> None:
            recorded.append(entry)

    engine = _RejectedExecEngine()
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: []))

    _reset_submit_state(monkeypatch)
    setattr(bot_engine.state, "_oms_ledger", _Ledger())
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: _base_cfg(ledger_enabled=True))
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            99.9,
            100.1,
            100.0,
            100.1,
            datetime.now(UTC),
        ),
    )

    order = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)

    assert order is None
    assert len(engine.calls) == 1
    assert recorded == []


def test_submit_order_propagates_generated_identity_to_execution_engine(monkeypatch) -> None:
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: []))
    fresh_quote_ts = datetime.now(UTC)

    _reset_submit_state(monkeypatch)
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: _base_cfg())
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            99.99,
            100.01,
            100.0,
            100.0,
            fresh_quote_ts,
        ),
    )

    order = bot_engine.submit_order(
        ctx,
        "AAPL",
        2,
        "buy",
        price=100.0,
        annotations={"strategy_label": "non-netting-phase2"},
    )

    assert order is not None
    assert engine.calls
    kwargs = engine.calls[0]["kwargs"]
    assert isinstance(kwargs, dict)
    assert isinstance(kwargs.get("client_order_id"), str)
    assert kwargs["client_order_id"]
    assert isinstance(kwargs.get("decision_trace_id"), str)
    assert kwargs["decision_trace_id"]
    assert kwargs["metadata"]["decision_trace_id"] == kwargs["decision_trace_id"]
    assert kwargs["annotations"]["strategy_label"] == "non-netting-phase2"
    assert getattr(order, "client_order_id", None) == kwargs["client_order_id"]


def test_submit_order_ignores_stale_ledger_when_disabled(monkeypatch) -> None:
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(market_data=None, api=SimpleNamespace(list_positions=lambda: []))
    fresh_quote_ts = datetime.now(UTC)

    _reset_submit_state(monkeypatch)
    setattr(bot_engine.state, "_oms_ledger", object())
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: _base_cfg())
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            99.99,
            100.01,
            100.0,
            100.0,
            fresh_quote_ts,
        ),
    )

    first = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)
    second = bot_engine.submit_order(ctx, "AAPL", 1, "buy", price=100.0)

    assert first is not None
    assert second is not None
    assert len(engine.calls) == 2
    assert getattr(bot_engine.state, "_oms_ledger", None) is None


def test_submit_order_paper_sampling_cap_does_not_block_reducing_sell(
    monkeypatch,
    tmp_path,
) -> None:
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(
        market_data=None,
        api=SimpleNamespace(list_positions=lambda: []),
        position_map={"AMZN": SimpleNamespace(qty="1", side="long")},
    )
    fresh_quote_ts = datetime.now(UTC)

    cfg = _base_cfg(
        paper_sampling_enabled=True,
        paper_sampling_allowed_symbols=("AAPL", "AMZN"),
        paper_sampling_max_trades_per_day=1,
        paper_sampling_max_notional_per_order=350.0,
        execution_mode="paper",
        paper=True,
        alpaca_base_url="https://paper-api.alpaca.markets",
        launch_profile="paper_trade",
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    _reset_submit_state(monkeypatch)
    reserve_paper_sampling_order(
        cfg,
        symbol="AMZN",
        side="buy",
        qty=1,
        price=100.0,
        now=datetime(2026, 5, 14, 13, 35, tzinfo=UTC),
    )
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: cfg)
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "broker_nbbo",
            267.3,
            267.34,
            267.32,
            267.32,
            fresh_quote_ts,
        ),
    )

    order = bot_engine.submit_order(ctx, "AMZN", 1, "sell", price=267.32)

    assert order is not None
    assert engine.calls
    assert engine.skips == []
    kwargs = engine.calls[0]["kwargs"]
    assert kwargs["closing_position"] is True
    assert kwargs["reduce_only"] is True
    assert kwargs["annotations"]["paper_sampling_consumes_daily_slot"] is False
    state_path = resolve_runtime_artifact_path(
        "runtime/paper_sampling_state_latest.json",
        default_relative="runtime/paper_sampling_state_latest.json",
    )
    assert state_path.read_text(encoding="utf-8").count('"count":1') == 1


def test_submit_order_defaults_opening_nbbo_requirement_when_cfg_missing(monkeypatch) -> None:
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(
        market_data=None,
        api=SimpleNamespace(list_positions=lambda: []),
        execution_mode="live",
    )
    fresh_quote_ts = datetime.now(UTC)

    _reset_submit_state(monkeypatch)
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", "1")
    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(
        bot_engine,
        "_resolve_trading_config",
        lambda _ctx: SimpleNamespace(
            seed="phase2-seed",
            rth_only=False,
            allow_extended=True,
            quote_max_age_ms=0,
            ledger_enabled=False,
        ),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote_basis",
        lambda *_args, **_kwargs: (
            "yahoo",
            99.99,
            100.01,
            100.0,
            100.0,
            fresh_quote_ts,
        ),
    )

    order = bot_engine.submit_order(
        ctx,
        "AAPL",
        1,
        "buy",
        price=100.0,
        opening_trade=True,
    )

    assert order is None
    assert engine.calls == []
    assert engine.skips
    assert engine.skips[-1]["reason"] == "NBBO_REQUIRED_OPENING_SKIP"
