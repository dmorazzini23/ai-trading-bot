from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.execution_outcome import (
    SubmittedOrderState,
    normalize_submitted_order,
    record_successful_submission,
)


def test_normalize_submitted_order_extracts_fill_and_status() -> None:
    order = SimpleNamespace(
        status=SimpleNamespace(value="filled"),
        id="broker-1",
        filled_quantity="2",
        requested_quantity="3",
        filled_avg_price="101.5",
        filled_at="2026-04-19T14:31:00Z",
        total_fees="1.25",
    )

    result = normalize_submitted_order(
        order,
        delta_shares=3,
        extract_order_value=lambda obj, *fields: next(
            (getattr(obj, field, None) for field in fields if getattr(obj, field, None) is not None),
            None,
        ),
        extract_order_fill_timestamp=lambda obj: datetime.fromisoformat(
            str(getattr(obj, "filled_at")).replace("Z", "+00:00")
        ),
        normalize_order_status_token=lambda value: str(getattr(value, "value", value)).strip().lower(),
        safe_float=lambda value: float(value) if value is not None else None,
        has_persistable_fill=lambda **kwargs: bool(kwargs["fill_price"]) and kwargs["filled_qty"] > 0,
    )

    assert result.status_text == "filled"
    assert result.status_token == "filled"
    assert result.broker_order_id == "broker-1"
    assert result.filled_qty == 2.0
    assert result.requested_qty == 3.0
    assert result.fill_price == 101.5
    assert result.fill_timestamp is not None
    assert result.fill_timestamp.tzinfo is not None
    assert result.fill_fees == 1.25
    assert result.persistable_fill is True


def test_record_successful_submission_updates_state_and_turnover() -> None:
    recorded = []

    class _Ledger:
        def record(self, entry) -> None:
            recorded.append(entry)

    state = SimpleNamespace(
        last_order_bar_ts={},
        last_order_client_id={},
        turnover_dollars={},
    )
    proposal_a = SimpleNamespace(sleeve="day", target_dollars=100.0)
    proposal_b = SimpleNamespace(sleeve="swing", target_dollars=300.0)

    record_successful_submission(
        ledger=_Ledger(),
        state=state,
        symbol="AAPL",
        client_order_id="coid-aapl-1",
        bar_ts=datetime(2026, 4, 19, 14, 30, tzinfo=UTC),
        delta_shares=4,
        side="buy",
        price=50.0,
        now=datetime(2026, 4, 19, 14, 31, tzinfo=UTC),
        order_state=SubmittedOrderState(
            status_text="submitted",
            status_token="submitted",
            broker_order_id="broker-1",
            filled_qty=0.0,
            requested_qty=4.0,
            fill_price=None,
            fill_timestamp=None,
            fill_fees=0.0,
            persistable_fill=False,
        ),
        proposals=[proposal_a, proposal_b],
    )

    assert len(recorded) == 1
    assert state.last_order_client_id["AAPL"] == "coid-aapl-1"
    assert len(state.turnover_dollars) == 2
