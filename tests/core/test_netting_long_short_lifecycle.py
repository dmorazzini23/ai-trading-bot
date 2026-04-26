from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core.netting_submit_execution import execute_netting_submission
from ai_trading.core.netting_symbol_approval import prepare_netting_symbol_approval
from ai_trading.oms import decision_events
from tests.test_netting_submit_execution import _base_kwargs as _submit_base_kwargs
from tests.test_netting_symbol_approval import _base_kwargs as _approval_base_kwargs


@pytest.mark.parametrize(
    (
        "case",
        "current_shares",
        "requested_delta",
        "expected_side",
        "expected_delta",
        "expected_target",
        "expected_opening",
        "expected_action",
        "expected_gate",
    ),
    [
        ("flat_to_long", 0, 4, "buy", 4, 4, True, "BUY", None),
        ("flat_to_short", 0, -4, "sell_short", -4, -4, True, "SELL", None),
        ("long_to_flat", 5, -5, "sell", -5, 0, False, "SELL", None),
        (
            "long_to_short_closes_first",
            5,
            -10,
            "sell",
            -5,
            0,
            False,
            "SELL",
            "PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION",
        ),
        ("short_to_flat", -5, 5, "buy", 5, 0, False, "BUY", None),
        (
            "short_to_long_covers_first",
            -5,
            10,
            "buy",
            5,
            0,
            False,
            "BUY",
            "PRE_SUBMIT_BUY_QTY_CLIP_SHORT_COVER",
        ),
    ],
)
def test_netting_long_short_lifecycle_approval_submission_and_decision_action(
    case: str,
    current_shares: int,
    requested_delta: int,
    expected_side: str,
    expected_delta: int,
    expected_target: int,
    expected_opening: bool,
    expected_action: str,
    expected_gate: str | None,
) -> None:
    approval_kwargs = _approval_base_kwargs()
    approval_kwargs["current_shares"] = current_shares
    approval_kwargs["delta_shares"] = requested_delta
    approval_kwargs["exec_engine"] = SimpleNamespace()
    approval_kwargs["clip_sell_qty_to_available_position_func"] = lambda **clip_kwargs: (
        min(int(clip_kwargs["requested_qty"]), max(int(clip_kwargs["current_shares"]), 0)),
        {"available": max(int(clip_kwargs["current_shares"]), 0), "case": case},
    )

    approval_result = prepare_netting_symbol_approval(**cast(Any, approval_kwargs))

    assert approval_result.blocked_reason is None
    assert approval_result.side == expected_side
    assert approval_result.delta_shares == expected_delta
    assert approval_result.target_shares == expected_target
    assert approval_result.opening_trade is expected_opening
    if expected_gate is not None:
        assert expected_gate in approval_result.gates_added

    submitted: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []
    tca_inputs: list[dict[str, Any]] = []
    submit_kwargs = _submit_base_kwargs()
    submit_kwargs.update(
        {
            "side": approval_result.side,
            "delta_shares": approval_result.delta_shares,
            "approval": approval_result.approval,
            "intent": SimpleNamespace(
                to_contract=lambda: {
                    "symbol": "AAPL",
                    "side": approval_result.side,
                    "qty": abs(approval_result.delta_shares),
                }
            ),
            "client_order_id": f"cid-{case}",
        }
    )

    def _submit_order(_runtime: Any, symbol: str, qty: int, side: str, **kwargs: Any) -> Any:
        submitted.append({"symbol": symbol, "qty": qty, "side": side, **kwargs})
        return SimpleNamespace(status="filled", id=f"order-{case}", qty=qty)

    def _record_success(**kwargs: Any) -> None:
        recorded.append(dict(kwargs))

    def _build_metrics_and_tca(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        tca_inputs.append(dict(kwargs))
        return {"side": kwargs["side"]}, {"side": kwargs["side"], "delta": kwargs["delta_shares"]}

    submit_kwargs["submit_order_func"] = _submit_order
    submit_kwargs["record_successful_submission_func"] = _record_success
    submit_kwargs["build_order_metrics_and_tca_func"] = _build_metrics_and_tca

    submit_result = execute_netting_submission(**cast(Any, submit_kwargs))

    assert submit_result.status == "submitted"
    assert submitted[-1]["side"] == expected_side
    assert submitted[-1]["qty"] == abs(expected_delta)
    assert recorded[-1]["side"] == expected_side
    assert recorded[-1]["delta_shares"] == expected_delta
    assert tca_inputs[-1]["side"] == expected_side
    assert tca_inputs[-1]["delta_shares"] == expected_delta
    assert submit_result.order_payload is not None
    assert submit_result.order_payload["side"] == expected_side
    assert submit_result.order_payload["qty"] == abs(expected_delta)

    action = decision_events._decision_action(
        {
            "gates": ["OK_TRADE"],
            "order": {"side": expected_side, "qty": abs(expected_delta)},
        }
    )
    assert action == expected_action
