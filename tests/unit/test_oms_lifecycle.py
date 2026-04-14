from __future__ import annotations

import pytest

from ai_trading.oms.lifecycle import (
    normalize_terminal_status,
    resolve_terminal_intent_status,
    status_for_fill,
    status_for_submit_ack,
    status_for_submit_claim,
    status_for_submit_error,
    terminal_event_type,
)


def test_lifecycle_submit_statuses_are_normalized() -> None:
    assert status_for_submit_claim() == "SUBMITTING"
    assert status_for_submit_ack() == "SUBMITTED"
    assert status_for_submit_error() == "PENDING_SUBMIT"


def test_status_for_fill_preserves_filled_and_closed() -> None:
    assert status_for_fill("FILLED") == "FILLED"
    assert status_for_fill("CLOSED") == "CLOSED"
    assert status_for_fill("SUBMITTED") == "PARTIALLY_FILLED"


def test_terminal_event_type_maps_terminal_statuses() -> None:
    assert terminal_event_type("FILLED") == "ORDER_FILLED"
    assert terminal_event_type("CANCELED") == "ORDER_CANCELED"
    assert terminal_event_type("FAILED") == "ORDER_FAILED"
    assert terminal_event_type("CLOSED") == "INTENT_CLOSED"


def test_normalize_terminal_status_rejects_non_terminal() -> None:
    with pytest.raises(ValueError):
        normalize_terminal_status("SUBMITTED")


def test_resolve_terminal_intent_status_maps_broker_aliases() -> None:
    assert resolve_terminal_intent_status(status="done_for_day") == "EXPIRED"
    assert resolve_terminal_intent_status(status="replaced") == "CLOSED"
    assert resolve_terminal_intent_status(status="accepted") is None
    assert resolve_terminal_intent_status(event_type="canceled") == "CANCELED"
