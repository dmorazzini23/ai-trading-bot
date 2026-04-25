from __future__ import annotations

from types import SimpleNamespace

import pytest

import ai_trading.timeframe as timeframe


def _unit_name(value: object) -> str:
    return str(getattr(value, "name", value))


def test_canonicalize_timeframe_accepts_existing_numeric_and_string_values() -> None:
    unit_cls = timeframe._resolve_timeframe_unit_cls()

    existing = timeframe.TimeFrame(2, getattr(unit_cls, "Hour"))

    assert timeframe.canonicalize_timeframe(existing) is existing
    assert getattr(timeframe.canonicalize_timeframe(3), "amount") == 3
    assert _unit_name(getattr(timeframe.canonicalize_timeframe(3), "unit")).lower().endswith("day")
    assert getattr(timeframe.canonicalize_timeframe("15 minute"), "amount") == 15
    assert _unit_name(getattr(timeframe.canonicalize_timeframe("1 h"), "unit")).lower().endswith("hour")
    assert _unit_name(getattr(timeframe.canonicalize_timeframe("2 mo"), "unit")).lower().endswith("month")
    assert _unit_name(getattr(timeframe.canonicalize_timeframe("surprise"), "unit")).lower().endswith("day")


def test_canonicalize_timeframe_object_with_amount_unit_and_none_unit() -> None:
    unit_cls = timeframe._resolve_timeframe_unit_cls()
    hour = getattr(unit_cls, "Hour")

    from_attrs = timeframe.canonicalize_timeframe(SimpleNamespace(amount="4", unit=hour))
    defaulted = timeframe.TimeFrame(5, None)

    assert getattr(from_attrs, "amount") == 4
    assert _unit_name(getattr(from_attrs, "unit")).lower().endswith("hour")
    assert getattr(defaulted, "amount") == 5
    assert _unit_name(getattr(defaulted, "unit")).lower().endswith("day")


def test_timeframe_fallback_unit_and_base_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class IncompleteUnit:
        Day = "Day"

    class FallbackUnit:
        Minute = "Minute"
        Hour = "Hour"
        Day = "Day"
        Week = "Week"
        Month = "Month"

    monkeypatch.setattr(timeframe, "TimeFrameUnit", IncompleteUnit)
    monkeypatch.setattr(
        "ai_trading.alpaca_api.TimeFrameUnit",
        FallbackUnit,
        raising=False,
    )

    assert timeframe._resolve_timeframe_unit_cls() is FallbackUnit

    original_init = timeframe._BaseTimeFrame.__init__

    def _raise_init(self: object, *_args: object, **_kwargs: object) -> None:
        raise ValueError("bad base")

    monkeypatch.setattr(timeframe._BaseTimeFrame, "__init__", _raise_init)
    try:
        fallback = timeframe.TimeFrame("bad", "Minute")  # type: ignore[arg-type]
    finally:
        monkeypatch.setattr(timeframe._BaseTimeFrame, "__init__", original_init)

    assert getattr(fallback, "amount", None) == 1
    assert getattr(fallback, "unit", None) == "Minute"


def test_timeframe_resolve_unit_uses_local_fallback_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteUnit:
        Day = "Day"

    monkeypatch.setattr(timeframe, "TimeFrameUnit", IncompleteUnit)

    import ai_trading.alpaca_api as alpaca_api

    monkeypatch.delattr(alpaca_api, "TimeFrameUnit", raising=False)
    resolved = timeframe._resolve_timeframe_unit_cls()

    assert resolved.Minute == "Minute"
    assert resolved.Month == "Month"
