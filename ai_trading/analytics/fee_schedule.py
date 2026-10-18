"""Declared fee-schedule provenance for new fill records.

The Alpaca paper activity schema does not provide broker-confirmed per-fill
total fees (research reset evidence gap 3). This module lets the owner
*declare* a fee schedule with explicit provenance so new fill records can carry
a computed fee that is clearly labeled ``fee_source="declared_schedule"``.

Rules enforced here:

* Disabled by default (``AI_TRADING_FEE_SCHEDULE_SOURCE=missing``). When
  disabled, :func:`declared_fee_fields` returns an empty mapping and fill
  records are unchanged.
* Every rate must be a number copied from the declared source; placeholders
  (``"<...>"``) are rejected so an unfilled example can never be applied.
* A declared fee is never broker evidence. Consumers must use
  :func:`fee_provenance_class` (or the constants below) and keep
  ``declared_schedule`` separate from ``broker_payload``/``broker_activity``.
* Nothing here rewrites or backfills historical records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger

logger = get_logger(__name__)

FEE_SCHEDULE_SOURCE_ENV = "AI_TRADING_FEE_SCHEDULE_SOURCE"
FEE_SCHEDULE_PATH_ENV = "AI_TRADING_FEE_SCHEDULE_PATH"
FEE_SCHEDULE_SOURCE_MISSING = "missing"
FEE_SCHEDULE_SOURCE_DECLARED = "declared"
FEE_SCHEDULE_SOURCES = frozenset({FEE_SCHEDULE_SOURCE_MISSING, FEE_SCHEDULE_SOURCE_DECLARED})

FEE_SOURCE_DECLARED_SCHEDULE = "declared_schedule"
FEE_BASIS_DECLARED_SCHEDULE = "declared_schedule_per_fill"
FEE_EVIDENCE_CLASS_DECLARED = "declared_schedule_not_broker_confirmed_per_fill"
BROKER_FEE_SOURCES = frozenset({"broker_payload", "broker_activity"})
BROKER_FEE_BASIS = "per_fill_total"

# Provenance classes returned by fee_provenance_class().
FEE_PROVENANCE_BROKER = "broker_verified"
FEE_PROVENANCE_DECLARED = "declared_schedule"
FEE_PROVENANCE_ESTIMATE = "configured_estimate"
FEE_PROVENANCE_SIMULATED = "simulated"
FEE_PROVENANCE_MISSING = "missing"
FEE_PROVENANCE_UNVERIFIED = "unverified_other"

SCHEMA_VERSION = 1
_SIDES = frozenset({"buy", "sell"})
_CATEGORIES = frozenset({"commission", "regulatory"})
_BASES = frozenset({"per_share", "per_notional", "per_fill"})
_ROUNDING_MODES = {"none": None, "ceil": ROUND_CEILING, "half_up": ROUND_HALF_UP}
_ROUNDING_SCOPES = frozenset({"per_fill", "per_component"})
# Side rules documented on Alpaca's Regulatory Fees page: TAF applies to sells
# only, CAT to buys and sells. SEC Section 31 applies to sells.
_REQUIRED_SIDES = {
    "finra_taf": frozenset({"sell"}),
    "sec_section_31": frozenset({"sell"}),
    "cat": frozenset({"buy", "sell"}),
}
_REQUIRED_PROVENANCE = ("source_url", "retrieved_at", "declared_by", "declared_at")
_MARKET_TZ = ZoneInfo("America/New_York")


class FeeScheduleError(ValueError):
    """Raised when a declared fee schedule is missing, invalid or unfilled."""


@dataclass(frozen=True)
class FeeComponent:
    name: str
    category: str
    basis: str
    rate: Decimal
    sides: frozenset[str]
    min_usd_per_fill: Decimal | None = None
    max_usd_per_fill: Decimal | None = None


@dataclass(frozen=True)
class FeeSchedule:
    schedule_id: str
    currency: str
    effective_from: date
    effective_to: date | None
    rounding_mode: str
    rounding_increment: Decimal | None
    rounding_scope: str
    venues: Mapping[str, tuple[FeeComponent, ...]]
    provenance: Mapping[str, str]
    sha256: str


@dataclass(frozen=True)
class DeclaredFee:
    fee_amount: float
    fee_bps: float
    components: Mapping[str, float]


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("<")


def _require_text(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = payload.get(key)
    if _is_placeholder(value):
        raise FeeScheduleError(f"{where}.{key} is an unfilled placeholder")
    if not isinstance(value, str) or not value.strip():
        raise FeeScheduleError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _decimal(value: Any, where: str, *, allow_none: bool = False) -> Decimal | None:
    if value is None and allow_none:
        return None
    if _is_placeholder(value):
        raise FeeScheduleError(f"{where} is an unfilled placeholder")
    if isinstance(value, bool) or value is None:
        raise FeeScheduleError(f"{where} must be a number")
    try:
        parsed = Decimal(str(value).strip()) if isinstance(value, (int, float, str)) else None
    except InvalidOperation as exc:
        raise FeeScheduleError(f"{where} must be a number") from exc
    if parsed is None or not parsed.is_finite() or parsed < 0:
        raise FeeScheduleError(f"{where} must be a finite non-negative number")
    return parsed


def _required_decimal(value: Any, where: str) -> Decimal:
    parsed = _decimal(value, where)
    if parsed is None:
        raise FeeScheduleError(f"{where} must be a number")
    return parsed


def _timestamp(value: Any, where: str) -> datetime:
    text = _require_text({"v": value}, "v", where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FeeScheduleError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise FeeScheduleError(f"{where} must include a timezone offset")
    return parsed


def _date(value: Any, where: str) -> date:
    text = _require_text({"v": value}, "v", where)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise FeeScheduleError(f"{where} must be an ISO-8601 date") from exc


def _component(raw: Any, where: str) -> FeeComponent:
    if not isinstance(raw, Mapping):
        raise FeeScheduleError(f"{where} must be an object")
    name = _require_text(raw, "name", where).lower()
    category = _require_text(raw, "category", where).lower()
    basis = _require_text(raw, "basis", where).lower()
    if category not in _CATEGORIES:
        raise FeeScheduleError(f"{where}.category must be one of {sorted(_CATEGORIES)}")
    if basis not in _BASES:
        raise FeeScheduleError(f"{where}.basis must be one of {sorted(_BASES)}")
    if category == "regulatory" and basis == "per_fill":
        raise FeeScheduleError(f"{where}: regulatory components must be per_share or per_notional")
    sides_raw = raw.get("sides")
    if not isinstance(sides_raw, Sequence) or isinstance(sides_raw, str) or not sides_raw:
        raise FeeScheduleError(f"{where}.sides must be a non-empty list")
    sides = frozenset(str(side).strip().lower() for side in sides_raw)
    if not sides <= _SIDES:
        raise FeeScheduleError(f"{where}.sides must only contain buy/sell")
    required = _REQUIRED_SIDES.get(name)
    if required is not None and sides != required:
        raise FeeScheduleError(f"{where}: {name} must apply to sides {sorted(required)}")
    rate = _required_decimal(raw.get("rate"), f"{where}.rate")
    minimum = _decimal(raw.get("min_usd_per_fill"), f"{where}.min_usd_per_fill", allow_none=True)
    maximum = _decimal(raw.get("max_usd_per_fill"), f"{where}.max_usd_per_fill", allow_none=True)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise FeeScheduleError(f"{where}: min_usd_per_fill exceeds max_usd_per_fill")
    return FeeComponent(name, category, basis, rate, sides, minimum, maximum)


def parse_fee_schedule(payload: Any, *, sha256: str) -> FeeSchedule:
    """Validate a declared fee schedule payload and return a :class:`FeeSchedule`."""

    if not isinstance(payload, Mapping):
        raise FeeScheduleError("schedule must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FeeScheduleError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("evidence_class") != FEE_EVIDENCE_CLASS_DECLARED:
        raise FeeScheduleError(f"evidence_class must be {FEE_EVIDENCE_CLASS_DECLARED!r}")
    if payload.get("promotion_authority") is not False:
        raise FeeScheduleError("promotion_authority must be false")
    schedule_id = _require_text(payload, "schedule_id", "schedule")
    currency = _require_text(payload, "currency", "schedule").upper()
    if currency != "USD":
        raise FeeScheduleError("currency must be USD")

    provenance_raw = payload.get("provenance")
    if not isinstance(provenance_raw, Mapping):
        raise FeeScheduleError("provenance block is mandatory")
    provenance = {key: _require_text(provenance_raw, key, "provenance") for key in _REQUIRED_PROVENANCE}
    if not provenance["source_url"].lower().startswith(("https://", "http://")):
        raise FeeScheduleError("provenance.source_url must be an http(s) URL")
    _timestamp(provenance["retrieved_at"], "provenance.retrieved_at")
    _timestamp(provenance["declared_at"], "provenance.declared_at")
    for key, value in provenance_raw.items():
        if key in provenance or value in (None, ""):
            continue
        if _is_placeholder(value):
            raise FeeScheduleError(f"provenance.{key} is an unfilled placeholder")
        provenance[str(key)] = str(value)

    effective_from = _date(payload.get("effective_from"), "effective_from")
    effective_to_raw = payload.get("effective_to")
    effective_to = None if effective_to_raw is None else _date(effective_to_raw, "effective_to")
    if effective_to is not None and effective_to < effective_from:
        raise FeeScheduleError("effective_to precedes effective_from")

    rounding = payload.get("rounding")
    if not isinstance(rounding, Mapping):
        raise FeeScheduleError("rounding block is mandatory")
    mode = _require_text(rounding, "mode", "rounding").lower()
    scope = _require_text(rounding, "scope", "rounding").lower()
    if mode not in _ROUNDING_MODES:
        raise FeeScheduleError(f"rounding.mode must be one of {sorted(_ROUNDING_MODES)}")
    if scope not in _ROUNDING_SCOPES:
        raise FeeScheduleError(f"rounding.scope must be one of {sorted(_ROUNDING_SCOPES)}")
    increment = None
    if mode != "none":
        increment = _required_decimal(rounding.get("increment_usd"), "rounding.increment_usd")
        if increment <= 0:
            raise FeeScheduleError("rounding.increment_usd must be positive")

    venues_raw = payload.get("venues")
    if not isinstance(venues_raw, Mapping) or not venues_raw:
        raise FeeScheduleError("venues must be a non-empty object")
    venues: dict[str, tuple[FeeComponent, ...]] = {}
    for venue_key, venue_raw in venues_raw.items():
        venue = str(venue_key).strip().upper()
        where = f"venues.{venue_key}"
        if not venue or not isinstance(venue_raw, Mapping):
            raise FeeScheduleError(f"{where} must be an object")
        components_raw = venue_raw.get("components")
        if not isinstance(components_raw, Sequence) or isinstance(components_raw, str) or not components_raw:
            raise FeeScheduleError(f"{where}.components must be a non-empty list")
        components = tuple(
            _component(raw, f"{where}.components[{index}]") for index, raw in enumerate(components_raw)
        )
        names = [component.name for component in components]
        if len(set(names)) != len(names):
            raise FeeScheduleError(f"{where}: component names must be unique")
        if not any(component.category == "commission" for component in components):
            raise FeeScheduleError(f"{where}: a commission component is mandatory (declare rate 0 explicitly)")
        venues[venue] = components
    return FeeSchedule(
        schedule_id=schedule_id,
        currency=currency,
        effective_from=effective_from,
        effective_to=effective_to,
        rounding_mode=mode,
        rounding_increment=increment,
        rounding_scope=scope,
        venues=venues,
        provenance=provenance,
        sha256=sha256,
    )


def load_fee_schedule(path: str | Path) -> FeeSchedule:
    """Read, hash and validate a declared schedule file."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise FeeScheduleError(f"schedule file unreadable: {target}") from exc
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise FeeScheduleError(f"schedule file is not valid JSON: {target}") from exc
    return parse_fee_schedule(payload, sha256=hashlib.sha256(raw).hexdigest())


def _round(amount: Decimal, schedule: FeeSchedule) -> Decimal:
    rounding = _ROUNDING_MODES[schedule.rounding_mode]
    if rounding is None or schedule.rounding_increment is None:
        return amount
    increment = schedule.rounding_increment
    return (amount / increment).to_integral_value(rounding=rounding) * increment


def _side_token(side: Any) -> str | None:
    token = str(side or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token == "buy":
        return "buy"
    if token in {"sell", "sell_short", "short", "sellshort"}:
        return "sell"
    return None


def compute_declared_fee(
    schedule: FeeSchedule,
    *,
    side: Any,
    qty: float,
    price: float,
    fill_ts: datetime,
    venue: str | None = None,
) -> tuple[DeclaredFee | None, str]:
    """Compute a declared fee for one fill.

    Returns ``(fee, "applied")`` or ``(None, reason)``; never imputes zero.
    """

    side_token = _side_token(side)
    if side_token is None:
        return None, "side_unrecognized"
    if not (math.isfinite(qty) and math.isfinite(price)) or qty <= 0 or price <= 0:
        return None, "invalid_fill_values"
    if fill_ts.tzinfo is None:
        return None, "fill_timestamp_naive"
    trade_date = fill_ts.astimezone(_MARKET_TZ).date()
    if trade_date < schedule.effective_from or (
        schedule.effective_to is not None and trade_date > schedule.effective_to
    ):
        return None, "outside_effective_window"
    venue_key = str(venue or "").strip().upper()
    components = schedule.venues.get(venue_key) if venue_key else None
    if components is None:
        components = schedule.venues.get("DEFAULT")
    if components is None:
        return None, "venue_not_declared"
    quantity = Decimal(str(qty))
    notional = quantity * Decimal(str(price))
    breakdown: dict[str, Decimal] = {}
    for component in components:
        if side_token not in component.sides:
            continue
        if component.basis == "per_share":
            amount = component.rate * quantity
        elif component.basis == "per_notional":
            amount = component.rate * notional
        else:
            amount = component.rate
        if component.min_usd_per_fill is not None:
            amount = max(amount, component.min_usd_per_fill)
        if component.max_usd_per_fill is not None:
            amount = min(amount, component.max_usd_per_fill)
        if schedule.rounding_scope == "per_component":
            amount = _round(amount, schedule)
        breakdown[component.name] = amount
    total = sum(breakdown.values(), Decimal(0))
    if schedule.rounding_scope == "per_fill":
        total = _round(total, schedule)
    fee_amount = float(total)
    return (
        DeclaredFee(
            fee_amount=fee_amount,
            fee_bps=float(total / notional * Decimal(10000)),
            components={name: float(value) for name, value in breakdown.items()},
        ),
        "applied",
    )


_SCHEDULE_CACHE: dict[tuple[str, int, int], FeeSchedule | str] = {}
_LOGGED_FAILURES: set[tuple[str, str]] = set()


def _resolve_config() -> tuple[str, str | None]:
    from ai_trading.config.management import get_env

    source = str(get_env(FEE_SCHEDULE_SOURCE_ENV, FEE_SCHEDULE_SOURCE_MISSING) or FEE_SCHEDULE_SOURCE_MISSING)
    path = get_env(FEE_SCHEDULE_PATH_ENV, None)
    return source.strip().lower(), (str(path).strip() if path not in (None, "") else None)


def _cached_schedule(path: str) -> FeeSchedule | str:
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return "schedule_file_missing"
    key = (str(target.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _SCHEDULE_CACHE.get(key)
    if cached is None:
        try:
            cached = load_fee_schedule(target)
        except FeeScheduleError as exc:
            cached = f"schedule_invalid:{exc}"
        _SCHEDULE_CACHE.clear()
        _SCHEDULE_CACHE[key] = cached
    return cached


def _unavailable(reason: str, *, path: str | None, schedule_hash: str | None = None) -> dict[str, Any]:
    failure_key = (str(path), reason)
    if failure_key not in _LOGGED_FAILURES:
        _LOGGED_FAILURES.add(failure_key)
        logger.error(
            "FEE_SCHEDULE_DECLARED_UNAVAILABLE",
            extra={"reason": reason, "path": path, "fee_schedule_hash": schedule_hash},
        )
    fields: dict[str, Any] = {"fee_schedule_status": reason}
    if schedule_hash:
        fields["fee_schedule_hash"] = schedule_hash
    return fields


def declared_fee_fields(
    *,
    side: Any,
    qty: float,
    price: float,
    fill_ts: datetime,
    venue: str | None = None,
) -> dict[str, Any]:
    """Return declared-schedule fee fields to merge into a new fill record.

    Returns ``{}`` when the declared source is disabled (the default). When
    enabled but not applicable, returns only ``fee_schedule_status`` so the
    fee stays missing rather than being imputed.
    """

    source, path = _resolve_config()
    if source == FEE_SCHEDULE_SOURCE_MISSING:
        return {}
    if source not in FEE_SCHEDULE_SOURCES:
        return _unavailable(f"invalid_source:{source}", path=path)
    if not path:
        return _unavailable("schedule_path_unset", path=None)
    schedule = _cached_schedule(path)
    if isinstance(schedule, str):
        return _unavailable(schedule, path=path)
    fee, reason = compute_declared_fee(
        schedule, side=side, qty=float(qty), price=float(price), fill_ts=fill_ts, venue=venue
    )
    if fee is None:
        return _unavailable(reason, path=path, schedule_hash=schedule.sha256)
    return {
        "fee_amount": fee.fee_amount,
        "fee_bps": fee.fee_bps,
        "fee_currency": schedule.currency,
        "fee_basis": FEE_BASIS_DECLARED_SCHEDULE,
        "fee_source": FEE_SOURCE_DECLARED_SCHEDULE,
        "fee_evidence_class": FEE_EVIDENCE_CLASS_DECLARED,
        "fee_schedule_id": schedule.schedule_id,
        "fee_schedule_hash": schedule.sha256,
        "fee_schedule_status": "applied",
        "fee_components": dict(fee.components),
    }


def declared_fee_provenance(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the provenance fields to carry downstream for a declared fee row."""

    if str(row.get("fee_source") or "").strip().lower() != FEE_SOURCE_DECLARED_SCHEDULE:
        return None
    keys = ("fee_source", "fee_basis", "fee_currency", "fee_evidence_class", "fee_schedule_id", "fee_schedule_hash")
    return {key: row.get(key) for key in keys if row.get(key) not in (None, "")}


def _fee_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("fee_amount")
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def is_broker_verified_fee(row: Mapping[str, Any]) -> bool:
    """True only for broker-sourced USD per-fill total fees."""

    return (
        str(row.get("fee_source") or "") in BROKER_FEE_SOURCES
        and row.get("fee_basis") == BROKER_FEE_BASIS
        and row.get("fee_currency") == "USD"
        and _fee_value(row) is not None
    )


def is_declared_schedule_fee(row: Mapping[str, Any]) -> bool:
    """True for a complete declared-schedule fee (usable for net-cost reporting only)."""

    return (
        str(row.get("fee_source") or "") == FEE_SOURCE_DECLARED_SCHEDULE
        and row.get("fee_basis") == FEE_BASIS_DECLARED_SCHEDULE
        and row.get("fee_currency") == "USD"
        and bool(row.get("fee_schedule_hash"))
        and _fee_value(row) is not None
    )


def fee_provenance_class(row: Mapping[str, Any]) -> str:
    """Classify a fill/TCA row's fee provenance into a distinct, labeled class."""

    if is_broker_verified_fee(row):
        return FEE_PROVENANCE_BROKER
    if is_declared_schedule_fee(row):
        return FEE_PROVENANCE_DECLARED
    source = str(row.get("fee_source") or "").strip().lower()
    if source == "configured_estimate":
        return FEE_PROVENANCE_ESTIMATE
    if source.startswith("simulated"):
        return FEE_PROVENANCE_SIMULATED
    if source in {"", "missing"}:
        return FEE_PROVENANCE_MISSING
    return FEE_PROVENANCE_UNVERIFIED


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a schedule file without enabling it (operator pre-flight)."""

    parser = argparse.ArgumentParser(description="Validate a declared fee schedule file.")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    try:
        schedule = load_fee_schedule(args.path)
    except FeeScheduleError as exc:
        logger.error("FEE_SCHEDULE_VALIDATION_FAILED", extra={"path": args.path, "error": str(exc)})
        return 1
    logger.info(
        "FEE_SCHEDULE_VALIDATION_OK",
        extra={
            "path": args.path,
            "schedule_id": schedule.schedule_id,
            "fee_schedule_hash": schedule.sha256,
            "venues": sorted(schedule.venues),
            "effective_from": schedule.effective_from.isoformat(),
            "effective_to": schedule.effective_to.isoformat() if schedule.effective_to else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
