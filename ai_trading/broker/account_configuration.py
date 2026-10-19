"""Read-only verification of Alpaca account configurations.

The broker-side account configuration is a backstop that holds even when a bug
gets past the bot's own risk checks.  This module only *reads*
``TradingClient.get_account_configurations()`` and compares it with what the
active launch profile implies.  It never writes configurations; the owner sets
them by hand (Alpaca dashboard, or the SDK call below) after reviewing drift.

Expected view (derived from the launch profile already resolved in-process):

* ``no_shorting`` must be ``True`` when the profile does not allow shorts.
* ``max_margin_multiplier`` must be ``"1"`` when ``max_gross_exposure <= 1``
  (``"2"`` up to 2x gross, otherwise ``"4"``).
* ``dtbp_check`` and ``pdt_check`` must run on entry (``"entry"`` or ``"both"``)
  when the API reports them; when it omits them they are listed under
  ``unreported_fields`` instead of counting as drift.
* ``fractional_trading`` is reported but not enforced: the runtime sizes whole
  shares and has no fractional-sizing setting to derive an expectation from.
* ``trade_confirm_email`` is irrelevant and ignored.

Setting the account by hand (paper and live accounts are configured
separately; run this with the credentials of the account you mean to change).
The typed SDK call sends a full ``AccountConfiguration``, so every field must
be supplied; copy the non-backstop values from the ``observed`` section of
``python -m ai_trading.tools.broker_account_configuration --check``::

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import DTBPCheck, PDTCheck, TradeConfirmationEmail
    from alpaca.trading.models import AccountConfiguration

    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    client.set_account_configurations(
        AccountConfiguration(
            no_shorting=True,
            max_margin_multiplier="1",
            dtbp_check=DTBPCheck.ENTRY,
            pdt_check=PDTCheck.ENTRY,
            fractional_trading=True,
            suspend_trade=False,
            trade_confirm_email=TradeConfirmationEmail.ALL,
            ptp_no_exception_entry=False,
        )
    )

When the API no longer reports ``dtbp_check``/``pdt_check`` (observed on the
paper account on 2026-10-19), the response cannot be parsed into that model;
the partial PATCH through the same client changes only the named fields::

    client.patch(
        "/account/configurations",
        {"no_shorting": True, "max_margin_multiplier": "1"},
    )

See ``docs/BROKER_ACCOUNT_CONFIGURATION.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from threading import Lock
from time import monotonic
from typing import Any

from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger

logger = get_logger(__name__)

OBSERVED_FIELDS: tuple[str, ...] = (
    "no_shorting",
    "max_margin_multiplier",
    "dtbp_check",
    "pdt_check",
    "fractional_trading",
    "suspend_trade",
    "closing_transactions_only",
    "disable_overnight_trading",
    "max_options_trading_level",
)
# Alpaca may stop reporting these; when absent they are listed as unreported
# instead of being counted as drift (the owner cannot set an absent field).
OPTIONAL_FIELDS: frozenset[str] = frozenset({"dtbp_check", "pdt_check"})
_CONFIGURATIONS_PATH = "/account/configurations"
_ENTRY_CHECK_VALUES: tuple[str, ...] = ("entry", "both")
_UNAVAILABLE_RETRY_SECONDS = 300.0

_STATE_LOCK = Lock()
_LAST_RESULT: dict[str, Any] | None = None
_LAST_ATTEMPT_MONOTONIC: float | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _api_error_types() -> tuple[type[Exception], ...]:
    try:
        from alpaca.common.exceptions import APIError
    except ImportError:  # pragma: no cover - alpaca-py is a runtime dependency
        return ()
    return (APIError,)


def _scalar(value: Any) -> Any:
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return raw.strip().lower()
    return raw


def _observed_view(config: Any) -> dict[str, Any]:
    """Return reported fields only; absent fields are omitted, not ``None``."""

    observed: dict[str, Any] = {}
    for field in OBSERVED_FIELDS:
        if isinstance(config, Mapping):
            if field not in config:
                continue
            raw = config.get(field)
        else:
            if not hasattr(config, field):
                continue
            raw = getattr(config, field)
        observed[field] = _scalar(raw)
    return observed


def _fetch_configuration(client: Any) -> tuple[Any, str]:
    """Return ``(configuration, source)`` via the SDK, read-only.

    alpaca-py 0.42.1's ``AccountConfiguration`` model requires ``dtbp_check``
    and ``pdt_check``; when the API omits them the typed call fails validation,
    so the same client's raw REST GET of the same endpoint is used instead.
    """

    from pydantic import ValidationError

    try:
        return client.get_account_configurations(), "sdk_model"
    except ValidationError:
        raw_get = getattr(client, "get", None)
        if not callable(raw_get):
            raise
        return raw_get(_CONFIGURATIONS_PATH), "sdk_raw_get"


def _margin_multiplier_for(max_gross_exposure: float) -> str:
    if max_gross_exposure <= 1.0:
        return "1"
    if max_gross_exposure <= 2.0:
        return "2"
    return "4"


def build_expected_configuration(profile: Any | None = None) -> dict[str, Any]:
    """Return the expected broker configuration implied by ``profile``.

    Each value is either a scalar that must match exactly or a list of
    acceptable values.  Fields with no expectation are omitted.
    """

    if profile is None:
        from ai_trading.config.launch_profiles import resolve_launch_profile

        profile = resolve_launch_profile()
    expected: dict[str, Any] = {}
    if not bool(getattr(profile, "shorts_allowed", False)):
        expected["no_shorting"] = True
    try:
        gross = float(getattr(profile, "max_gross_exposure", 1.0))
    except (TypeError, ValueError):
        gross = 1.0
    expected["max_margin_multiplier"] = _margin_multiplier_for(gross)
    expected["dtbp_check"] = list(_ENTRY_CHECK_VALUES)
    expected["pdt_check"] = list(_ENTRY_CHECK_VALUES)
    return expected


def _matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, list):
        return observed in expected
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, str) and observed is not None:
        try:
            return float(str(observed)) == float(expected)
        except ValueError:
            return str(observed) == expected
    return bool(observed == expected)


def compare_configuration(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return ``[{field, expected, observed}]`` for every unmet expectation."""

    mismatches: list[dict[str, Any]] = []
    for field, want in expected.items():
        if field in OPTIONAL_FIELDS and field not in observed:
            continue
        got = observed.get(field)
        if not _matches(want, got):
            mismatches.append({"field": field, "expected": want, "observed": got})
    return mismatches


def _unavailable(reason: str, *, expected: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "ok": False,
        "available": False,
        "reason": reason,
        "observed": {},
        "expected": dict(expected or {}),
        "mismatches": [],
        "unreported_fields": [],
        "source": None,
        "fetched_at": _now_iso(),
    }


def check_account_configuration(
    client: Any,
    *,
    profile: Any | None = None,
) -> dict[str, Any]:
    """Fetch, compare and summarize account configurations; never raises."""

    try:
        expected = build_expected_configuration(profile)
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return _unavailable(f"launch_profile_unavailable:{type(exc).__name__}", expected=None)
    if client is None:
        return _unavailable("trading_client_unavailable", expected=expected)
    if not callable(getattr(client, "get_account_configurations", None)):
        return _unavailable("get_account_configurations_unsupported", expected=expected)
    try:
        config, source = _fetch_configuration(client)
    except AI_TRADING_FALLBACK_EXCEPTIONS + _api_error_types() as exc:
        return _unavailable(f"fetch_failed:{type(exc).__name__}:{exc}"[:240], expected=expected)
    if config is None:
        return _unavailable("empty_response", expected=expected)
    observed = _observed_view(config)
    mismatches = compare_configuration(observed, expected)
    return {
        "ok": not mismatches,
        "available": True,
        "reason": "match" if not mismatches else "drift",
        "observed": observed,
        "expected": expected,
        "mismatches": mismatches,
        "unreported_fields": sorted(
            field for field in expected if field in OPTIONAL_FIELDS and field not in observed
        ),
        "source": source,
        "fetched_at": _now_iso(),
    }


def record_account_configuration_result(result: Mapping[str, Any]) -> None:
    global _LAST_RESULT
    with _STATE_LOCK:
        _LAST_RESULT = dict(result)


def observe_account_configuration() -> dict[str, Any] | None:
    """Return the cached startup check result (``None`` before the first run)."""

    with _STATE_LOCK:
        return None if _LAST_RESULT is None else dict(_LAST_RESULT)


def reset_account_configuration_state() -> None:
    global _LAST_RESULT, _LAST_ATTEMPT_MONOTONIC
    with _STATE_LOCK:
        _LAST_RESULT = None
        _LAST_ATTEMPT_MONOTONIC = None


def run_startup_account_configuration_check(client: Any) -> dict[str, Any] | None:
    """Run the read-only check once per process after broker sync.

    A successful fetch is never repeated; an unavailable result is retried at
    most every few minutes.  Drift is logged, not enforced.  Never raises.
    """

    global _LAST_ATTEMPT_MONOTONIC
    now = monotonic()
    with _STATE_LOCK:
        previous = _LAST_RESULT
        if previous is not None and previous.get("available"):
            return None
        if (
            _LAST_ATTEMPT_MONOTONIC is not None
            and now - _LAST_ATTEMPT_MONOTONIC < _UNAVAILABLE_RETRY_SECONDS
        ):
            return None
        _LAST_ATTEMPT_MONOTONIC = now
    try:
        result = check_account_configuration(client)
        record_account_configuration_result(result)
        if not result["available"]:
            logger.warning(
                "BROKER_ACCOUNT_CONFIGURATION_UNAVAILABLE",
                extra={"reason": result.get("reason")},
            )
        elif result["mismatches"]:
            logger.warning(
                "BROKER_ACCOUNT_CONFIGURATION_DRIFT",
                extra={
                    "mismatches": result["mismatches"],
                    "observed": result["observed"],
                },
            )
        else:
            logger.info(
                "BROKER_ACCOUNT_CONFIGURATION_OK",
                extra={"observed": result["observed"]},
            )
        return result
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.warning("BROKER_ACCOUNT_CONFIGURATION_CHECK_FAILED", exc_info=True)
        return None


def health_section(result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``broker_account_configuration`` health section."""

    if result is None:
        result = observe_account_configuration()
    if result is None:
        return {
            "ok": None,
            "available": False,
            "reason": "not_checked",
            "mismatches": [],
            "unreported_fields": [],
            "fetched_at": None,
        }
    mismatches = result.get("mismatches")
    return {
        "ok": result.get("ok") if result.get("available") else None,
        "available": bool(result.get("available")),
        "reason": result.get("reason"),
        "mismatches": list(mismatches) if isinstance(mismatches, list) else [],
        "unreported_fields": list(result.get("unreported_fields") or []),
        "fetched_at": result.get("fetched_at"),
    }


__all__ = [
    "OBSERVED_FIELDS",
    "OPTIONAL_FIELDS",
    "build_expected_configuration",
    "check_account_configuration",
    "compare_configuration",
    "health_section",
    "observe_account_configuration",
    "record_account_configuration_result",
    "reset_account_configuration_state",
    "run_startup_account_configuration_check",
]
