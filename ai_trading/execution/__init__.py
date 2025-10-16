"""Execution Module - Institutional Grade Order Management with Enhanced Debugging."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Tuple

from ai_trading.logging import get_logger
from ai_trading.utils.env import (
    alpaca_credential_status,
    get_alpaca_base_url,
)

_logger = get_logger(__name__)


# Core exports that should always be available
from .classes import ExecutionResult, OrderRequest
from .engine import ExecutionAlgorithm, Order
from .engine import ExecutionEngine as _SimExecutionEngine

_DEFAULT_PRICE_PROVIDER_ORDER: tuple[str, ...] = (
    "alpaca_quote",
    "alpaca_trade",
    "alpaca_minute_close",
    "yahoo",
    "bars",
)

_DEFAULT_EXECUTION_SETTINGS = SimpleNamespace(
    mode="sim",
    shadow_mode=False,
    order_timeout_seconds=300,
    slippage_limit_bps=75,
    price_provider_order=_DEFAULT_PRICE_PROVIDER_ORDER,
    data_feed_intraday="iex",
)


@dataclass(frozen=True)
class ExecutionEngineStatus:
    """Snapshot of the last execution engine selection outcome."""

    mode: str
    engine_class: str | None
    shadow_mode: bool
    missing_credentials: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    reason: str | None
    settings_fallback: str | None


_RUNTIME_STATUS = ExecutionEngineStatus(
    mode="sim",
    engine_class=f"{_SimExecutionEngine.__module__}.{_SimExecutionEngine.__qualname__}",
    shadow_mode=False,
    missing_credentials=(),
    missing_dependencies=(),
    reason=None,
    settings_fallback=None,
)


def _collect_dependency_gaps(mode: str) -> list[str]:
    """Return missing optional runtime dependencies for the requested *mode*."""

    required_modules: tuple[str, ...] = ()
    if mode in {"paper", "live"}:
        required_modules = (
            "alpaca",
            "alpaca.common",
            "alpaca.trading",
            "alpaca.trading.client",
            "alpaca.trading.requests",
            "alpaca.trading.enums",
            "alpaca.data",
        )
    missing: list[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


def _load_execution_settings() -> Tuple[Any, str | None]:
    """Return execution settings and a fallback reason when defaults are used."""

    fallback = SimpleNamespace(**vars(_DEFAULT_EXECUTION_SETTINGS))

    try:
        from ai_trading.config import (  # type: ignore[import-not-found]
            DATA_FEED_INTRADAY as _DATA_FEED_INTRADAY,
            PRICE_PROVIDER_ORDER as _PRICE_PROVIDER_ORDER,
            get_execution_settings as _get_execution_settings,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        missing = getattr(exc, "name", "")
        if missing in {"pydantic", "pydantic_settings"}:
            _logger.warning(
                "EXECUTION_SETTINGS_DEPENDENCY_MISSING",
                extra={"dependency": missing},
            )
            return fallback, "config_dependency_missing"
        raise
    except ImportError as exc:  # pragma: no cover - unexpected import failure
        _logger.error(
            "EXECUTION_SETTINGS_IMPORT_FAILED",
            extra={"error": str(exc)},
        )
        return fallback, "config_import_failed"

    # Update fallback defaults with values exported from the config module when available.
    fallback = SimpleNamespace(
        mode="sim",
        shadow_mode=False,
        order_timeout_seconds=300,
        slippage_limit_bps=75,
        price_provider_order=tuple(_PRICE_PROVIDER_ORDER),
        data_feed_intraday=str(_DATA_FEED_INTRADAY or "iex"),
    )

    try:
        settings = _get_execution_settings()
    except Exception as exc:  # pragma: no cover - configuration load failure
        _logger.error(
            "EXECUTION_SETTINGS_LOAD_FAILED",
            extra={"error": str(exc)},
        )
        return fallback, "config_load_failed"
    return settings, None


def _missing_creds() -> list[str]:
    has_key, has_secret = alpaca_credential_status()
    missing: list[str] = []
    if not has_key:
        missing.append("ALPACA_API_KEY")
    if not has_secret:
        missing.append("ALPACA_SECRET_KEY")
    return missing


def _creds_ok() -> bool:
    has_key, has_secret = alpaca_credential_status()
    return has_key and has_secret


def _select_execution_engine() -> type[_SimExecutionEngine]:
    """Return the execution engine class based on runtime configuration."""

    settings, settings_reason = _load_execution_settings()

    mode = "sim"
    shadow = False
    if settings is not None:
        mode = str(settings.mode or "sim").lower()
        shadow = bool(settings.shadow_mode)

    normalized_mode = {
        "broker": "paper",
        "alpaca": "paper",
    }.get(mode, mode)

    engine_cls: type[_SimExecutionEngine] = _SimExecutionEngine
    engine_class_path = f"{engine_cls.__module__}.{engine_cls.__qualname__}"
    missing_creds: list[str] | None = None
    reason: str | None = settings_reason
    missing_dependencies = _collect_dependency_gaps(normalized_mode)

    has_key, has_secret = alpaca_credential_status()
    base_url = get_alpaca_base_url()

    if normalized_mode in {"paper", "live"}:
        missing_creds = _missing_creds() or None
        try:
            from .live_trading import ExecutionEngine as _LiveExecutionEngine
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            reason = reason or "import_failed"
            missing_mod = getattr(exc, "name", None)
            if missing_mod and missing_mod not in missing_dependencies:
                missing_dependencies.append(missing_mod)
            _logger.error(
                "EXECUTION_ENGINE_IMPORT_FAILED",
                extra={
                    "mode": normalized_mode,
                    "error": str(exc),
                    "missing_module": missing_mod,
                },
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            reason = reason or "import_failed"
            _logger.error(
                "EXECUTION_ENGINE_IMPORT_FAILED",
                extra={"mode": normalized_mode, "error": str(exc)},
            )
        else:
            engine_cls = _LiveExecutionEngine  # type: ignore[assignment]
            engine_class_path = f"{engine_cls.__module__}.{engine_cls.__qualname__}"
            if missing_creds:
                _logger.error(
                    "EXECUTION_CREDS_MISSING",
                    extra={
                        "has_key": has_key,
                        "has_secret": has_secret,
                        "base_url": base_url,
                    },
                )
    elif normalized_mode not in {"sim"}:
        reason = reason or "mode_unknown"
        _logger.warning(
            "EXECUTION_MODE_UNKNOWN",
            extra={"requested_mode": mode},
        )

    global _RUNTIME_STATUS
    _RUNTIME_STATUS = ExecutionEngineStatus(
        mode=normalized_mode,
        engine_class=engine_class_path,
        shadow_mode=shadow,
        missing_credentials=tuple(missing_creds or ()),
        missing_dependencies=tuple(missing_dependencies),
        reason=reason,
        settings_fallback=settings_reason,
    )

    _logger.info(
        "EXECUTION_ENGINE_SELECTED",
        extra={
            "execution_mode": normalized_mode,
            "engine_class": engine_class_path,
            "shadow_mode": shadow,
            "creds_missing": tuple(missing_creds) if missing_creds else None,
            "reason": reason,
            "settings_fallback": settings_reason,
            "has_key": has_key,
            "has_secret": has_secret,
            "base_url": base_url,
            "missing_dependencies": tuple(missing_dependencies) or None,
        },
    )
    return engine_cls


ExecutionEngine = _select_execution_engine()


def get_execution_runtime_status() -> ExecutionEngineStatus:
    """Return the most recent execution engine selection status."""

    return _RUNTIME_STATUS

# Optional submodule: algorithms
try:  # pragma: no cover - optional dependency
    from . import algorithms
except Exception:  # noqa: BLE001 - broad to guard optional deps
    algorithms = None

# Optional utilities guarded against missing dependencies
try:  # pragma: no cover - optional dependency
    from .debug_tracker import (
        ExecutionPhase,
        OrderStatus,
        enable_debug_mode,
        get_debug_tracker,
        get_execution_statistics,
        log_execution_phase,
        log_order_outcome,
        log_position_change,
        log_signal_to_execution,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    ExecutionPhase = OrderStatus = None
    enable_debug_mode = get_debug_tracker = None
    get_execution_statistics = None
    log_execution_phase = log_order_outcome = None
    log_position_change = log_signal_to_execution = None

try:  # pragma: no cover - optional dependency
    from .liquidity import (
        LiquidityAnalyzer,
        LiquidityLevel,
        LiquidityManager,
        MarketHours,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    LiquidityAnalyzer = LiquidityLevel = None
    LiquidityManager = MarketHours = None

try:  # pragma: no cover - optional dependency
    from .pnl_attributor import (
        PnLEvent,
        PnLSource,
        explain_recent_pnl_changes,
        get_pnl_attribution_stats,
        get_pnl_attributor,
        get_portfolio_pnl_summary,
        get_symbol_pnl_breakdown,
        record_dividend_income,
        record_trade_pnl,
        update_position_for_pnl,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    PnLEvent = PnLSource = None
    explain_recent_pnl_changes = get_pnl_attribution_stats = None
    get_pnl_attributor = get_portfolio_pnl_summary = None
    get_symbol_pnl_breakdown = record_dividend_income = None
    record_trade_pnl = update_position_for_pnl = None

try:  # pragma: no cover - optional dependency
    from .position_reconciler import (
        PositionDiscrepancy,
        adjust_bot_position,
        force_position_reconciliation,
        get_position_discrepancies,
        get_position_reconciler,
        get_reconciliation_statistics,
        start_position_monitoring,
        stop_position_monitoring,
        update_bot_position,
    )
except Exception:  # noqa: BLE001 - broad to guard optional deps
    PositionDiscrepancy = None
    adjust_bot_position = force_position_reconciliation = None
    get_position_discrepancies = get_position_reconciler = None
    get_reconciliation_statistics = start_position_monitoring = None
    stop_position_monitoring = update_bot_position = None

try:  # pragma: no cover - optional dependency
    from .transaction_costs import estimate_cost
except Exception:  # noqa: BLE001 - broad to guard optional deps
    estimate_cost = None

try:  # pragma: no cover - optional dependency
    from .production_engine import ProductionExecutionCoordinator
except Exception:  # noqa: BLE001 - broad to guard optional deps
    ProductionExecutionCoordinator = None

try:  # pragma: no cover - optional dependency
    from .pdt_manager import PDTManager
    from .swing_mode import SwingTradingMode, get_swing_mode, enable_swing_mode, disable_swing_mode
except Exception:  # noqa: BLE001 - broad to guard optional deps
    PDTManager = SwingTradingMode = None
    get_swing_mode = enable_swing_mode = disable_swing_mode = None

__all__ = [
    "Order",
    "ExecutionAlgorithm",
    "ExecutionEngine",
    "ProductionExecutionCoordinator",
    "ExecutionResult",
    "OrderRequest",
    "algorithms",
    "LiquidityAnalyzer",
    "LiquidityManager",
    "LiquidityLevel",
    "MarketHours",
    "get_debug_tracker",
    "log_signal_to_execution",
    "log_execution_phase",
    "log_order_outcome",
    "log_position_change",
    "enable_debug_mode",
    "get_execution_statistics",
    "ExecutionPhase",
    "OrderStatus",
    "get_position_reconciler",
    "update_bot_position",
    "adjust_bot_position",
    "force_position_reconciliation",
    "start_position_monitoring",
    "stop_position_monitoring",
    "get_position_discrepancies",
    "get_reconciliation_statistics",
    "PositionDiscrepancy",
    "get_pnl_attributor",
    "update_position_for_pnl",
    "record_trade_pnl",
    "record_dividend_income",
    "get_symbol_pnl_breakdown",
    "get_portfolio_pnl_summary",
    "explain_recent_pnl_changes",
    "get_pnl_attribution_stats",
    "PnLSource",
    "PnLEvent",
    "estimate_cost",
    "PDTManager",
    "SwingTradingMode",
    "get_swing_mode",
    "enable_swing_mode",
    "disable_swing_mode",
]

