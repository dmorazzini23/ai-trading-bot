"""Execution Module - Institutional Grade Order Management with Enhanced Debugging."""

from __future__ import annotations

from ai_trading.logging import get_logger
from ai_trading.config import get_execution_settings
from ai_trading.utils.env import (
    alpaca_credential_status,
    get_alpaca_base_url,
)

_logger = get_logger(__name__)


# Core exports that should always be available
from .classes import ExecutionResult, OrderRequest
from .engine import ExecutionAlgorithm, Order
from .engine import ExecutionEngine as _SimExecutionEngine


def _missing_creds() -> list[str]:
    has_key, has_secret = alpaca_credential_status()
    missing: list[str] = []
    if not has_key:
        missing.append("ALPACA_API_KEY_ID")
    if not has_secret:
        missing.append("ALPACA_API_SECRET_KEY")
    return missing


def _creds_ok() -> bool:
    has_key, has_secret = alpaca_credential_status()
    return has_key and has_secret


def _select_execution_engine() -> type[_SimExecutionEngine]:
    """Return the execution engine class based on runtime configuration."""

    try:
        settings = get_execution_settings()
    except Exception as exc:  # pragma: no cover - defensive configuration guard
        _logger.error(
            "EXECUTION_SETTINGS_UNAVAILABLE",
            extra={"error": str(exc)},
        )
        settings = None

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
    reason: str | None = None

    has_key, has_secret = alpaca_credential_status()
    base_url = get_alpaca_base_url()

    if normalized_mode in {"paper", "live"}:
        missing_creds = _missing_creds() or None
        try:
            from .live_trading import ExecutionEngine as _LiveExecutionEngine
        except Exception as exc:  # pragma: no cover - runtime guard
            reason = "import_failed"
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
        reason = "mode_unknown"
        _logger.warning(
            "EXECUTION_MODE_UNKNOWN",
            extra={"requested_mode": mode},
        )

    _logger.info(
        "EXECUTION_ENGINE_SELECTED",
        extra={
            "execution_mode": normalized_mode,
            "engine_class": engine_class_path,
            "shadow_mode": shadow,
            "creds_missing": tuple(missing_creds) if missing_creds else None,
            "reason": reason,
            "has_key": has_key,
            "has_secret": has_secret,
            "base_url": base_url,
        },
    )
    return engine_cls


ExecutionEngine = _select_execution_engine()

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
]

