from __future__ import annotations

"""ThreadPool executors for compute/prediction with resource guardrails.

This module centralizes creation and cleanup of thread pools. It is imported
by core.bot_engine and other modules to avoid oversubscription on 1 vCPU
targets and to provide simple, testable behavior.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ai_trading.logging import get_logger
from ai_trading.utils.exec import get_worker_env_override

logger = get_logger(__name__)

# Module-level executors
executor: Optional[ThreadPoolExecutor] = None
prediction_executor: Optional[ThreadPoolExecutor] = None


def _ensure_executors() -> None:
    """Create thread pool executors on demand with clamped worker counts."""
    global executor, prediction_executor
    if executor is not None and prediction_executor is not None:
        return
    from ai_trading.config.settings import get_settings

    cpu = os.cpu_count() or 2
    s = get_settings()
    exec_fn = getattr(s, "effective_executor_workers", None)
    # Base default for 1 vCPU target machines is 1–2 workers to avoid oversubscription
    default_workers = max(1, min(2, cpu))
    # Allow env override for emergency tuning
    env_exec = get_worker_env_override(
        "AI_TRADING_EXEC_WORKERS", fallback_keys=("EXECUTOR_WORKERS",)
    )
    env_pred = get_worker_env_override(
        "AI_TRADING_PRED_WORKERS", fallback_keys=("PREDICTION_WORKERS",)
    )
    exec_workers = exec_fn(cpu) if callable(exec_fn) else (env_exec or default_workers)
    pred_fn = getattr(s, "effective_prediction_workers", None)
    pred_workers = pred_fn(cpu) if callable(pred_fn) else (env_pred or default_workers)
    # Clamp to [1, cpu] to respect resource guardrails
    try:
        exec_workers = max(1, min(int(exec_workers), max(1, cpu)))
    except Exception:
        exec_workers = default_workers
    try:
        pred_workers = max(1, min(int(pred_workers), max(1, cpu)))
    except Exception:
        pred_workers = default_workers
    executor = ThreadPoolExecutor(max_workers=exec_workers)
    prediction_executor = ThreadPoolExecutor(max_workers=pred_workers)


def cleanup_executors() -> None:
    """Cleanup ThreadPoolExecutor resources to prevent resource leaks."""
    global executor, prediction_executor
    try:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            logger.debug("Main executor shutdown successfully")
    except Exception as e:  # defensive: never raise in cleanup
        logger.warning("Error shutting down main executor: %s", e)

    try:
        if prediction_executor is not None:
            prediction_executor.shutdown(wait=True, cancel_futures=True)
            logger.debug("Prediction executor shutdown successfully")
    except Exception as e:  # defensive: never raise in cleanup
        logger.warning("Error shutting down prediction executor: %s", e)

