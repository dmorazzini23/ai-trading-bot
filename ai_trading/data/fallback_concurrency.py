"""Compatibility shim for legacy fallback concurrency helpers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ai_trading.data.fallback import concurrency as _concurrency

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from ai_trading.data.fallback.concurrency import (
        FAILED_SYMBOLS,
        PEAK_SIMULTANEOUS_WORKERS,
        SUCCESSFUL_SYMBOLS,
        reset_peak_simultaneous_workers,
        reset_tracking_state,
        run_with_concurrency,
        run_with_concurrency_limit,
        _collect_asyncio_primitive_types,
        _get_effective_host_limit,
        _get_host_limit_semaphore,
        _maybe_recreate_lock,
        _normalise_pooling_state,
    )

# Ensure ``ai_trading.data.fallback_concurrency`` resolves to the modern
# implementation while preserving backwards-compatible attribute access and
# mutation semantics.
__doc__ = _concurrency.__doc__
__all__ = getattr(_concurrency, "__all__", ())

sys.modules[__name__] = _concurrency
