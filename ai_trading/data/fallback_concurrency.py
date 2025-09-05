"""Utilities for running fallback fetches with bounded concurrency.

This module provides a small helper to execute per-symbol tasks in parallel
while keeping track of which symbols have been processed.  Each worker adds its
symbol to the shared ``PROCESSED_SYMBOLS`` set *before* the task completes so
callers can inspect progress or verify coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TypeVar

T = TypeVar("T")

# A module level set recording which symbols have completed processing.
PROCESSED_SYMBOLS: set[str] = set()

# Internal lock to ensure thread-safe updates to the shared set.
_LOCK = Lock()


def run_in_threads(
    symbols: Iterable[str],
    worker: Callable[[str], T],
    *,
    max_workers: int = 4,
) -> dict[str, T | None]:
    """Execute ``worker`` for each symbol using a thread pool.

    Parameters
    ----------
    symbols:
        Iterable of symbol strings to process.
    worker:
        Callable invoked with each symbol.  Its return value becomes the
        value in the result mapping.  Exceptions are caught and result in
        ``None`` entries.
    max_workers:
        Maximum number of threads to spawn.

    Returns
    -------
    dict[str, T | None]
        Mapping of symbol to the worker's return value or ``None`` if the
        worker raised an exception.
    """

    results: dict[str, T | None] = {}

    def _task(sym: str) -> T | None:
        try:
            return worker(sym)
        except Exception:  # pragma: no cover - worker errors become None
            return None
        finally:
            # Ensure the symbol is recorded even if the worker fails.
            with _LOCK:
                PROCESSED_SYMBOLS.add(sym)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fallback") as ex:
        future_to_sym = {ex.submit(_task, s): s for s in symbols}
        for fut in as_completed(future_to_sym):
            sym = future_to_sym[fut]
            results[sym] = fut.result()
    return results
