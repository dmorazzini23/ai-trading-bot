"""Async helpers for running fallback fetches with bounded concurrency."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, List, Tuple, Any


async def run_with_concurrency(
    jobs: Iterable[Callable[[], Awaitable[Any]]],
    limit: int,
) -> Tuple[List[Any], int, int]:
    """Execute *jobs* concurrently while keeping at most ``limit`` tasks in flight."""

    max_concurrency = max(int(limit or 1), 1)
    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[Any] = []
    succeeded = failed = 0

    async def _run(job: Callable[[], Awaitable[Any]]):
        await semaphore.acquire()
        try:
            value = await job()
            return True, value
        except Exception as exc:  # pragma: no cover - surfaced to caller
            return False, exc
        finally:
            semaphore.release()

    tasks = [_run(job) for job in jobs]
    if not tasks:
        return [], 0, 0

    for ok, value in await asyncio.gather(*tasks, return_exceptions=False):
        results.append(value)
        if ok:
            succeeded += 1
        else:
            failed += 1
    return results, succeeded, failed


__all__ = ["run_with_concurrency"]
