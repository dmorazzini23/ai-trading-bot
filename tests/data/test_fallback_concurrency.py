import asyncio

from ai_trading.data.fallback import concurrency


def test_run_with_concurrency_returns_results():
    async def worker(sym: str) -> str:
        await asyncio.sleep(0)
        return sym

    symbols = ["AAPL", "MSFT", "GOOG"]
    results, succeeded, failed = asyncio.run(
        concurrency.run_with_concurrency(symbols, worker, max_concurrency=2)
    )

    assert results and set(results) == set(symbols)
    assert succeeded == set(symbols)
    assert not failed


def test_run_with_concurrency_respects_limit():
    max_seen = 0
    current = 0
    lock = asyncio.Lock()

    async def worker(sym: str) -> str:
        nonlocal max_seen, current
        async with lock:
            current += 1
            if current > max_seen:
                max_seen = current
        await asyncio.sleep(0.01)
        async with lock:
            current -= 1
        return sym

    symbols = [f"SYM{i}" for i in range(5)]
    asyncio.run(concurrency.run_with_concurrency(symbols, worker, max_concurrency=2))

    assert max_seen <= 2
    assert concurrency.PEAK_SIMULTANEOUS_WORKERS <= 2
