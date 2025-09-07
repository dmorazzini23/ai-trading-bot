import asyncio
from tests.conftest import reload_module


def test_host_limit_enforced(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = reload_module("ai_trading.http.pooling")
    sem = pooling.get_host_semaphore()

    current = 0
    max_seen = 0

    async def worker():
        nonlocal current, max_seen
        async with sem:
            current += 1
            max_seen = max(max_seen, current)
            await asyncio.sleep(0.01)
            current -= 1

    async def main():
        await asyncio.gather(*(worker() for _ in range(5)))

    asyncio.run(main())
    assert max_seen == 2

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    reload_module(pooling)
