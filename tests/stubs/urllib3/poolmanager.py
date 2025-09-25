from __future__ import annotations

"""Minimal urllib3.poolmanager stub for requests compatibility."""

from types import SimpleNamespace


class PoolManager:
    def __init__(self, num_pools: int = 0, maxsize: int = 0, block: bool = False, **kwargs):
        self.num_pools = num_pools
        self.maxsize = maxsize
        self.block = block
        self.pool_kwargs = dict(kwargs)

    def clear(self) -> None:  # pragma: no cover - trivial stub
        return None

    def connection_from_host(self, host: str, port: int | None = None, scheme: str = "http") -> SimpleNamespace:
        return SimpleNamespace(host=host, port=port, scheme=scheme)

    def connection_from_url(self, url: str) -> SimpleNamespace:
        return SimpleNamespace(url=url)


def proxy_from_url(*args, **kwargs) -> PoolManager:
    return PoolManager(**kwargs)


__all__ = ["PoolManager", "proxy_from_url"]
