from __future__ import annotations

"""Minimal urllib3.util stub for requests compatibility."""

from types import SimpleNamespace
from urllib.parse import urlparse
import base64


def make_headers(
    keep_alive: bool | None = None,
    accept_encoding: str | None = None,
    user_agent: str | None = None,
    proxy_basic_auth: str | None = None,
    proxy_auth: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if keep_alive:
        headers["connection"] = "keep-alive"
    if accept_encoding:
        if accept_encoding is True:
            headers["accept-encoding"] = "gzip, deflate, br"
        else:
            headers["accept-encoding"] = str(accept_encoding)
    if user_agent:
        headers["user-agent"] = user_agent
    auth_value = proxy_auth or proxy_basic_auth
    if auth_value:
        token = base64.b64encode(auth_value.encode()).decode()
        headers["proxy-authorization"] = f"Basic {token}"
    return headers


def parse_url(url: str) -> SimpleNamespace:
    parsed = urlparse(url)
    return SimpleNamespace(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=parsed.port,
        path=parsed.path,
        query=parsed.query,
    )


class Timeout:
    def __init__(self, total: float | None = None, connect: float | None = None, read: float | None = None):
        self.total = total
        self.connect = connect
        self.read = read

    @classmethod
    def from_int(cls, value: int | float | None) -> "Timeout":
        return cls(total=value)


__all__ = ["make_headers", "parse_url", "Timeout"]
