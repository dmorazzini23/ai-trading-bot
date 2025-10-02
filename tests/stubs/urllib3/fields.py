"""Minimal urllib3.fields stub providing RequestField for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class RequestField:
    """Stub implementation matching urllib3.fields.RequestField API surface."""

    name: str | None = None
    data: Any = None
    headers: Mapping[str, str] | None = None

    @classmethod
    def from_tuples(
        cls,
        fieldname: str,
        value: Any,
        header_formatter: Any | None = None,
    ) -> "RequestField":
        field = cls(name=fieldname, data=value, headers={})
        return field

    def make_multipart(
        self,
        content_disposition: str | None = None,
        content_type: str | None = None,
        header_formatter: Any | None = None,
    ) -> None:
        # No-op for tests; real implementation configures headers for multipart forms.
        return None


__all__ = ["RequestField"]
