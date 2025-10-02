"""Minimal urllib3.filepost stub for dependency-light test imports."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple


def encode_multipart_formdata(
    fields: Iterable[Tuple[str, Any]] | Mapping[str, Any],
    boundary: str | None = None,
) -> Tuple[bytes, str]:
    """Return placeholder multipart payload and content type."""

    if isinstance(fields, dict):
        items = list(fields.items())
    else:
        items = list(fields)
    body = str(items).encode("utf-8")
    content_type = "multipart/form-data"
    if boundary:
        content_type += f"; boundary={boundary}"
    return body, content_type


def choose_boundary() -> str:
    return "stub-boundary"


__all__ = ["encode_multipart_formdata", "choose_boundary"]
