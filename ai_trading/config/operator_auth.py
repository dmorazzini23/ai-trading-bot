"""Shared parsing helpers for operator authentication settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def parse_operator_token_map(raw: Any) -> dict[str, str]:
    """Parse JSON or comma-separated operator token mappings."""

    if isinstance(raw, Mapping):
        parsed_mapping: dict[str, str] = {}
        for key, value in raw.items():
            operator_id = str(key or "").strip().lower()
            token = str(value or "").strip()
            if operator_id and token:
                parsed_mapping[operator_id] = token
        return parsed_mapping

    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if isinstance(parsed_json, Mapping):
        parsed_payload: dict[str, str] = {}
        for key, value in parsed_json.items():
            operator_id = str(key or "").strip().lower()
            token = str(value or "").strip()
            if operator_id and token:
                parsed_payload[operator_id] = token
        return parsed_payload

    parsed_fallback: dict[str, str] = {}
    for item in text.split(","):
        token = str(item or "").strip()
        if not token:
            continue
        if "=" in token:
            operator_id, secret = token.split("=", 1)
        elif ":" in token:
            operator_id, secret = token.split(":", 1)
        else:
            continue
        operator_key = str(operator_id or "").strip().lower()
        secret_value = str(secret or "").strip()
        if operator_key and secret_value:
            parsed_fallback[operator_key] = secret_value
    return parsed_fallback


__all__ = ["parse_operator_token_map"]
