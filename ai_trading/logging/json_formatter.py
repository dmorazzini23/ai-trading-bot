from __future__ import annotations

import json
import logging
import traceback
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.exc import COMMON_EXC

try:  # pragma: no cover - defensive import guard
    import unittest.mock as _mock_mod
except Exception:  # pragma: no cover - fallback when mock is unavailable
    _MOCK_TYPES: tuple[type[Any], ...] = ()
else:  # pragma: no cover - exercised via logging tests
    _MOCK_TYPES = tuple(
        getattr(_mock_mod, name)
        for name in ("Mock", "MagicMock", "NonCallableMock", "AsyncMock", "PropertyMock")
        if hasattr(_mock_mod, name)
    )


def _is_mock(obj: Any) -> bool:
    return bool(_MOCK_TYPES) and isinstance(obj, _MOCK_TYPES)


def _mask_secret(value: str) -> str:
    """Non-throwing redactor for secret-like values (config-independent)."""
    try:
        s = '' if value is None else str(value)
        n = len(s)
        if n == 0:
            return ''
        if n <= 4:
            return '***'
        return f'***{s[-4:]}'
    except COMMON_EXC:
        return '***'


_UTC = ZoneInfo("UTC")


class JSONFormatter(logging.Formatter):
    """JSON log formatter with optional extra fields and masking."""

    def __init__(
        self,
        datefmt: str | None = None,
        *,
        extra_fields: dict[str, Any] | None = None,
        mask_keys: list[str] | None = None,
    ) -> None:
        super().__init__(fmt=None, datefmt=datefmt)
        self._extra_fields = extra_fields or {}
        self._mask_keys = {k.lower() for k in (mask_keys or [])}

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Render timestamps in UTC using :mod:`zoneinfo` aware datetimes."""

        dt = datetime.fromtimestamp(record.created, tz=_UTC)
        formatted = dt.isoformat(timespec="milliseconds")
        if formatted.endswith("+00:00"):
            formatted = f"{formatted[:-6]}Z"
        elif not formatted.endswith("Z"):
            formatted = f"{formatted}Z"
        return formatted

    def _json_default(self, obj: Any) -> Any:
        """Fallback serialization for unsupported types."""
        if isinstance(obj, datetime | date):
            return obj.isoformat()
        if _is_mock(obj):
            return repr(obj)
        tolist_attr = getattr(obj, "tolist", None)
        if callable(tolist_attr) and not _is_mock(tolist_attr):
            try:
                return tolist_attr()
            except COMMON_EXC:
                pass
            except Exception:
                pass
        return str(obj)

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - exercised in tests
        payload: dict[str, Any] = {
            'ts': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'name': record.name,
            'msg': record.getMessage(),
        }
        omit = {
            'msg', 'message', 'args', 'levelname', 'levelno', 'name', 'created',
            'msecs', 'relativeCreated', 'asctime', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno',
            'funcName', 'thread', 'threadName', 'processName', 'process',
            'taskName'
        }

        def _should_mask_secret(field: str, value: object) -> bool:
            lk = field.lower()
            if lk.startswith('has_'):
                return False
            if not isinstance(value, (str, bytes)):
                return False
            sensitive_tokens = (
                'api_key', 'secret_key', 'apca_api_key_id', 'apca_api_secret_key',
                'token', 'password', 'bearer', 'private', 'access_key'
            )
            return any(tok in lk for tok in sensitive_tokens)

        for k, v in record.__dict__.items():
            if k in omit:
                continue
            if k.lower() in self._mask_keys:
                v = '***'
            elif _should_mask_secret(k, v):
                v = _mask_secret(v)  # type: ignore[arg-type]
            payload[k] = v

        for k, v in self._extra_fields.items():
            payload[k] = '***' if k.lower() in self._mask_keys else v

        if record.exc_info:
            exc_type, exc_value, _exc_tb = record.exc_info
            payload['exc'] = ''.join(
                traceback.format_exception_only(exc_type, exc_value)
            ).strip()
        return json.dumps(payload, default=self._json_default, ensure_ascii=False)
