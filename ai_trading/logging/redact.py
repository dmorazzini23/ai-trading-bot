from __future__ import annotations

# Utility to redact sensitive fields in mappings.  # AI-AGENT-REF
import re
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any

_RE_KEYS = re.compile(r"(key|secret|token|password)", re.IGNORECASE)
_MASK = "***REDACTED***"


def _redact_inplace(obj: Any) -> Any:
    """Recursively redact matching keys."""  # AI-AGENT-REF

    if isinstance(obj, Mapping):
        for k, v in list(obj.items()):
            if isinstance(k, str) and _RE_KEYS.search(k):
                obj[k] = _MASK
            else:
                obj[k] = _redact_inplace(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _redact_inplace(v)
        return obj
    return obj


def redact(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a redacted copy of *payload*."""  # AI-AGENT-REF

    dup: MutableMapping[str, Any] = deepcopy(payload)  # type: ignore[assignment]
    return _redact_inplace(dup)

