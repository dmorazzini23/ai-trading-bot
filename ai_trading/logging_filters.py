from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
import logging
import re
import sys
from typing import Any

from ai_trading.logging.redact import _ENV_MASK
from typing import Iterable

SENSITIVE_KEYS = {
    'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'API_KEY',
    'SECRET_KEY', 'BEARER_TOKEN', 'AUTH_TOKEN'
}
TOKEN_RE = re.compile('([A-Za-z0-9_\\-]{12,})')

def _mask(val: str) -> str:
    return _ENV_MASK if val else val


def _runtime_env_snapshot() -> dict[str, str]:
    mgmt_mod = sys.modules.get("ai_trading.config.management")
    snapshot_getter = getattr(mgmt_mod, "merged_env_snapshot", None) if mgmt_mod is not None else None
    if not callable(snapshot_getter):
        return {}
    try:
        snapshot = snapshot_getter()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return {}
    return {
        str(k): str(v)
        for k, v in snapshot.items()
        if isinstance(k, str) and isinstance(v, str)
    }


class SecretFilter(logging.Filter):
    """Masks secrets in log records and structured extras."""  # AI-AGENT-REF: scrub structured extras

    SECRET_KEYS = {"api_key", "api_secret", "secret", "token", "password"}

    def _candidate_values(self) -> set[str]:
        """Collect potential secret values from environment for masking."""
        vals: set[str] = set()
        try:
            for k, v in _runtime_env_snapshot().items():
                if not v:
                    continue
                kl = str(k).upper()
                if (
                    kl in SENSITIVE_KEYS
                    or "KEY" in kl
                    or "SECRET" in kl
                    or "TOKEN" in kl
                ):
                    vals.add(str(v))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return vals
        return vals

    def _mask_in_text(self, text: str, candidates: Iterable[str]) -> str:
        out = text
        try:
            for val in sorted(candidates, key=len, reverse=True):
                if val and val in out:
                    out = out.replace(val, _ENV_MASK)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return text
        return out

    def _mask_value(self, value: Any, candidates: Iterable[str]) -> Any:
        if isinstance(value, str):
            return self._mask_in_text(value, candidates)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Mask structured fields by key
            for k in list(record.__dict__.keys()):
                lk = str(k).lower()
                if lk.startswith("has_"):
                    continue
                if any(s in lk for s in self.SECRET_KEYS):
                    record.__dict__[k] = _ENV_MASK
            # Sanitize positional/keyword args only; avoid mutating format strings
            candidates = self._candidate_values()
            if candidates and isinstance(record.msg, str) and not record.args:
                record.msg = self._mask_in_text(record.msg, candidates)
            if record.args and candidates:
                try:
                    if isinstance(record.args, tuple):
                        record.args = tuple(
                            self._mask_value(a, candidates)
                            for a in record.args
                        )
                    elif isinstance(record.args, dict):
                        for k, v in list(record.args.items()):
                            record.args[k] = self._mask_value(v, candidates)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    pass
            if candidates:
                try:
                    formatted = record.getMessage()
                    redacted = self._mask_in_text(formatted, candidates)
                    if redacted != formatted:
                        record.msg = redacted
                        record.args = ()
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    pass
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass
        return True
