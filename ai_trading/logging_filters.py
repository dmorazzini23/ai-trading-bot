import logging
import os
import re
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

class SecretFilter(logging.Filter):
    """Masks secrets in log records and structured extras."""  # AI-AGENT-REF: scrub structured extras

    SECRET_KEYS = {"api_key", "api_secret", "secret", "token", "password"}

    def _candidate_values(self) -> set[str]:
        """Collect potential secret values from environment for masking."""
        vals: set[str] = set()
        try:
            for k, v in os.environ.items():
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
        except Exception:
            return vals
        return vals

    def _mask_in_text(self, text: str, candidates: Iterable[str]) -> str:
        out = text
        try:
            for val in candidates:
                if val and val in out:
                    out = out.replace(val, _ENV_MASK)
        except Exception:
            return text
        return out

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Mask structured fields by key
            for k in list(record.__dict__.keys()):
                lk = str(k).lower()
                if any(s in lk for s in self.SECRET_KEYS):
                    record.__dict__[k] = _ENV_MASK
            # Sanitize positional/keyword args only; avoid mutating format strings
            candidates = self._candidate_values()
            if record.args and candidates:
                try:
                    if isinstance(record.args, tuple):
                        record.args = tuple(
                            _ENV_MASK if (isinstance(a, str) and a in candidates) else a
                            for a in record.args
                        )
                    elif isinstance(record.args, dict):
                        for k, v in list(record.args.items()):
                            if isinstance(v, str) and v in candidates:
                                record.args[k] = _ENV_MASK
                except Exception:
                    pass
        except Exception:
            pass
        return True
