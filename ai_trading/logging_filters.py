import logging
import os
import re
from typing import Any

from ai_trading.logging.redact import _ENV_MASK

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

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            for k in list(record.__dict__.keys()):
                lk = str(k).lower()
                if any(s in lk for s in self.SECRET_KEYS):
                    record.__dict__[k] = _ENV_MASK
        except Exception:
            pass
        return True
