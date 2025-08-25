import logging
import os
import re
from typing import Any

SENSITIVE_KEYS = {
    'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'API_KEY',
    'SECRET_KEY', 'BEARER_TOKEN', 'AUTH_TOKEN'
}
TOKEN_RE = re.compile('([A-Za-z0-9_\\-]{12,})')

def _mask(val: str) -> str:
    if not val:
        return val
    if len(val) <= 8:
        return '***'
    return val[:4] + '…' + val[-4:]

class SecretFilter(logging.Filter):
    """Masks secrets in log records and structured extras."""  # AI-AGENT-REF: scrub structured extras

    SECRET_KEYS = {"api_key", "api_secret", "secret", "token", "password"}

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(getattr(record, 'msg', ''))
            for k in SENSITIVE_KEYS:
                env = os.getenv(k)
                if env and env in msg:
                    msg = msg.replace(env, _mask(env))
            msg = TOKEN_RE.sub(lambda m: _mask(m.group(1)), msg)
            record.msg = msg

            for k in list(record.__dict__.keys()):
                lk = str(k).lower()
                if any(s in lk for s in self.SECRET_KEYS):
                    record.__dict__[k] = "***REDACTED***"
        except Exception:
            pass
        return True
