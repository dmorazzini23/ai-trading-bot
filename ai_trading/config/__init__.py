from .settings import Settings, get_settings, broker_keys  # noqa: F401
from .alpaca import get_alpaca_config, AlpacaConfig  # noqa: F401
from .management import TradingConfig  # AI-AGENT-REF: expose TradingConfig
import os
from typing import Any
import logging

logger = logging.getLogger(__name__)


def reload_env() -> None:
    """Reload .env if python-dotenv is present; ignore failures."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass


def get_env(name: str, default: Any = None, *, reload: bool = False, required: bool = False) -> Any:
    """Return env var; if reload=True, call reload_env() first."""
    if reload:
        reload_env()
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _require_env_vars(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.critical(msg)
        raise RuntimeError(msg)


def validate_environment() -> None:
    """
    Validate that required environment/config values are available and non-empty.
    Raise RuntimeError on failure. Should be idempotent, no blocking/locks.
    """
    s = get_settings()
    missing = []
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "WEBHOOK_SECRET"]
    env = os.environ
    for k in required:
        v = env.get(k, "") or getattr(
            s,
            {
                "ALPACA_API_KEY": "alpaca_api_key",
                "ALPACA_SECRET_KEY": "alpaca_secret_key",
                "ALPACA_BASE_URL": "alpaca_base_url",
                "WEBHOOK_SECRET": "webhook_secret",
            }[k],
            "",
        )
        if not isinstance(v, str) or not v.strip():
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def validate_alpaca_credentials() -> None:
    """
    Raise RuntimeError if ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL
    are missing or empty. Prefer module attributes if present (tests patch them),
    otherwise read os.environ or settings.
    """
    api = (
        globals().get("ALPACA_API_KEY")
        or os.getenv("ALPACA_API_KEY", "")
        or getattr(get_settings(), "alpaca_api_key", "")
    )
    sec = (
        globals().get("ALPACA_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY", "")
        or getattr(get_settings(), "alpaca_secret_key", "")
    )
    url = (
        globals().get("ALPACA_BASE_URL")
        or os.getenv("ALPACA_BASE_URL", "")
        or getattr(get_settings(), "alpaca_base_url", "")
    )
    if not (str(api).strip() and str(sec).strip() and str(url).strip()):
        raise RuntimeError("Alpaca credentials are missing or empty")


def validate_env_vars() -> None:
    return validate_environment()


def log_config(secrets_to_redact: list[str] | None = None) -> dict:
    """
    Return a sanitized snapshot of current config for diagnostics.
    MUST NOT log or print in tests.
    """
    s = get_settings()
    conf = {
        "ALPACA_API_KEY": "***" if s.alpaca_api_key else "",
        "ALPACA_SECRET_KEY": "***REDACTED***" if s.alpaca_secret_key else "",
        "ALPACA_BASE_URL": s.alpaca_base_url or "",
        "CAPITAL_CAP": getattr(s, "capital_cap", None) or 0.25,
        "CONF_THRESHOLD": getattr(s, "conf_threshold", None) or 0.75,
        "DAILY_LOSS_LIMIT": getattr(s, "daily_loss_limit", None) or 0.03,
    }
    if secrets_to_redact:
        for key in secrets_to_redact:
            if key in conf:
                conf[key] = "***"
    return conf

__all__ = [
    "Settings",
    "get_settings",
    "broker_keys",
    "get_alpaca_config",
    "AlpacaConfig",
    "TradingConfig",
    "get_env",
    "_require_env_vars",
    "reload_env",
    "validate_environment",
    "validate_alpaca_credentials",
    "validate_env_vars",
    "log_config",
]

