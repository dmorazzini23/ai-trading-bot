from __future__ import annotations

import importlib
import importlib.util
import logging
import os

from ai_trading.utils.env import refresh_alpaca_credentials_cache

def _ensure_dotenv_module() -> tuple[object | None, bool]:
    """Return (module, resolved) ensuring ``dotenv_values`` availability."""

    try:  # pragma: no cover - import resolution validated by tests
        module = importlib.import_module("dotenv")
    except Exception:  # pragma: no cover - fallback path when python-dotenv missing
        return None, False

    if not hasattr(module, "dotenv_values"):
        def _stub_dotenv_values(*_args, **_kwargs):
            return {}

        setattr(module, "dotenv_values", _stub_dotenv_values)
        return module, False

    spec = importlib.util.find_spec("dotenv")
    resolved = bool(spec and getattr(spec, "origin", None))
    if not resolved:
        resolved = bool(getattr(module, "__file__", None))
    return module, resolved


_dotenv, PYTHON_DOTENV_RESOLVED = _ensure_dotenv_module()


_ENV_LOADED = False


def load_dotenv_if_present(dotenv_path: str = ".env") -> bool:
    """Load ``dotenv_path`` when python-dotenv is available."""

    if not PYTHON_DOTENV_RESOLVED:
        return False
    try:
        _dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - defensive logging is handled upstream
        return False
    return True


def _log_env_loaded(source: str) -> None:
    try:
        logger = logging.getLogger(__name__)
        if not logger.handlers and not logging.getLogger().handlers:
            return
        from ai_trading.logging import get_logger

        lg = get_logger(__name__)
        if source == "<default>":
            lg.info("ENV_LOADED_DEFAULT override=True", extra={"key": "env_loaded:default"})
        else:
            lg.info(
                "ENV_LOADED_FROM override=True",
                extra={"key": f"env_loaded:{source}", "dotenv_path": source},
            )
    except Exception:  # pragma: no cover - keep env init resilient
        return


def ensure_dotenv_loaded(dotenv_path: str | None = None) -> None:
    """Idempotently load environment variables from ``.env`` if available."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if os.getenv("PYTEST_RUNNING") or os.getenv("TESTING"):
        _ENV_LOADED = True
        refresh_alpaca_credentials_cache()
        return
    os.environ.setdefault("MULTI_LOAD_TEST", "safe_value")
    path = dotenv_path or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)), ".env")
    loaded = load_dotenv_if_present(path)
    if not loaded and path != ".env":
        loaded = load_dotenv_if_present()
        source = "<default>" if loaded else "<none>"
    else:
        source = path if loaded else "<none>"
    _ENV_LOADED = True
    if loaded:
        _log_env_loaded(source)
        try:
            refresh_alpaca_credentials_cache()
        except Exception:  # pragma: no cover - keep env init resilient
            pass


__all__ = ["ensure_dotenv_loaded", "load_dotenv_if_present", "PYTHON_DOTENV_RESOLVED"]
