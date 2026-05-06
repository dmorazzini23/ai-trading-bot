from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import importlib.util
import logging
from pathlib import Path

from ai_trading.config.management import get_env, is_test_runtime, set_runtime_env_override
from ai_trading.utils.env import refresh_alpaca_credentials_cache

def _ensure_dotenv_module() -> tuple[object | None, bool]:
    """Return (module, resolved) ensuring ``dotenv_values`` availability."""

    try:  # pragma: no cover - import resolution validated by tests
        module = importlib.import_module("dotenv")
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - fallback path when python-dotenv missing
        logging.getLogger(__name__).debug("DOTENV_IMPORT_FAILED", exc_info=True)
        return None, False

    if not hasattr(module, "dotenv_values"):
        def _stub_dotenv_values(*_args: object, **_kwargs: object) -> dict[str, str]:
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
_DOTENV_OVERRIDE_DEFAULT = False
_DOTENV_RUNTIME_OVERRIDE_FLAG = "AI_TRADING_DOTENV_RUNTIME_OVERRIDE"


def _runtime_dotenv_override_enabled() -> bool:
    raw = get_env(_DOTENV_RUNTIME_OVERRIDE_FLAG, "", cast=str, resolve_aliases=False)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_runtime_dotenv_path(dotenv_path: str) -> str:
    raw = get_env("AI_TRADING_RUNTIME_ENV_PATH", "", cast=str, resolve_aliases=False)
    if str(raw or "").strip():
        return str(Path(str(raw).strip()).expanduser())
    return str(Path(dotenv_path).with_name(".env.runtime"))


def _populate_blank_values_from_dotenv(dotenv_path: str) -> None:
    """Fill explicitly blank process values without replacing configured ones."""

    if not PYTHON_DOTENV_RESOLVED:
        return
    try:
        if hasattr(_dotenv, "dotenv_values"):
            values = _dotenv.dotenv_values(dotenv_path)  # type: ignore[union-attr]
        else:
            values = {}
            for raw in Path(dotenv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                if key:
                    values[key] = value
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logging.getLogger(__name__).debug(
            "DOTENV_VALUES_LOAD_FAILED",
            extra={"dotenv_path": dotenv_path},
            exc_info=True,
        )
        return
    for key, value in values.items():
        if value is None:
            continue
        env_key = str(key)
        if get_env(env_key, None, cast=str, resolve_aliases=False) == "":
            set_runtime_env_override(env_key, str(value))


def load_dotenv_if_present(
    dotenv_path: str = ".env",
    *,
    override: bool = _DOTENV_OVERRIDE_DEFAULT,
) -> bool:
    """Load ``dotenv_path`` when python-dotenv is available."""

    if not PYTHON_DOTENV_RESOLVED:
        return False
    try:
        _dotenv.load_dotenv(dotenv_path=dotenv_path, override=override)  # type: ignore[union-attr]
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - defensive logging is handled upstream
        logging.getLogger(__name__).debug(
            "DOTENV_LOAD_FAILED",
            extra={"dotenv_path": dotenv_path},
            exc_info=True,
        )
        return False
    return True


def _log_env_loaded(source: str, *, override: bool) -> None:
    try:
        logger = logging.getLogger(__name__)
        if not logger.handlers and not logging.getLogger().handlers:
            return
        from ai_trading.logging import get_logger

        lg = get_logger(__name__)
        override_text = "override=True" if override else "override=False"
        if source == "<default>":
            lg.info(
                f"ENV_LOADED_DEFAULT {override_text}",
                extra={"key": "env_loaded:default"},
            )
        else:
            lg.info(
                f"ENV_LOADED_FROM {override_text}",
                extra={"key": f"env_loaded:{source}", "dotenv_path": source},
            )
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - keep env init resilient
        logging.getLogger(__name__).debug("ENV_LOADED_LOG_EMIT_FAILED", exc_info=True)
        return


def ensure_dotenv_loaded(dotenv_path: str | None = None) -> None:
    """Idempotently load environment variables from ``.env`` if available."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    # Never load `.env` during pytest collection/execution; tests must be
    # hermetic and not depend on developer/production environment files.
    if is_test_runtime(include_pytest_module=True):
        _ENV_LOADED = True
        refresh_alpaca_credentials_cache()
        return
    set_runtime_env_override("MULTI_LOAD_TEST", "safe_value")
    path = dotenv_path or str(Path(__file__).resolve().parents[2] / ".env")
    runtime_path = _resolve_runtime_dotenv_path(path)
    runtime_loaded = False
    runtime_loaded_now = False
    runtime_override = _runtime_dotenv_override_enabled()
    if Path(runtime_path).exists():
        runtime_loaded_now = load_dotenv_if_present(runtime_path, override=runtime_override)
        if runtime_loaded_now and not runtime_override:
            _populate_blank_values_from_dotenv(runtime_path)
        runtime_loaded = runtime_loaded_now
    loaded = load_dotenv_if_present(path, override=_DOTENV_OVERRIDE_DEFAULT)
    if not loaded and path != ".env":
        loaded = load_dotenv_if_present(override=_DOTENV_OVERRIDE_DEFAULT)
        source = "<default>" if loaded else "<none>"
    else:
        source = path if loaded else "<none>"
    _ENV_LOADED = True
    if runtime_loaded_now:
        _log_env_loaded(runtime_path, override=runtime_override)
    if loaded:
        _log_env_loaded(source, override=_DOTENV_OVERRIDE_DEFAULT)
    if runtime_loaded or loaded:
        try:
            refresh_alpaca_credentials_cache()
        except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - keep env init resilient
            logging.getLogger(__name__).debug("ALPACA_CREDENTIAL_CACHE_REFRESH_FAILED", exc_info=True)


__all__ = ["ensure_dotenv_loaded", "load_dotenv_if_present", "PYTHON_DOTENV_RESOLVED"]
