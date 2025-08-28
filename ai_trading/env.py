from __future__ import annotations
import os

_ENV_LOADED = False

def ensure_dotenv_loaded() -> None:
    """Idempotently load `.env` with override=True so file values win."""
    global _ENV_LOADED
    os.environ.setdefault('MULTI_LOAD_TEST', 'safe_value')
    if _ENV_LOADED:
        return
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    explicit = os.path.join(repo_root, '.env')
    try:
        from dotenv import load_dotenv  # type: ignore
    except ModuleNotFoundError:
        return
    if os.path.exists(explicit):
        load_dotenv(dotenv_path=explicit, override=True)
        loaded_from = explicit
    else:
        load_dotenv(override=True)
        loaded_from = '<default>'
    _ENV_LOADED = True
    try:
        import logging as _logging
        if _logging.getLogger().handlers:
            from ai_trading.logging import get_logger, logger_once
            get_logger(__name__)
            if loaded_from == '<default>':
                logger_once.info('ENV_LOADED_DEFAULT override=True', key='env_loaded:default')
            else:
                logger_once.info('ENV_LOADED_FROM override=True', key=f'env_loaded:{loaded_from}', extra={'dotenv_path': loaded_from})
    except (KeyError, ValueError, TypeError):
        pass
