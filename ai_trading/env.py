from __future__ import annotations
import os
import importlib.util
from pathlib import Path
import sys

# AI-AGENT-REF: optional dotenv via optdeps without heavy package import
_spec = importlib.util.spec_from_file_location(
    "_optdeps", Path(__file__).resolve().parent / "utils" / "optdeps.py"
)
_optdeps = importlib.util.module_from_spec(_spec)
sys.modules["_optdeps"] = _optdeps
assert _spec.loader is not None
_spec.loader.exec_module(_optdeps)
optional_import = _optdeps.optional_import
module_ok = _optdeps.module_ok

load_dotenv = optional_import(
    "dotenv", attr="load_dotenv", purpose=".env loading", extra="pip install python-dotenv"
)
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
    if not module_ok(load_dotenv):
        return
    if os.path.exists(explicit):
        load_dotenv(dotenv_path=explicit, override=True)
        loaded_from = explicit
    else:
        load_dotenv(override=True)
        loaded_from = '<default>'
    _ENV_LOADED = True
    try:
        from ai_trading.logging import get_logger, logger_once
        get_logger(__name__)
        if loaded_from == '<default>':
            logger_once.info('ENV_LOADED_DEFAULT override=True', key='env_loaded:default')
        else:
            logger_once.info('ENV_LOADED_FROM override=True', key=f'env_loaded:{loaded_from}', extra={'dotenv_path': loaded_from})
    except (KeyError, ValueError, TypeError):
        pass
