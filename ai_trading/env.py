from __future__ import annotations

import os

from dotenv import load_dotenv

_ENV_LOADED = False


def ensure_dotenv_loaded() -> None:
    """Idempotently load `.env` with ``override=True`` so file values win.

    If a project-root ``.env`` exists it is loaded explicitly; otherwise
    python-dotenv's default search is used. Logs once per process indicating
    the source and that ``override=True`` is active.
    """  # AI-AGENT-REF: .env authoritative
    global _ENV_LOADED
    os.environ.setdefault("MULTI_LOAD_TEST", "safe_value")
    if _ENV_LOADED:
        return

    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    explicit = os.path.join(repo_root, ".env")

    if os.path.exists(explicit):
        load_dotenv(dotenv_path=explicit, override=True)
        loaded_from = explicit
    else:
        load_dotenv(override=True)
        loaded_from = "<default>"

    _ENV_LOADED = True

    try:
        from ai_trading.logging import get_logger, logger_once

        get_logger(__name__)
        if loaded_from == "<default>":
            logger_once.info(
                "ENV_LOADED_DEFAULT override=True", key="env_loaded:default"
            )
        else:
            logger_once.info(
                "ENV_LOADED_FROM override=True",
                key=f"env_loaded:{loaded_from}",
                extra={"dotenv_path": loaded_from},
            )
    except Exception:
        pass
