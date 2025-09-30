from __future__ import annotations

import importlib
import logging
import logging.config
import logging.handlers
import pathlib
import sys
import unittest.mock


def test_safe_clear_dict_restores_import_machinery():
    # Ensure the sitecustomize hook has patched unittest.mock._clear_dict.
    import sitecustomize  # noqa: F401  # pylint: disable=unused-import

    assert unittest.mock._clear_dict.__name__ == "_safe_clear_dict"

    original_modules = dict(sys.modules)
    original_sys = sys.modules["sys"]
    original_importlib = sys.modules["importlib"]
    original_logging = sys.modules["logging"]
    original_pathlib = sys.modules["pathlib"]

    try:
        unittest.mock._clear_dict(sys.modules)

        # Essential modules should be restored or re-imported on demand.
        assert sys.modules["sys"] is original_sys
        assert sys.modules["importlib"] is original_importlib
        assert sys.modules["logging"] is original_logging
        assert sys.modules["pathlib"] is original_pathlib
        assert "logging.config" in sys.modules
        assert "logging.handlers" in sys.modules

        module = importlib.import_module("ai_trading")
        assert module is sys.modules["ai_trading"]

        logger = logging.getLogger(__name__)
        assert logger.name == __name__
        assert hasattr(logging.config, "dictConfig")
        assert hasattr(logging.handlers, "BufferingHandler")
        assert isinstance(pathlib.Path.cwd(), pathlib.Path)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
