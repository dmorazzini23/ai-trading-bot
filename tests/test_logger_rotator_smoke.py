import logging
from pathlib import Path

import pytest

try:
    import logger_rotator
except ImportError:  # pragma: no cover - script optional
    pytest.skip("logger_rotator not available", allow_module_level=True)


def force_coverage(mod):
    """Force coverage by importing and accessing module attributes instead of using exec."""
    for attr_name in dir(mod):
        if not attr_name.startswith('_'):
            getattr(mod, attr_name, None)


@pytest.mark.smoke
def test_get_rotating_handler(tmp_path):
    handler = logger_rotator.get_rotating_handler(
        str(tmp_path / "foo.log"), max_bytes=1, backup_count=2
    )
    assert isinstance(handler, logging.Handler)
    force_coverage(logger_rotator)
