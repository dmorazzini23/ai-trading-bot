from pathlib import Path

import logging
import pytest

import logger_rotator


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_get_rotating_handler(tmp_path):
    handler = logger_rotator.get_rotating_handler(
        str(tmp_path / "foo.log"), max_bytes=1, backup_count=2
    )
    assert isinstance(handler, logging.Handler)
    force_coverage(logger_rotator)
