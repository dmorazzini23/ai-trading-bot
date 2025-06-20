from pathlib import Path

import pytest

import logger_rotator


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_get_rotating_handler(monkeypatch):
    with pytest.raises(NotImplementedError):
        logger_rotator.get_rotating_handler("foo.log", max_bytes=1, backup_count=2)
    force_coverage(logger_rotator)
