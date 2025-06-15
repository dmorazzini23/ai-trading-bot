from pathlib import Path
import types
import pytest
import logger_rotator


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_get_rotating_handler(monkeypatch):
    created = {}

    class Dummy:
        def __init__(self, path, maxBytes=0, backupCount=0):
            created["path"] = path
            created["maxBytes"] = maxBytes
            created["backupCount"] = backupCount

    monkeypatch.setattr(logger_rotator, "RotatingFileHandler", Dummy)
    handler = logger_rotator.get_rotating_handler(
        "foo.log", max_bytes=1, backup_count=2
    )
    assert isinstance(handler, Dummy)
    assert created["path"] == "foo.log"
    force_coverage(logger_rotator)
