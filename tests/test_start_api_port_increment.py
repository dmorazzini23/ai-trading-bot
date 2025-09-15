import socket

import pytest

from ai_trading import main


def test_start_api_raises_when_port_busy(monkeypatch):
    """Configured API port conflicts should raise an error instead of retrying."""
    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    start_port = probe.getsockname()[1]
    probe.close()

    blocker = socket.socket()
    blocker.bind(("0.0.0.0", start_port))
    blocker.listen(1)

    class DummySettings:
        def __init__(self, api_port):
            self.api_port = api_port

    monkeypatch.setattr(main, "get_settings", lambda: DummySettings(start_port))
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    def _fail(*_args, **_kwargs):
        raise AssertionError("run should not be invoked")

    monkeypatch.setattr(main, "run_flask_app", _fail)

    try:
        with pytest.raises(main.PortInUseError) as excinfo:
            main.start_api()
    finally:
        blocker.close()

    assert excinfo.value.port == start_port
