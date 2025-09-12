import socket

from ai_trading import main, app


def test_start_api_increments_port_when_busy(monkeypatch):
    # Find an unused port and occupy it to simulate a busy port
    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    start_port = probe.getsockname()[1]
    probe.close()
    blocker = socket.socket()
    blocker.bind(("0.0.0.0", start_port))
    blocker.listen(1)

    captured = {}

    class DummyApp:
        def run(self, host, port, debug=False, **kwargs):
            captured["host"] = host
            captured["port"] = port
            captured["debug"] = debug

    class DummySettings:
        def __init__(self, api_port):
            self.api_port = api_port

    monkeypatch.setattr(main, "get_settings", lambda: DummySettings(start_port))
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(app, "create_app", lambda: DummyApp())

    try:
        main.start_api()
    finally:
        blocker.close()

    assert captured["port"] == start_port + 1
