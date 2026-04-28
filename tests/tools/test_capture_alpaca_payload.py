from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "capture_alpaca_payload.py"
_SPEC = importlib.util.spec_from_file_location("capture_alpaca_payload", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
capture_alpaca_payload = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = capture_alpaca_payload
_SPEC.loader.exec_module(capture_alpaca_payload)


def test_redacted_headers_omit_alpaca_credentials() -> None:
    headers = {
        "APCA-API-KEY-ID": "live-key",
        "APCA-API-SECRET-KEY": "live-secret",
        "Authorization": "Bearer live-token",
        "Accept": "application/json",
    }

    redacted = capture_alpaca_payload._redacted_headers(headers)

    assert redacted["APCA-API-KEY-ID"] == "<redacted>"
    assert redacted["APCA-API-SECRET-KEY"] == "<redacted>"
    assert redacted["Authorization"] == "<redacted>"
    assert redacted["Accept"] == "application/json"
    assert "live-secret" not in json.dumps(redacted)


def test_write_private_json_uses_owner_only_mode(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"

    capture_alpaca_payload._write_private_json(output, {"ok": True})

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
