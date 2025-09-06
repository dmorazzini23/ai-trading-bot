import pathlib
import ai_trading.analytics.peak_equity as peak_equity


def test_peak_equity_permission(monkeypatch, tmp_path, caplog):
    peak = tmp_path / "peak.txt"
    peak.touch()
    peak.chmod(0)
    monkeypatch.setattr(peak_equity, "_PEAK_EQUITY_PERMISSION_LOGGED", False)
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError()))
    caplog.set_level("WARNING")

    assert peak_equity.read_peak_equity(peak) == 0.0
    assert "permission denied" in caplog.text.lower()

    caplog.clear()
    assert peak_equity.read_peak_equity(peak) == 0.0
    assert caplog.text == ""
