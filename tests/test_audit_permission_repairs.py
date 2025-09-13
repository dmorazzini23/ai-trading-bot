from ai_trading import audit


def test_log_trade_calls_fix_permissions_on_parent_dir_failure(monkeypatch, tmp_path):
    path = tmp_path / "trades.csv"
    monkeypatch.setenv("TRADE_LOG_FILE", str(path))
    audit.TRADE_LOG_FILE = str(path)

    called = {}

    def raise_perm(_p):
        raise PermissionError("denied")

    def fix(_p):
        called["hit"] = True
        return True

    monkeypatch.setattr(audit, "_ensure_parent_dir", raise_perm)
    monkeypatch.setattr(audit, "fix_file_permissions", fix)
    # ensure file header isn't accidentally touched
    monkeypatch.setattr(audit, "_ensure_file_header", lambda p, h: None)

    audit.log_trade("SYM", 1, "BUY", 1.0)

    assert called


def test_log_trade_calls_fix_permissions_on_file_header_failure(monkeypatch, tmp_path):
    path = tmp_path / "trades.csv"
    monkeypatch.setenv("TRADE_LOG_FILE", str(path))
    audit.TRADE_LOG_FILE = str(path)

    monkeypatch.setattr(audit, "_ensure_parent_dir", lambda p: None)

    called = {"fix": 0}

    def fix(_p):
        called["fix"] += 1
        return True

    attempts = {"count": 0}

    def raise_then_pass(p, headers):
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("denied")
        attempts["count"] += 1
        return None

    monkeypatch.setattr(audit, "fix_file_permissions", fix)
    monkeypatch.setattr(audit, "_ensure_file_header", raise_then_pass)

    audit.log_trade("SYM", 1, "BUY", 1.0)

    assert called["fix"] == 1
    assert attempts["count"] == 2
