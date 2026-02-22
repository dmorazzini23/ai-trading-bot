from ai_trading.core import bot_engine
import pytest
import logging
from pathlib import Path


class _LoggerOnceStub:
    """Mimic ``logger_once`` behaviour by deduplicating on key."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._emitted: set[str] = set()

    def warning(self, message: str, *args, **kwargs) -> None:  # noqa: D401, ANN001
        key = kwargs.get("key") or message
        if key in self._emitted:
            return
        self._emitted.add(key)
        extra = kwargs.get("extra") or {}
        self.calls.append({
            "message": message,
            "key": key,
            "extra": dict(extra),
        })


def _as_classmethod(path: Path):
    """Return a classmethod returning ``path`` for monkeypatching ``Path`` hooks."""

    return classmethod(lambda cls: path)


def test_parse_local_positions_creates_trade_log(tmp_path, monkeypatch):
    """Smoke test: reading positions initializes the trade log file."""

    log_path = tmp_path / "trades.jsonl"
    # Point the bot engine to our temporary log file and reset singleton
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    # Ensure file does not yet exist
    assert not log_path.exists()

    # Parsing positions should trigger trade log initialization
    positions = bot_engine._parse_local_positions()

    assert positions == {}
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_trade_logger_records_entry(tmp_path, monkeypatch):
    """Trade logger writes header then appends entries on first use."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("AAPL", 100.0, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert len(lines) == 2
    assert "AAPL" in lines[1]


def test_trade_logger_header_without_portalocker_lock(tmp_path, monkeypatch):
    """Trade logger should still write when portalocker lacks lock/unlock."""

    class _PortalockerStub:
        LOCK_EX = object()

    stub = _PortalockerStub()

    monkeypatch.setattr(bot_engine, "portalocker", stub)
    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("STUB", 1.23, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert any("STUB" in line for line in lines[1:])


def test_get_trade_logger_creates_header_when_missing(tmp_path, monkeypatch):
    """get_trade_logger should create the file with a header if absent."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    bot_engine.get_trade_logger()

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_read_trade_log_initializes_file_with_header(tmp_path, monkeypatch):
    """_read_trade_log initializes missing file and writes header."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    df = bot_engine._read_trade_log(str(log_path))

    assert df is None
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")


def test_existing_empty_log_gets_header_and_entry(tmp_path, monkeypatch):
    """Existing empty log gets header and first entry on startup."""

    log_path = tmp_path / "trades.jsonl"
    log_path.touch()
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger = bot_engine.get_trade_logger()
    logger.log_entry("MSFT", 123.0, 1, "buy", "test")

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_time")
    assert len(lines) == 2
    assert "MSFT" in lines[1]


def test_get_trade_logger_creates_missing_directory(tmp_path, monkeypatch):
    """get_trade_logger creates the parent directory when absent."""

    log_dir = tmp_path / "nested"
    log_path = log_dir / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    bot_engine.get_trade_logger()

    assert log_dir.exists()
    assert log_path.exists()


def test_get_trade_logger_falls_back_when_dir_not_writable(tmp_path, monkeypatch, caplog):
    """get_trade_logger falls back to a user state dir when the target is read-only."""

    state_home = tmp_path / "state-home"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    log_dir = tmp_path / "readonly"
    log_dir.mkdir()
    log_dir.chmod(0o555)
    log_path = log_dir / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    fallback_path = Path(bot_engine._compute_user_state_trade_log_path(log_path.name))
    assert logger_instance.path == str(fallback_path)
    assert bot_engine._TRADE_LOGGER_SINGLETON.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(fallback_path)
    assert fallback_path.exists()
    messages = [record.getMessage() for record in caplog.records]
    assert "TRADE_LOG_FALLBACK_USER_STATE" in messages
    assert "TRADE_LOGGER_FALLBACK_ACTIVE" in messages

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bot_engine.get_trade_logger()

    messages = [record.getMessage() for record in caplog.records]
    assert "TRADE_LOG_FALLBACK_USER_STATE" not in messages
    assert "TRADE_LOGGER_FALLBACK_ACTIVE" not in messages


def test_default_trade_log_path_prefers_local_logs_over_state(tmp_path, monkeypatch):
    """default_trade_log_path() should prefer ./logs before state fallbacks."""

    workspace = tmp_path / "workspace"
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True)

    state_home = tmp_path / "state-home"
    state_home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    home_dir = tmp_path / "home-dir"
    home_dir.mkdir()
    monkeypatch.setattr(bot_engine.Path, "home", _as_classmethod(home_dir))

    tmp_root = tmp_path / "tmp-root"
    tmp_root.mkdir()
    monkeypatch.setattr(bot_engine.tempfile, "gettempdir", lambda: str(tmp_root))

    monkeypatch.setattr(bot_engine.Path, "cwd", _as_classmethod(workspace))

    original_makedirs = bot_engine.os.makedirs

    def fake_makedirs(path: str, *args, **kwargs):  # noqa: D401, ANN001
        if bot_engine.os.path.abspath(path) == "/var/log/ai-trading-bot":
            raise PermissionError("mocked permission error")
        return original_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(bot_engine.os, "makedirs", fake_makedirs)

    original_access = bot_engine.os.access

    def fake_access(path: str, mode: int) -> bool:  # noqa: D401, ANN001
        if bot_engine.os.path.abspath(path) == "/var/log/ai-trading-bot":
            return False
        return original_access(path, mode)

    monkeypatch.setattr(bot_engine.os, "access", fake_access)

    resolved_path = bot_engine.default_trade_log_path()

    assert Path(resolved_path) == logs_dir / "trades.jsonl"


def test_get_trade_logger_keeps_child_dir_when_parent_readonly(tmp_path, monkeypatch, caplog, request):
    """Existing writable child directories should not trigger fallback when parent is read-only."""

    parent_dir = tmp_path / "readonly-parent"
    log_dir = parent_dir / "child"
    log_dir.mkdir(parents=True)
    log_dir.chmod(0o700)
    parent_dir.chmod(0o555)

    def _restore_parent_permissions() -> None:
        if parent_dir.exists():
            parent_dir.chmod(0o755)

    request.addfinalizer(_restore_parent_permissions)
    log_path = log_dir / "trades.jsonl"

    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    assert logger_instance.path == str(log_path)
    assert bot_engine.TRADE_LOG_FILE == str(log_path)
    assert bot_engine._TRADE_LOG_FALLBACK_PATH is None
    assert log_path.exists()
    assert not any(record.getMessage() == "TRADE_LOG_FALLBACK_USER_STATE" for record in caplog.records)


def test_get_trade_logger_falls_back_when_env_override_unwritable(tmp_path, monkeypatch, caplog):
    """Env-configured unwritable paths fall back to ./logs without exiting."""

    workspace = tmp_path / "workspace"
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True)

    monkeypatch.chdir(workspace)

    state_home = tmp_path / "state-home"
    state_home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    env_dir = tmp_path / "env-target"
    env_dir.mkdir()
    env_target = env_dir / "trades.jsonl"

    original_is_dir_writable = bot_engine._is_dir_writable

    def fake_is_dir_writable(path: str) -> bool:  # noqa: D401, ANN001
        resolved = Path(path).resolve(strict=False)
        if resolved == env_dir.resolve():
            return False
        return original_is_dir_writable(path)

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)

    monkeypatch.setenv("AI_TRADING_TRADE_LOG_PATH", str(env_target))
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(env_target))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    expected_path = logs_dir / env_target.name
    assert logger_instance.path == str(expected_path)
    assert bot_engine._TRADE_LOGGER_SINGLETON.path == str(expected_path)
    assert bot_engine.TRADE_LOG_FILE == str(expected_path)
    assert expected_path.exists()
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(expected_path)

    messages = [record.getMessage() for record in caplog.records]
    assert "TRADE_LOG_FALLBACK_USER_STATE" in messages
    assert "TRADE_LOGGER_FALLBACK_ACTIVE" in messages


def test_get_trade_logger_falls_back_on_dir_creation_permission_error(tmp_path, monkeypatch, caplog):
    """get_trade_logger falls back when os.makedirs raises PermissionError."""

    state_home = tmp_path / "state-permission"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o555)
    log_path = parent / "child" / "trades.jsonl"
    log_dir = log_path.parent
    original_makedirs = bot_engine.os.makedirs
    original_path_mkdir = bot_engine.Path.mkdir

    def fake_makedirs(path: str, mode: int = 0o777, exist_ok: bool = False) -> None:
        if bot_engine.os.path.abspath(path) == bot_engine.os.path.abspath(str(log_dir)):
            raise PermissionError("mocked permission error")
        return original_makedirs(path, mode=mode, exist_ok=exist_ok)

    monkeypatch.setattr(bot_engine.os, "makedirs", fake_makedirs)

    def fake_path_mkdir(self, *args, **kwargs):  # noqa: D401, ANN001
        if Path(self) == log_dir:
            raise PermissionError("mocked permission error")
        return original_path_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(bot_engine.Path, "mkdir", fake_path_mkdir)

    class _StubTradeLogger:
        def __init__(self, path: str | Path | None = None, *args, **kwargs) -> None:  # noqa: D401, ARG002
            self.path = bot_engine.abspath_safe(path)

        def log_entry(self, *args, **kwargs) -> None:  # noqa: D401, ARG002
            return None

    monkeypatch.setattr(bot_engine, "TradeLogger", _StubTradeLogger)
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    with caplog.at_level(logging.WARNING):
        logger_instance = bot_engine.get_trade_logger()

    fallback_path = Path(bot_engine._compute_user_state_trade_log_path(log_path.name))
    assert logger_instance.path == str(fallback_path)
    assert bot_engine._TRADE_LOGGER_SINGLETON.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert fallback_path.exists()
    messages = [record.getMessage() for record in caplog.records]
    assert "TRADE_LOG_FALLBACK_USER_STATE" in messages
    assert "TRADE_LOGGER_FALLBACK_ACTIVE" in messages


def test_trade_log_fallback_uses_tempdir_when_everything_blocked(tmp_path, monkeypatch):
    """Trade log path resolves into a tempdir when no candidate directory is writable."""

    # Unwritable XDG state home so the primary fallback fails.
    state_home = tmp_path / "state-home"
    state_home.mkdir()
    state_home.chmod(0o555)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    # Unwritable home directory prevents ~/.local/state usage.
    home_dir = tmp_path / "home-readonly"
    home_dir.mkdir()
    home_dir.chmod(0o555)
    monkeypatch.setattr(bot_engine.Path, "home", _as_classmethod(home_dir))

    # Project directory with non-writable logs/ to block cwd-based fallback.
    project_root = tmp_path / "project"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True)
    logs_dir.chmod(0o555)
    monkeypatch.setattr(bot_engine.Path, "cwd", _as_classmethod(project_root))

    # Deterministic temp dir for this test.
    temp_parent = tmp_path / "tempdir"
    temp_parent.mkdir()
    monkeypatch.setattr(bot_engine.tempfile, "gettempdir", lambda: str(temp_parent))

    blocked_roots = [state_home.resolve(), home_dir.resolve(), logs_dir.resolve()]

    original_is_dir_writable = bot_engine._is_dir_writable

    def fake_is_dir_writable(path: str) -> bool:
        resolved = Path(path).resolve(strict=False)
        for root in blocked_roots:
            try:
                if resolved.is_relative_to(root):
                    return False
            except ValueError:
                continue
        return original_is_dir_writable(path)

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)

    log_name = "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(logs_dir / log_name))
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger_instance = bot_engine.get_trade_logger()

    expected_dir = temp_parent / "ai-trading-bot"
    expected_path = expected_dir / log_name
    assert logger_instance.path == str(expected_path)
    assert bot_engine._TRADE_LOGGER_SINGLETON.path == str(expected_path)
    assert bot_engine.TRADE_LOG_FILE == str(expected_path)
    assert expected_path.exists()
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(expected_path)


def test_trade_log_fallback_prefers_state_home_when_no_writable_ancestor(tmp_path, monkeypatch):
    """Fallback path resolution should defer to XDG state home when ancestors are blocked."""

    state_home = tmp_path / "state-home"
    state_home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    blocked_root = tmp_path / "blocked"
    blocked_root.mkdir()
    log_path = blocked_root / "child" / "trades.jsonl"
    log_dir = log_path.parent

    original_path_mkdir = bot_engine.Path.mkdir

    def fake_mkdir(self, *args, **kwargs):  # noqa: D401, ANN001
        if Path(self) == log_dir:
            raise PermissionError("mocked permission error")
        return original_path_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(bot_engine.Path, "mkdir", fake_mkdir)

    def fake_is_dir_writable(path: str) -> bool:
        resolved = Path(path).resolve(strict=False)
        try:
            if resolved == state_home or resolved.is_relative_to(state_home):
                return True
        except AttributeError:
            pass
        return False

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)

    class _StubTradeLogger:
        def __init__(self, path: str | Path | None = None, *args, **kwargs) -> None:  # noqa: D401, ARG002
            self.path = bot_engine.abspath_safe(path)

        def log_entry(self, *args, **kwargs) -> None:  # noqa: D401, ARG002
            return None

    logger_once_stub = _LoggerOnceStub()

    monkeypatch.setattr(bot_engine, "TradeLogger", _StubTradeLogger)
    monkeypatch.setattr(bot_engine, "logger_once", logger_once_stub)
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))

    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_FALLBACK_PATH = None

    logger_instance = bot_engine.get_trade_logger()

    fallback_path = Path(bot_engine._compute_user_state_trade_log_path(log_path.name))
    assert logger_instance.path == str(fallback_path)
    assert bot_engine._TRADE_LOGGER_SINGLETON.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert fallback_path.exists()

    # Ensure once-per-process warning semantics remain intact.
    messages = [call["message"] for call in logger_once_stub.calls]
    assert messages == [
        "TRADE_LOG_FALLBACK_USER_STATE",
        "TRADE_LOGGER_FALLBACK_ACTIVE",
    ]

    bot_engine.get_trade_logger()

    assert len(logger_once_stub.calls) == 2


def test_emit_trade_log_fallback_logs_per_destination(monkeypatch):
    """Repeated fallbacks with different targets should emit separate warnings."""

    fallback_paths = iter([
        "/tmp/state/ai-trading-bot/trades.jsonl",
        "/tmp/state/ai-trading-bot/other.jsonl",
    ])

    def fake_compute_user_state_trade_log_path(filename: str = "trades.jsonl") -> str:  # noqa: D401, ARG001
        return next(fallback_paths)

    logger_once_stub = _LoggerOnceStub()

    monkeypatch.setattr(bot_engine, "logger_once", logger_once_stub)
    monkeypatch.setattr(
        bot_engine,
        "_compute_user_state_trade_log_path",
        fake_compute_user_state_trade_log_path,
    )

    first_path = bot_engine._emit_trade_log_fallback(
        preferred_path="/blocked/first/trades.jsonl",
        reason="first",
    )
    second_path = bot_engine._emit_trade_log_fallback(
        preferred_path="/blocked/second/other.jsonl",
        reason="second",
    )

    fallback_messages = [
        call for call in logger_once_stub.calls if call["message"] == "TRADE_LOG_FALLBACK_USER_STATE"
    ]

    assert len(fallback_messages) == 2
    keys = [call["key"] for call in fallback_messages]
    assert keys[0] != keys[1]
    assert first_path in keys[0]
    assert second_path in keys[1]
