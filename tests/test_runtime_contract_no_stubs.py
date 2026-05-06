from __future__ import annotations

import pytest

from ai_trading.core.runtime_contract import require_no_stubs


class _MarkedStub:
    _IS_STUB = True


def test_require_no_stubs_blocks_paper_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_TESTING", "0")
    monkeypatch.setenv("PYTEST_RUNNING", "0")
    monkeypatch.setenv("TESTING", "0")

    with pytest.raises(RuntimeError, match="stub dependency active"):
        require_no_stubs(
            {"_REQUESTS_STUB": True, "_HTTP_SESSION_STUB": False},
            execution_mode="paper",
        )


def test_require_no_stubs_allows_testing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_TESTING", "1")
    require_no_stubs({"_REQUESTS_STUB": True}, execution_mode="paper")


def test_dependency_stub_contract_is_testing_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    require_no_stubs({"alpaca_client": _MarkedStub()}, execution_mode="paper")

    monkeypatch.setenv("AI_TRADING_TESTING", "0")
    monkeypatch.setenv("PYTEST_RUNNING", "0")
    monkeypatch.setenv("TESTING", "0")
    with pytest.raises(RuntimeError, match="alpaca_client"):
        require_no_stubs({"alpaca_client": _MarkedStub()}, execution_mode="paper")
