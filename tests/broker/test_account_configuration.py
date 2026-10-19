from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.models import AccountConfiguration

import ai_trading.health_payload as health_payload_module
from ai_trading.broker import account_configuration as account_config
from ai_trading.tools import broker_account_configuration as cli

LONG_ONLY_PROFILE = SimpleNamespace(shorts_allowed=False, max_gross_exposure=1.0)

_MATCHING_PAYLOAD: dict[str, Any] = {
    "dtbp_check": "entry",
    "fractional_trading": True,
    "max_margin_multiplier": "1",
    "no_shorting": True,
    "pdt_check": "entry",
    "suspend_trade": False,
    "trade_confirm_email": "all",
    "ptp_no_exception_entry": False,
}
# Shape returned by the paper API on 2026-10-19: no dtbp_check / pdt_check.
_LIVE_PAPER_PAYLOAD: dict[str, Any] = {
    "closing_transactions_only": False,
    "disable_overnight_trading": False,
    "fractional_trading": True,
    "max_margin_multiplier": "4",
    "no_shorting": False,
    "ptp_no_exception_entry": False,
    "suspend_trade": False,
    "trade_confirm_email": "all",
}


class _FakeTradingClient:
    """Mimics alpaca-py: typed parse of the payload plus a raw ``get``."""

    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0
        self.raw_paths: list[str] = []

    def get_account_configurations(self) -> AccountConfiguration:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AccountConfiguration(**dict(self.payload or {}))

    def get(self, path: str) -> dict[str, Any]:
        self.raw_paths.append(path)
        return dict(self.payload or {})


@pytest.fixture(autouse=True)
def _reset_account_configuration_state():
    account_config.reset_account_configuration_state()
    yield
    account_config.reset_account_configuration_state()


def test_expected_configuration_follows_launch_profile() -> None:
    assert account_config.build_expected_configuration(LONG_ONLY_PROFILE) == {
        "no_shorting": True,
        "max_margin_multiplier": "1",
        "dtbp_check": ["entry", "both"],
        "pdt_check": ["entry", "both"],
    }
    shorting = account_config.build_expected_configuration(
        SimpleNamespace(shorts_allowed=True, max_gross_exposure=1.5)
    )
    assert "no_shorting" not in shorting
    assert shorting["max_margin_multiplier"] == "2"


def test_check_reports_match() -> None:
    client = _FakeTradingClient(_MATCHING_PAYLOAD)

    result = account_config.check_account_configuration(client, profile=LONG_ONLY_PROFILE)

    assert result["available"] is True
    assert result["ok"] is True
    assert result["reason"] == "match"
    assert result["mismatches"] == []
    assert result["unreported_fields"] == []
    assert result["source"] == "sdk_model"
    assert result["observed"]["dtbp_check"] == "entry"
    assert result["fetched_at"]
    assert client.raw_paths == []


def test_check_reports_drift() -> None:
    payload = dict(_MATCHING_PAYLOAD, no_shorting=False, max_margin_multiplier="4", pdt_check="exit")
    client = _FakeTradingClient(payload)

    result = account_config.check_account_configuration(client, profile=LONG_ONLY_PROFILE)

    assert result["available"] is True
    assert result["ok"] is False
    assert result["reason"] == "drift"
    assert result["mismatches"] == [
        {"field": "no_shorting", "expected": True, "observed": False},
        {"field": "max_margin_multiplier", "expected": "1", "observed": "4"},
        {"field": "pdt_check", "expected": ["entry", "both"], "observed": "exit"},
    ]


def test_check_falls_back_to_raw_get_when_sdk_model_rejects_response() -> None:
    client = _FakeTradingClient(_LIVE_PAPER_PAYLOAD)

    result = account_config.check_account_configuration(client, profile=LONG_ONLY_PROFILE)

    assert client.raw_paths == ["/account/configurations"]
    assert result["available"] is True
    assert result["source"] == "sdk_raw_get"
    assert result["unreported_fields"] == ["dtbp_check", "pdt_check"]
    assert "dtbp_check" not in result["observed"]
    assert [item["field"] for item in result["mismatches"]] == [
        "no_shorting",
        "max_margin_multiplier",
    ]


@pytest.mark.parametrize(
    ("client", "reason_prefix"),
    [
        (None, "trading_client_unavailable"),
        (object(), "get_account_configurations_unsupported"),
        (_FakeTradingClient(error=ConnectionError("down")), "fetch_failed:ConnectionError"),
        (_FakeTradingClient(error=APIError("forbidden")), "fetch_failed:APIError"),
    ],
)
def test_check_reports_unavailable_without_raising(client: Any, reason_prefix: str) -> None:
    result = account_config.check_account_configuration(client, profile=LONG_ONLY_PROFILE)

    assert result["available"] is False
    assert result["ok"] is False
    assert result["mismatches"] == []
    assert str(result["reason"]).startswith(reason_prefix)


def test_startup_hook_never_raises_when_client_fails(monkeypatch) -> None:
    client = _FakeTradingClient(error=RuntimeError("socket closed"))

    result = account_config.run_startup_account_configuration_check(client)

    assert result is not None and result["available"] is False
    assert account_config.observe_account_configuration()["available"] is False

    def _explode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unexpected")

    account_config.reset_account_configuration_state()
    monkeypatch.setattr(account_config, "check_account_configuration", _explode)
    assert account_config.run_startup_account_configuration_check(client) is None


def test_main_startup_helper_never_raises(monkeypatch) -> None:
    import ai_trading.main as main_module

    def _explode(_client: Any) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(account_config, "run_startup_account_configuration_check", _explode)
    main_module._check_broker_account_configuration(None, _FakeTradingClient())


def test_startup_hook_runs_once_after_success_and_throttles_retries(monkeypatch) -> None:
    monkeypatch.setattr(
        account_config,
        "build_expected_configuration",
        lambda profile=None: {"no_shorting": True, "max_margin_multiplier": "1"},
    )
    failing = _FakeTradingClient(error=ConnectionError("down"))
    assert account_config.run_startup_account_configuration_check(failing) is not None
    assert account_config.run_startup_account_configuration_check(failing) is None
    assert failing.calls == 1

    account_config.reset_account_configuration_state()
    healthy = _FakeTradingClient(_MATCHING_PAYLOAD)
    first = account_config.run_startup_account_configuration_check(healthy)
    assert first is not None and first["ok"] is True
    assert account_config.run_startup_account_configuration_check(healthy) is None
    assert healthy.calls == 1


def test_startup_hook_logs_drift(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        account_config,
        "build_expected_configuration",
        lambda profile=None: {"no_shorting": True, "max_margin_multiplier": "1"},
    )
    caplog.set_level("INFO")
    account_config.run_startup_account_configuration_check(
        _FakeTradingClient(_LIVE_PAPER_PAYLOAD)
    )
    assert any(rec.message == "BROKER_ACCOUNT_CONFIGURATION_DRIFT" for rec in caplog.records)


def test_health_section_renders_states() -> None:
    assert account_config.health_section() == {
        "ok": None,
        "available": False,
        "reason": "not_checked",
        "mismatches": [],
        "unreported_fields": [],
        "fetched_at": None,
    }
    drift = account_config.check_account_configuration(
        _FakeTradingClient(_LIVE_PAPER_PAYLOAD), profile=LONG_ONLY_PROFILE
    )
    section = account_config.health_section(drift)
    assert section["ok"] is False
    assert section["available"] is True
    assert len(section["mismatches"]) == 2
    assert section["unreported_fields"] == ["dtbp_check", "pdt_check"]

    unavailable = account_config.check_account_configuration(None, profile=LONG_ONLY_PROFILE)
    assert account_config.health_section(unavailable)["ok"] is None
    assert account_config.health_section(unavailable)["available"] is False


def test_runtime_health_payload_flags_drift_only_when_available() -> None:
    payload = health_payload_module.build_runtime_health_payload()
    assert payload["broker_account_configuration"]["available"] is False
    assert "broker_account_configuration_drift" not in payload["attention_flags"]

    account_config.record_account_configuration_result(
        account_config.check_account_configuration(None, profile=LONG_ONLY_PROFILE)
    )
    payload = health_payload_module.build_runtime_health_payload()
    assert payload["broker_account_configuration"]["available"] is False
    assert "broker_account_configuration_drift" not in payload["attention_flags"]

    account_config.record_account_configuration_result(
        account_config.check_account_configuration(
            _FakeTradingClient(_LIVE_PAPER_PAYLOAD), profile=LONG_ONLY_PROFILE
        )
    )
    payload = health_payload_module.build_runtime_health_payload()
    section = payload["broker_account_configuration"]
    assert section["available"] is True
    assert section["ok"] is False
    assert [item["field"] for item in section["mismatches"]] == [
        "no_shorting",
        "max_margin_multiplier",
    ]
    assert "broker_account_configuration_drift" in payload["attention_flags"]


def test_runtime_health_payload_degrades_when_section_fails(monkeypatch) -> None:
    def _explode(_result: Any = None) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(account_config, "health_section", _explode)
    payload = health_payload_module.build_runtime_health_payload()
    assert payload["broker_account_configuration"]["available"] is False
    assert "broker_account_configuration_drift" not in payload["attention_flags"]


@pytest.mark.parametrize(
    ("client", "expected_code"),
    [
        (_FakeTradingClient(_MATCHING_PAYLOAD), cli.EXIT_MATCH),
        (_FakeTradingClient(_LIVE_PAPER_PAYLOAD), cli.EXIT_DRIFT),
        (_FakeTradingClient(error=ConnectionError("down")), cli.EXIT_UNAVAILABLE),
    ],
)
def test_cli_exit_codes(monkeypatch, client: Any, expected_code: int) -> None:
    monkeypatch.setattr(
        account_config,
        "build_expected_configuration",
        lambda profile=None: {"no_shorting": True, "max_margin_multiplier": "1"},
    )
    code, payload = cli.run_check(client)
    assert code == expected_code
    assert "observed" in payload and "expected" in payload and "mismatches" in payload


def test_cli_main_prints_json_and_reports_missing_client(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "reload_env", lambda _path=None: None)
    monkeypatch.setattr(
        cli,
        "_build_client",
        lambda: (None, {"paper": True, "client_error": "alpaca_credentials_missing"}),
    )

    code = cli.main(["--check"])

    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_UNAVAILABLE
    assert payload["available"] is False
    assert payload["reason"] == "alpaca_credentials_missing"
    assert payload["paper"] is True


def test_cli_has_no_apply_option() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--check", "--apply"])
