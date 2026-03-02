from __future__ import annotations

import json
from types import SimpleNamespace

import ai_trading.__main__ as cli
import ai_trading.config.management as management
import ai_trading.config.runtime as runtime_cfg


def test_parser_accepts_print_config_flag() -> None:
    parser = cli._build_parser("test")
    args = parser.parse_args(["--print-config"])
    assert args.print_config is True


def test_print_resolved_config_outputs_sanitized_canonical_payload(
    monkeypatch,
    capsys,
) -> None:
    dummy_cfg = SimpleNamespace(alpha="value", secret="super-secret")
    dummy_specs = (
        SimpleNamespace(env=("ALPHA",), field="alpha", mask=False),
        SimpleNamespace(env=("SECRET_TOKEN",), field="secret", mask=True),
    )

    monkeypatch.setattr(management, "validate_no_deprecated_env", lambda env=None: None)
    monkeypatch.setattr(management, "get_trading_config", lambda: dummy_cfg)
    monkeypatch.setattr(management, "canonical_env_map", lambda: {"ALPHA": ("ALPHA_OLD",)})
    monkeypatch.setattr(runtime_cfg, "CONFIG_SPECS", dummy_specs, raising=False)

    rc = cli._print_resolved_config()

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ALPHA"] == "value"
    assert payload["SECRET_TOKEN"] == "***"
    assert payload["ENV_CANONICALIZATION_MAP"]["ALPHA"] == ["ALPHA_OLD"]


def test_print_resolved_config_fails_fast_on_deprecated_env(monkeypatch, capsys) -> None:
    def _raise(_env=None) -> None:
        raise RuntimeError("Deprecated environment keys are not supported.")

    monkeypatch.setattr(management, "validate_no_deprecated_env", _raise)

    rc = cli._print_resolved_config()

    assert rc == 2
    assert "Deprecated environment keys are not supported." in capsys.readouterr().out
