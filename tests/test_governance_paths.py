from __future__ import annotations

import ai_trading.governance.paths as governance_paths
from ai_trading.governance.paths import resolve_governance_base_path
from ai_trading.governance.promotion import ModelPromotion


def test_resolve_governance_base_path_defaults_to_output_dir(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        governance_paths,
        "get_env",
        lambda key, default="", **_: (
            str(tmp_path / "output")
            if key == "AI_TRADING_OUTPUT_DIR"
            else ""
        ),
    )

    resolved = resolve_governance_base_path()

    assert resolved == (tmp_path / "output" / "governance").resolve()


def test_resolve_governance_base_path_honors_relative_override(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        governance_paths,
        "get_env",
        lambda key, default="", **_: (
            "custom/governance"
            if key == "AI_TRADING_GOVERNANCE_BASE_PATH"
            else default
        ),
    )

    resolved = resolve_governance_base_path()

    assert resolved == (tmp_path / "custom" / "governance").resolve()


def test_model_promotion_default_base_path_uses_runtime_output_dir(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        governance_paths,
        "get_env",
        lambda key, default="", **_: (
            str(tmp_path / "output")
            if key == "AI_TRADING_OUTPUT_DIR"
            else ""
        ),
    )

    promotion = ModelPromotion(base_path=None)

    assert promotion.base_path == (tmp_path / "output" / "governance").resolve()
    assert promotion.active_dir.exists()
