from __future__ import annotations

import pytest

import ai_trading.model_loader as model_loader
import ai_trading.paths as paths
from ai_trading.training.train_ml import train_model_cli


def test_train_model_cli_full_mode_trains_normalized_symbols(monkeypatch, tmp_path):
    trained: list[tuple[str, object]] = []

    def fake_train_and_save(symbol: str, models_dir):
        trained.append((symbol, models_dir))
        return object()

    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(model_loader, "train_and_save_model", fake_train_and_save)

    train_model_cli([" aapl ", "AAPL", "msft"], model_type="ridge")

    assert [symbol for symbol, _ in trained] == ["AAPL", "MSFT"]
    assert all(models_dir == tmp_path for _, models_dir in trained)


def test_train_model_cli_full_mode_raises_when_all_symbols_fail(monkeypatch, tmp_path):
    def always_fail(_symbol: str, _models_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(model_loader, "train_and_save_model", always_fail)

    with pytest.raises(RuntimeError):
        train_model_cli(["AAPL"], model_type="ridge")


def test_train_model_cli_requires_non_empty_symbol(monkeypatch):
    monkeypatch.setattr(model_loader, "train_and_save_model", lambda *_a, **_k: object())

    with pytest.raises(ValueError):
        train_model_cli(["", "   "], model_type="ridge")
