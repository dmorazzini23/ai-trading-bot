from __future__ import annotations

import json
from pathlib import Path

from ai_trading.governance.promotion import ModelPromotion
from ai_trading.model_registry import ModelRegistry


def _register_test_model(registry: ModelRegistry, *, strategy: str, marker: str) -> str:
    return registry.register_model(
        model={"marker": marker},
        strategy=strategy,
        model_type="dict",
        metadata={"marker": marker},
    )


def test_promote_marks_previous_production_as_challenger(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "momentum"

    model_a = _register_test_model(registry, strategy=strategy, marker="a")
    model_b = _register_test_model(registry, strategy=strategy, marker="b")
    registry.update_governance_status(model_a, "production")

    promoted = promotion.promote_to_production(model_b, force=True)

    assert promoted is True
    production = registry.get_production_model(strategy)
    assert production is not None
    prod_id, prod_meta = production
    assert prod_id == model_b
    assert prod_meta.get("governance", {}).get("previous_production_model_id") == model_a
    assert registry.model_index[model_a]["governance"]["status"] == "challenger"


def test_rollback_to_previous_production(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "mean_reversion"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True

    rolled_back = promotion.rollback_to_previous_production(
        strategy=strategy,
        reason="live_degradation",
    )

    assert rolled_back is True
    production = registry.get_production_model(strategy)
    assert production is not None
    prod_id, _prod_meta = production
    assert prod_id == champion
    assert registry.model_index[challenger]["governance"]["status"] == "challenger"
    assert (
        registry.model_index[challenger]["governance"]["rolled_back_to_model_id"]
        == champion
    )


def test_record_challenger_evaluation_writes_jsonl(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))

    eval_path = promotion.record_challenger_evaluation(
        strategy="swing",
        champion_model_id="champ-1",
        challenger_model_id="challenger-2",
        metrics={"is_bps": 4.2, "reject_rate": 0.01},
    )

    assert eval_path is not None
    lines = (tmp_path / "governance" / "challenger_evaluations.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["strategy"] == "swing"
    assert payload["champion_model_id"] == "champ-1"
    assert payload["challenger_model_id"] == "challenger-2"
