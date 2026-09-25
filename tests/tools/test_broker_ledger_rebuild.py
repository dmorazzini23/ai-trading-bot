from copy import deepcopy
import json

import pytest

from ai_trading.tools.broker_accounting_evidence import rebuild_quantity_ledger


@pytest.fixture
def evidence():
    opening = dict(account_id="paper", trading_mode="paper", positions_complete=True,
                   timestamp="2026-09-18T13:29:00Z", positions={"AAPL": "2"})
    closing = {**opening, "timestamp": "2026-09-18T20:01:00Z", "positions": {"AAPL": "1"}}
    fill = dict(id="fill", activity_type="FILL", order_id="order", symbol="AAPL",
                side="sell", qty="1", transaction_time="2026-09-18T15:00:00Z")
    snapshot = dict(account_id="paper", trading_mode="paper", pagination_complete=True,
                    after="2026-09-18T04:00:00Z", fetched_at="2026-09-18T21:00:00Z", activities=[fill])
    return snapshot, opening, closing


def test_verified_anchor_and_deduplicated_execution_match_without_rewriting(evidence):
    snapshot, opening, closing = evidence
    snapshot["activities"] *= 2
    before = deepcopy(evidence)
    report = rebuild_quantity_ledger(*evidence)
    assert report["status"] == "matched"
    assert len(report["executions"]) == 1
    assert not report["promotion_authority"]
    assert evidence == before
    closing["positions"] = {}
    assert rebuild_quantity_ledger(*evidence)["position_reconciliation"]["differences"] == {"AAPL": "1"}


@pytest.mark.parametrize("target,key,value", [(0,"pagination_complete",False),(0,"after","2026-09-18T14:00Z"),
    (0,"fetched_at","2026-09-18T18:00Z"),(1,"positions_complete",False),(2,"account_id","other"),
    (1,"timestamp","invalid"),(2,"trading_mode","live")])
def test_unverified_boundary_or_acquisition_is_rejected(evidence, target, key, value):
    evidence[target][key] = value
    with pytest.raises(ValueError):
        rebuild_quantity_ledger(*evidence)


@pytest.mark.parametrize("key,value", [("id",""),("qty","nan"),("qty","-1"),("side","unknown"),
    ("transaction_time","invalid"),("order_id","")])
def test_invalid_fill_is_not_silently_discarded(evidence, key, value):
    evidence[0]["activities"][0][key] = value
    with pytest.raises(ValueError):
        rebuild_quantity_ledger(*evidence)


def test_conflicting_execution_is_rejected(evidence):
    evidence[0]["activities"].append({**evidence[0]["activities"][0], "qty":"2"})
    with pytest.raises(ValueError):
        rebuild_quantity_ledger(*evidence)


def test_cli_writes_separate_ledger_and_preserves_sources(evidence, tmp_path, monkeypatch):
    from ai_trading.tools import broker_accounting_evidence as tool
    snapshot, opening, closing = evidence
    paths = {name: tmp_path / (name + ".json") for name in ["snapshot", "opening", "closing", "fills", "output", "ledger", "bundle"]}
    for name, payload in [("snapshot",snapshot),("opening",opening),("closing",closing),("fills",[])]:
        paths[name].write_text(json.dumps(payload))
    original = paths["snapshot"].read_bytes()
    monkeypatch.setattr("sys.argv", ["accounting", "--snapshot", str(paths["snapshot"]), "--fills", str(paths["fills"]),
        "--output", str(paths["output"]), "--opening-positions", str(paths["opening"]),
        "--closing-positions", str(paths["closing"]), "--ledger-output", str(paths["ledger"]),
        "--position-evidence-output", str(paths["bundle"])])
    tool.main()
    assert json.loads(paths["ledger"].read_text())["status"] == "matched"
    assert paths["snapshot"].read_bytes() == original
    assert json.loads(paths["bundle"].read_text()) == {
        "snapshot": snapshot, "opening": opening, "closing": closing,
    }
    assert json.loads(paths["output"].read_text())["orders_sent"] == 0


def test_boundary_interval_excludes_opening_and_includes_closing(evidence):
    snapshot, opening, closing = evidence
    first = snapshot["activities"][0]
    snapshot["activities"] = [{**first,"id":"opening","transaction_time":opening["timestamp"]},
                              {**first,"id":"closing","transaction_time":closing["timestamp"]}]
    report = rebuild_quantity_ledger(*evidence)
    assert [r["fill_id"] for r in report["executions"]] == ["closing"]
    assert report["status"] == "matched"


def test_position_changing_non_fill_activity_blocks_quantity_audit(evidence):
    snapshot, opening, _closing = evidence
    snapshot["activities"].append({
        "id": "split-1",
        "activity_type": "SPLIT",
        "transaction_time": "2026-09-18T16:00:00Z",
    })
    with pytest.raises(ValueError, match="position-changing non-fill"):
        rebuild_quantity_ledger(*evidence)
