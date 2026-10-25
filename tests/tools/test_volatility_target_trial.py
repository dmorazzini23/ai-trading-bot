from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import volatility_target_trial as trial
from ai_trading.tools.research_feasibility import daily_development_dates

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/volatility_target_core_trial_campaign.json"
CAMPAIGN = "vol_target_trial_test"


def contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.update(changes)
    return payload


def small_contract(**changes):
    return contract(campaign_id=CAMPAIGN, development_start="2024-01-02", development_end="2024-12-31",
                    acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
                    support_thresholds={"minimum_periods": 20, "minimum_supported_years": 1, "minimum_periods_per_year": 20},
                    bootstrap={"unit": "circular_blocks_of_periods", "block_periods": 2, "replicates": 100, "seed": 7}) | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


def test_repository_contract_is_registered_with_the_approved_rules():
    registered = json.loads(REPO_CONTRACT.read_text())
    assert registered["status"] == "registered"
    assert "RESEARCH_RESET_AMENDMENT_20261025_VOLATILITY_TARGET.md" in registered["authorization"]
    reset = json.loads((REPO_CONTRACT.parents[1] / trial.RESET_CONFIG).read_text())
    assert registered["campaign_id"] in reset["registered_trials"]
    assert any(a["campaign_id"] == registered["campaign_id"] for a in reset["amendments"])
    full = trial.contract_spec(registered)
    assert (full.risk, full.cash, full.lookback, full.target, full.max_weight) == ("SPY", "BIL", 21, 0.15, 1.0)
    assert (full.every, full.first_decision, full.band) == (5, 21, 0.10)
    assert (full.min_periods, full.min_years, full.min_per_year) == (400, 8, 40)
    assert (full.drawdown_ratio, full.max_turnover) == (0.85, 4.0)
    assert [(name, bps) for name, bps, _ in full.tiers] == [("primary", 10.0), ("stress_high", 20.0), ("stress_low", 6.0)]
    assert full.tier("decision")[1] == 10.0 and full.tier(trial.STRESS_ROLE)[1] == 20.0


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "split"}, "adjustment=all"),
        ({"promotion_authority": True}, "authority"),
        ({"max_risk_weight": 1.5}, "max_risk_weight"),
        ({"rebalance_grid": {"every_sessions": 5, "first_decision_index": 10, "no_trade_band": 0.1}}, "lookback"),
        ({"acceptance": {**contract()["acceptance"], "positive_sharpe_difference_at_stress_high": False}}, "disabled"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))


def test_schedule_never_overlaps_and_respects_lookback_and_missing_sessions():
    s = spec()
    periods = trial.schedule(s, 60)
    assert periods[0] == (21, 22, 27) and periods[1] == (26, 27, 32)
    assert all(nxt[1] >= cur[2] for cur, nxt in zip(periods, periods[1:]))
    assert all(x <= 59 for _, _, x in periods)
    complete = np.ones(80, dtype=bool)
    complete[3] = False  # inside the first volatility window
    complete[32] = False  # exit of the second period, then inside every window through decision 53
    usable = trial.usable_periods(s, complete, 80)
    assert (21, 22, 27) not in usable and (26, 27, 32) not in usable and (51, 52, 57) not in usable
    assert usable[0] == (56, 57, 62)


def _panel(n, *, daily_vol, start="2024-01-02"):
    dates = daily_development_dates(small_contract())[:n]
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))
    index = [d.isoformat() for d in dates]
    risk = pd.DataFrame({"open": closes * 0.999, "close": closes, "complete": True}, index=index)
    cash = pd.DataFrame({"open": 100 * 1.0001 ** np.arange(n), "close": 100 * 1.0001 ** np.arange(n), "complete": True}, index=index)
    return {"SPY": risk, "BIL": cash}, dates


def test_weights_are_point_in_time_and_follow_the_band():
    s = spec()
    panel, dates = _panel(80, daily_vol=0.02)
    periods = trial.usable_periods(s, np.ones(80, dtype=bool), 80)
    frame = trial.build_periods(panel, dates, s, periods)
    # Changing prices after a decision cannot change that decision's volatility or weight.
    later = {k: v.copy() for k, v in panel.items()}
    later["SPY"].iloc[periods[0][0] + 1:, later["SPY"].columns.get_loc("close")] *= 3
    changed = trial.build_periods(later, dates, s, periods)
    assert changed.realized_vol.iloc[0] == frame.realized_vol.iloc[0]
    assert changed.risk_weight.iloc[0] == frame.risk_weight.iloc[0]
    assert frame.traded_weight.iloc[0] == 1.0
    for prev, row in zip(frame.itertuples(), frame.iloc[1:].itertuples()):
        expected = min(1.0, 0.15 / row.realized_vol)
        if abs(expected - prev.risk_weight) >= 0.10:
            assert row.risk_weight == pytest.approx(expected) and row.traded_weight == pytest.approx(2 * abs(expected - prev.risk_weight))
        else:
            assert row.risk_weight == prev.risk_weight and row.traded_weight == 0.0
    assert frame.risk_weight.between(0, 1).all()
    trial.validate_periods(frame, small_contract(), s)


def test_cost_accounting_charges_traded_weight_and_one_control_entry():
    frame = pd.DataFrame({"risk_return": [0.01, 0.02, -0.01], "cash_return": [0.001] * 3,
                          "risk_weight": [0.5, 0.8, 0.8], "traded_weight": [1.0, 0.6, 0.0]})
    nets = trial.tier_returns(frame, 10.0)
    half = 10 / 10_000 / 2
    assert nets["strategy"] == pytest.approx([0.5 * 0.01 + 0.5 * 0.001 - half, 0.8 * 0.02 + 0.2 * 0.001 - half * 0.6, 0.8 * -0.01 + 0.2 * 0.001])
    assert nets["buy_and_hold"] == pytest.approx([0.01 - half, 0.02, -0.01])
    average = 0.7
    assert nets["static_mix"][1] == pytest.approx(average * 0.02 + (1 - average) * 0.001)


def test_calm_market_holds_fully_and_cannot_pass():
    s = spec()
    panel, dates = _panel(250, daily_vol=0.002)  # about 3% annual volatility: weight capped at 1
    frame = trial.build_periods(panel, dates, s, trial.usable_periods(s, np.ones(250, dtype=bool), 250))
    assert (frame.risk_weight == 1.0).all() and frame.traded_weight.iloc[1:].sum() == 0
    report = trial.evaluate(frame, s)
    assert report["criteria"]["adequate_support"] is True
    assert report["criteria"]["positive_sharpe_difference_lower_bound_at_primary"] is False
    assert report["decision"] == "retired"
    assert report["annual_turnover"] == 0.0
    json.dumps(report, allow_nan=False)


def test_max_drawdown_includes_the_starting_value():
    assert trial.max_drawdown(np.array([-0.1, 0.05, -0.2])) == pytest.approx(1 - 0.9 * 1.05 * 0.8)
    assert trial.max_drawdown(np.array([0.1, 0.1])) == 0.0


# --- end-to-end with a fake data client -----------------------------------------------------------

class _FakeClient:
    def __init__(self, dates, *, extra=None):
        self.dates, self.extra, self.requests = dates, extra or [], []

    def get_stock_bars(self, request):
        self.requests.append(request)
        symbol = request.symbol_or_symbols
        rng = np.random.default_rng(len(self.requests))
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.012 if symbol == "SPY" else 0.0001, len(self.dates))))
        bars = [SimpleNamespace(timestamp=datetime(d.year, d.month, d.day, 5, tzinfo=UTC), open=c * 0.999,
                                high=c * 1.01, low=c * 0.99, close=c, volume=1000.0) for d, c in zip(self.dates, closes)]
        bars += [SimpleNamespace(timestamp=t, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0) for t in self.extra]
        return SimpleNamespace(data={symbol: bars})


@pytest.fixture
def registered(tmp_path):
    payload = small_contract()
    (tmp_path / "config").mkdir()
    (tmp_path / "config/trial.json").write_text(json.dumps(payload))
    (tmp_path / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = tmp_path / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    client = _FakeClient(daily_development_dates(payload))
    return tmp_path, state_path, client


def _run(root, mode, client):
    return trial.run(root, mode=mode, contract_path=Path("config/trial.json"), client_factory=lambda: client)


def test_acquire_preflight_never_claim_and_fit_claims_exactly_once(registered, monkeypatch):
    root, state_path, client = registered
    acquired = _run(root, "acquire", client)
    assert acquired["status"] == "acquisition_written" and acquired["trial_claimed"] is False
    assert [str(r.feed) for r in client.requests] == ["DataFeed.SIP", "DataFeed.SIP"]
    assert [str(r.adjustment) for r in client.requests] == ["Adjustment.ALL", "Adjustment.ALL"]
    manifest = json.loads((state_path.parent / "data/acquisition.json").read_text())
    assert set(manifest["sources"]) == {"SPY", "BIL"} and manifest["sources"]["SPY"]["rows"] == 252
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", client)

    preflight = _run(root, "preflight", client)
    assert preflight["status"] == "preflight_passed" and preflight["trial_claimed"] is False
    assert json.loads(state_path.read_text())["trials"] == []
    audit = json.loads((state_path.parent / "support_audit.json").read_text())
    assert audit["prices_read_for_analysis"] is False and audit["possible_periods"] == preflight["possible_periods"]
    assert not (state_path.parent / "report.json").exists()

    report = _run(root, "fit", client)
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning"}
    assert report["holdout_evaluated"] is False and report["orders_sent"] == 0
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    monkeypatch.setattr(trial, "load_sessions", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for mode in ("fit", "preflight", "acquire"):
        assert _run(root, mode, client)["status"] == "budget_already_claimed"


def test_acquire_refuses_2026_rows_before_writing_anything(registered):
    root, state_path, client = registered
    client.extra = [datetime(2026, 9, 10, 4, tzinfo=UTC)]
    with pytest.raises(ValueError, match="outside development"):
        _run(root, "acquire", client)
    assert not (state_path.parent / "data").exists()


def test_preflight_refuses_a_changed_source_and_crash_after_claim_consumes_budget(registered, monkeypatch):
    root, state_path, client = registered
    _run(root, "acquire", client)
    source = state_path.parent / "data/SPY.csv"
    original = source.read_text()
    source.write_text(original.replace("\n", "\n", 1) + "\n")
    with pytest.raises(ValueError, match="source changed"):
        _run(root, "preflight", client)
    source.write_text(original)

    def crash(*args, **kwargs):
        raise RuntimeError("evaluation crashed")

    monkeypatch.setattr(trial, "build_periods", crash)
    with pytest.raises(RuntimeError):
        _run(root, "fit", client)
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    assert _run(root, "fit", client)["status"] == "budget_already_claimed"


def test_fail_closed_without_registration_or_reset_entry(registered):
    root, state_path, client = registered
    (root / "config/draft.json").write_text(json.dumps(small_contract(status="draft_not_registered")))
    with pytest.raises(ValueError, match="not registered"):
        trial.run(root, mode="acquire", contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": []}))
    with pytest.raises(ValueError, match="registered_trials"):
        trial.run(root, mode="acquire", contract_path=Path("config/trial.json"), client_factory=lambda: pytest.fail("fetched"))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    (root / "config/trial.json").write_text(json.dumps(small_contract(hypothesis="changed after registration")))
    with pytest.raises(ValueError, match="immutable"):
        trial.run(root, mode="acquire", contract_path=Path("config/trial.json"), client_factory=lambda: pytest.fail("fetched"))
    (root / "config/trial.json").write_text(json.dumps(small_contract()))
    state_path.unlink()
    with pytest.raises(ValueError, match="registration is required"):
        trial.run(root, mode="acquire", contract_path=Path("config/trial.json"), client_factory=lambda: pytest.fail("fetched"))
