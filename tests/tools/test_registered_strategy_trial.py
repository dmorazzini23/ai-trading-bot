from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_trading.strategies import CrossSectionalMomentumStrategy
from ai_trading.tools import registered_strategy_trial as trial
from ai_trading.tools.research_feasibility import DailyGrid, daily_development_dates, regular_session_minutes

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/cross_sectional_momentum_trial_campaign.json"
SYMBOLS = ["DIA", "IWM", "QQQ", "SPY", "XLE", "XLF"]
GRID = DailyGrid(63, 5, "open_and_close_minutes", 0.0)


def contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.update(status="registered")
    payload.update(changes)
    return payload


def small_contract(**changes):
    return contract(campaign_id="csm_trial_test", development_end="2024-05-31", acquisition="data/acquisition.json",
                    support_thresholds={"minimum_decision_periods": 5, "minimum_selections": 5,
                                        "minimum_selections_per_fold": 1, "minimum_supported_folds": 1},
                    acceptance=contract()["acceptance"] | {"minimum_profitable_folds": 0},
                    bootstrap={"unit": "circular_blocks_of_decision_periods", "block_periods": 2,
                               "replicates": 100, "seed": 7}) | changes


def sessions(start="2024-01-01", end="2024-06-28"):
    return daily_development_dates(contract(development_start=start, development_end=end))


def daily_panel(dates, *, drifts=None):
    """Symbol i trends at drifts[i] per session; opens equal the prior close plus a fixed gap."""
    drifts = drifts or {s: 0.0005 * (i - 2) for i, s in enumerate(SYMBOLS)}
    steps = np.arange(len(dates))
    index = [d.isoformat() for d in dates]
    return {s: pd.DataFrame({"close": 100 * (1 + drifts[s]) ** steps, "open": 100 * (1 + drifts[s]) ** (steps - .5),
                             "complete": True}, index=index) for s in SYMBOLS}


def test_point_in_time_truncation_future_prices_cannot_change_a_decision():
    dates = sessions()
    panel = daily_panel(dates)
    seen = []

    class Spy(CrossSectionalMomentumStrategy):
        def generate_signals(self, market_data):
            seen.append({s: (len(v), v.index[-1]) for s, v in market_data["prices"].items()})
            return super().generate_signals(market_data)

    periods = trial.build_periods(panel, dates, GRID, Spy(), SYMBOLS)
    assert len(periods) == len(seen) > 2
    for row, prices in zip(periods.itertuples(), seen):
        assert set(prices.values()) == {(row.decision_index + 1, row.decision_session)}
    assert set(periods.selected) == {"XLF"}
    first = int(periods.decision_index.iloc[0])
    shocked = {s: f.copy() for s, f in panel.items()}
    # Collapse the leader after the first decision; later decisions must change, the first must not.
    shocked["XLF"].iloc[first + 1:, shocked["XLF"].columns.get_loc("close")] *= 0.5
    shocked["XLF"].iloc[first + 1:, shocked["XLF"].columns.get_loc("open")] *= 0.5
    again = trial.build_periods(shocked, dates, GRID, CrossSectionalMomentumStrategy(), SYMBOLS)
    assert again.selected.iloc[0] == periods.selected.iloc[0] == "XLF"
    assert (again.selected.iloc[1:] != "XLF").all()
    # A panel misaligned with the session calendar would hand the strategy a later close: refuse it.
    shifted = {s: f.set_axis(list(f.index[1:]) + ["2099-01-01"]) for s, f in panel.items()}
    with pytest.raises(ValueError, match="point-in-time"):
        trial.build_periods(shifted, dates, GRID, CrossSectionalMomentumStrategy(), SYMBOLS)


def test_holds_never_overlap_and_validation_rejects_overlap():
    schedule = GRID.schedule(502)
    assert len(schedule) == 87
    assert all(e - d == 1 and x - e == 5 for d, e, x in schedule)
    assert all(nxt[1] >= prev[2] for prev, nxt in zip(schedule, schedule[1:]))
    dates = sessions()
    periods = trial.build_periods(daily_panel(dates), dates, GRID, CrossSectionalMomentumStrategy(), SYMBOLS)
    payload = contract()
    trial.validate_periods(periods, payload, GRID)
    overlapping = periods.copy()
    overlapping.loc[1, ["entry_index", "exit_index", "decision_index"]] -= 2
    with pytest.raises(ValueError, match="overlapping"):
        trial.validate_periods(overlapping, payload, GRID)
    with pytest.raises(ValueError, match="hold horizon"):
        trial.validate_periods(periods, payload, DailyGrid(63, 4, "open_and_close_minutes", 0.0))


def test_cost_accounting_per_tier_and_reported_tiers_cannot_rescue():
    spec = trial.contract_spec(contract(bootstrap={"unit": "circular_blocks_of_decision_periods",
                                                   "block_periods": 1, "replicates": 100, "seed": 1}))
    periods = pd.DataFrame(dict(selections=[1, 0, 1], selected=["QQQ", "", "SPY"], fold=["2024Q2"] * 3,
                                strategy_gross_bps=[30.0, 0.0, -5.0], control_gross_bps=[10.0, 4.0, 2.0]))
    report = trial.evaluate(periods, spec)
    tiers = report["cost_tiers"]
    assert set(tiers) == {"primary", "stress_high", "stress_low", "measured_paper_taker_median"}
    for name, cost in (("primary", 10), ("stress_high", 20), ("stress_low", 6), ("measured_paper_taker_median", 2.8)):
        strategy = np.array([30 - cost, 0.0, -5 - cost])  # abstention holds cash and pays nothing
        control = np.array([10.0, 4.0, 2.0]) - cost
        assert tiers[name]["net_per_period_bps"] == pytest.approx(strategy.mean())
        assert tiers[name]["net_per_selected_period_bps"] == pytest.approx(strategy[[0, 2]].mean())
        assert tiers[name]["always_long_net_per_period_bps"] == pytest.approx(control.mean())
        assert tiers[name]["paired_vs_always_long_bps"] == pytest.approx((strategy - control).mean())
    assert report["abstentions"] == 1 and report["selections"] == 2
    # Positive at the measured 2.8 bps tier, negative at the 10 bps decision tier: never eligible.
    flat = pd.DataFrame(dict(selections=[1] * 8, selected=["SPY"] * 8, fold=["2024Q2"] * 8,
                             strategy_gross_bps=[5.0] * 8, control_gross_bps=[5.0] * 8))
    outcome = trial.evaluate(flat, spec)
    assert outcome["cost_tiers"]["measured_paper_taker_median"]["net_per_period_bps"] == pytest.approx(2.2)
    assert outcome["cost_tiers"]["primary"]["net_per_period_bps"] == pytest.approx(-5)
    assert outcome["decision"] != "eligible_for_untouched_test_planning"
    assert outcome["criteria"]["positive_cash_lower_bound_at_primary"] is False
    assert "measured_paper_taker_median" in outcome["reported_only_tiers"]


def test_control_uses_the_same_universe_entry_and_exit_opens():
    dates = sessions()
    panel = daily_panel(dates)
    panel["XLE"].loc[dates[110].isoformat(), ["open", "close", "complete"]] = [np.nan, np.nan, False]
    periods = trial.build_periods(panel, dates, GRID, CrossSectionalMomentumStrategy(), SYMBOLS)
    # Session 110 lies in the lookback-through-exit windows of decisions 108 and 113: both drop for all names.
    assert periods.decision_index.tolist() == [63, 68, 73, 78, 83, 88, 93, 98, 103]
    for row in periods.itertuples():
        gross = {s: (panel[s].open.iloc[row.exit_index] / panel[s].open.iloc[row.entry_index] - 1) * 1e4 for s in SYMBOLS}
        assert row.control_gross_bps == pytest.approx(np.mean(list(gross.values())))
        assert row.strategy_gross_bps == pytest.approx(gross[row.selected])
        assert getattr(row, f"gross_bps_{row.selected}") == pytest.approx(gross[row.selected])


def test_contract_refuses_holdout_2026_and_mismatched_strategy_before_data():
    with pytest.raises(ValueError, match="holdout"):
        trial.contract_spec(contract(development_end="2026-09-10"))
    with pytest.raises(ValueError, match="2026"):
        trial.contract_spec(contract(development_end="2026-01-05"))
    with pytest.raises(ValueError, match="defaults differ"):
        trial.contract_spec(contract(strategy_parameters={"lookback": 60, "top_quantile": 0.2, "min_universe": 6}))
    with pytest.raises(ValueError, match="min_universe"):
        trial.contract_spec(contract(symbols=["AAPL", "AMZN", "MSFT"]))
    with pytest.raises(ValueError, match="cannot be disabled"):
        trial.contract_spec(contract(acceptance=contract()["acceptance"] | {"positive_always_long_lower_bound": False}))
    with pytest.raises(ValueError, match="decision tier"):
        trial.contract_spec(contract(primary_round_trip_cost_bps=2.8))
    index = pd.DatetimeIndex([pd.Timestamp("2025-12-31 20:59", tz="UTC"), pd.Timestamp("2026-01-02 14:30", tz="UTC")])
    with pytest.raises(ValueError, match="outside development"):
        trial.check_development_bounds({"SPY": index}, contract())
    trial.check_development_bounds({"SPY": index[:1]}, contract())


def test_repository_draft_contract_is_complete_but_refused_until_registered(tmp_path, monkeypatch):
    draft = json.loads(REPO_CONTRACT.read_text())
    assert draft["status"] == "draft_not_registered" and draft["promotion_authority"] is False
    spec = trial.contract_spec(contract())
    assert spec.primary.round_trip_bps == 10 and spec.stress.round_trip_bps == 20
    measured = next(t for t in draft["cost_tiers"] if t["name"] == "measured_paper_taker_median")
    assert measured["round_trip_bps"] == 2.8 and measured["role"] == "reported_only"
    assert measured["source"] == "docs/PAPER_ROUND_TRIP_COST_EVIDENCE_20261018.md"
    strategy = CrossSectionalMomentumStrategy()
    assert (strategy.lookback, strategy.top_quantile, strategy.min_universe) == (63, 0.2, 6)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft))
    monkeypatch.setattr(trial, "load_governed_development_timestamps", lambda *a: pytest.fail("data read"))
    with pytest.raises(ValueError, match="not registered"):
        trial.run(tmp_path, fit=False, contract_path=Path("config/draft.json"))
    assert not (tmp_path / "artifacts").exists()


def test_daily_bars_completeness_rules_and_price_validity():
    dates = sessions("2024-07-01", "2024-07-05")  # includes the July 3 early close
    parts = regular_session_minutes(dates)
    index = parts[0].append(parts[1:])
    price = 100 + np.arange(len(index)) / 1000
    minutes = pd.DataFrame(dict(open=price, high=price + 1, low=price - 1, close=price + .5, volume=10.0), index=index)
    bars, rejected = trial.daily_bars(minutes, dates, GRID)
    assert bars.complete.all() and not rejected
    july3 = bars.loc["2024-07-03"]
    assert july3.open == minutes.loc[parts[2][0], "open"] and july3.close == minutes.loc[parts[2][-1], "close"]
    assert len(parts[2]) == 210
    middle = minutes.drop(parts[0][100:104])
    assert trial.daily_bars(middle, dates, GRID)[0].complete.all()
    every, rejected = trial.daily_bars(middle, dates, DailyGrid(63, 5))
    assert not every.complete.iloc[0] and rejected == {"incomplete_session": 1}
    assert np.isnan(every.close.iloc[0])
    no_open = trial.daily_bars(minutes.drop(parts[2][:1]), dates, GRID)[0]
    assert no_open.complete.tolist() == [True, True, False, True]
    bad = minutes.copy()
    bad.loc[parts[3][5], "high"] = 1.0
    assert trial.daily_bars(bad, dates, GRID)[1] == {"invalid_prices": 1}


def write_acquisition(root: Path, payload: dict) -> None:
    dates = daily_development_dates(payload)
    parts = regular_session_minutes(dates)
    index = parts[0].append(parts[1:])
    data = root / "data"
    data.mkdir(parents=True)
    rows = []
    for i, symbol in enumerate(payload["symbols"]):
        price = 100 * (1 + 0.000002 * (i - 2)) ** np.arange(len(index))
        frame = pd.DataFrame(dict(symbol=symbol, timestamp=index.astype(str), open=price, high=price * 1.001,
                                  low=price * 0.999, close=price, volume=100.0))
        path = data / f"{symbol}.csv"
        frame.to_csv(path, index=False)
        rows.append(dict(symbol=symbol, csv_path=str(path), content_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(quality_passed=True, symbols=rows, dataset_identity=dict(
        request_interval=dict(start_date=payload["development_start"], end_date=payload["development_end"]),
        feed=payload["feed"], adjustment=payload["adjustment"], symbols=payload["symbols"]))
    (data / "dataset.provenance.json").write_text(json.dumps(manifest))
    (data / "acquisition.json").write_text(json.dumps(dict(manifest_path="dataset.provenance.json", quality_passed=True)))


@pytest.fixture
def registered(tmp_path):
    payload = small_contract()
    write_acquisition(tmp_path, payload)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/trial.json").write_text(json.dumps(payload))
    (tmp_path / "config/research_reset.json").write_text(json.dumps({"registered_trials": [payload["campaign_id"]]}))
    state_path = tmp_path / f"artifacts/{payload['campaign_id']}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return tmp_path, state_path


def test_preflight_never_claims_and_fit_claims_exactly_once(registered, monkeypatch):
    root, state_path = registered
    preflight = trial.run(root, fit=False, contract_path=Path("config/trial.json"))
    assert preflight["status"] == "preflight_passed" and preflight["trial_claimed"] is False
    assert json.loads(state_path.read_text())["trials"] == []
    support = json.loads((state_path.parent / "sampling_support.json").read_text())
    # range(63, 105 - 1 - 5, 5) over January-May 2024 sessions.
    assert support["possible_periods"] == preflight["possible_periods"] == 8
    report = trial.run(root, fit=True, contract_path=Path("config/trial.json"))
    assert report["decision_periods"] == 8 and report["selections_by_symbol"] == {"XLF": 8}
    assert report["holdout_evaluated"] is False and report["promotion_authority"] is False
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    monkeypatch.setattr(trial, "load_governed_development_timestamps", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for fit in (True, False):
        assert trial.run(root, fit=fit, contract_path=Path("config/trial.json"))["status"] == "budget_already_claimed"


def test_crash_after_claim_consumes_budget_without_retry(registered, monkeypatch):
    root, state_path = registered

    def crash(*args, **kwargs):
        raise RuntimeError("evaluation crashed")

    monkeypatch.setattr(trial, "build_periods", crash)
    with pytest.raises(RuntimeError):
        trial.run(root, fit=True, contract_path=Path("config/trial.json"))
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    monkeypatch.setattr(trial, "load_governed_development_timestamps", lambda *a: pytest.fail("data read"))
    assert trial.run(root, fit=True, contract_path=Path("config/trial.json"))["status"] == "budget_already_claimed"


def test_fail_closed_without_registration_reset_entry_or_with_holdout_rows(registered, monkeypatch):
    root, state_path = registered
    reset = root / "config/research_reset.json"
    reset.write_text(json.dumps({"registered_trials": []}))
    with pytest.raises(ValueError, match="registered_trials"):
        trial.run(root, fit=False, contract_path=Path("config/trial.json"))
    reset.write_text(json.dumps({"registered_trials": ["csm_trial_test"]}))
    mutated = small_contract(hypothesis="changed after registration")
    (root / "config/trial.json").write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="immutable"):
        trial.run(root, fit=False, contract_path=Path("config/trial.json"))
    (root / "config/trial.json").write_text(json.dumps(small_contract()))
    late = pd.DatetimeIndex([pd.Timestamp("2024-01-02 14:30", tz="UTC"), pd.Timestamp("2026-09-10 14:30", tz="UTC")])
    monkeypatch.setattr(trial, "load_governed_development_timestamps", lambda *a: ({s: late for s in SYMBOLS}, {}))
    monkeypatch.setattr(trial, "load_daily_panel", lambda *a: pytest.fail("prices read"))
    with pytest.raises(ValueError, match="outside development"):
        trial.run(root, fit=True, contract_path=Path("config/trial.json"))
    assert json.loads(state_path.read_text())["trials"] == []
    state_path.unlink()
    with pytest.raises(ValueError, match="registration is required"):
        trial.run(root, fit=True, contract_path=Path("config/trial.json"))
