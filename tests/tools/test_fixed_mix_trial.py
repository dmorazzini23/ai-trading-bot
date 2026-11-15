from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import fixed_mix_trial as trial
from ai_trading.tools.research_feasibility import daily_development_dates

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/fixed_mix_trial_campaign.json"
CAMPAIGN = "fixed_mix_trial_test"
MIX = {"owner_confirmed": True, "cash_buffer_weight": "0.02",
       "assets": [{"symbol": "AAA", "target_weight": "0.58"}, {"symbol": "BBB", "target_weight": "0.40"}]}


def small_contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.update(campaign_id=CAMPAIGN, status="registered", mix=MIX, development_start="2024-01-02",
                   development_end="2024-12-31", acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
                   support_thresholds={"minimum_sessions": 240, "minimum_supported_years": 1,
                                       "minimum_sessions_per_year": 240, "maximum_missing_sessions": 2},
                   bootstrap={"unit": "circular_blocks_of_periods", "block_periods": 2, "replicates": 100, "seed": 7})
    return payload | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


def test_repository_contract_is_an_unregistered_draft_the_runner_refuses(tmp_path):
    draft = json.loads(REPO_CONTRACT.read_text())
    assert draft["status"] == "draft_not_registered" and draft["mix"]["owner_confirmed"] is False
    reset = json.loads((REPO_CONTRACT.parents[1] / trial.RESET_CONFIG).read_text())
    assert draft["campaign_id"] not in reset["registered_trials"]
    with pytest.raises(ValueError, match="owner-confirmed"):
        trial.contract_spec(draft)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="not registered"):
        trial.run(tmp_path, mode="acquire", contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    # Once confirmed, the drafted rules and thresholds parse as written.
    full = trial.contract_spec(draft | {"mix": draft["mix"] | {"owner_confirmed": True}})
    assert full.symbols == ["VTI", "VXUS", "BND", "BIL", "SPY"]
    assert (full.band, full.frequency, full.calendar_min_drift, full.cash_buffer) == (0.05, "quarterly", 0.01, 0.01)
    assert (full.drawdown_ratio_benchmark, full.drawdown_ratio_never, full.sharpe_margin, full.max_turnover) == (0.85, 1.0, 0.0, 0.5)
    assert full.tier("decision")[1] == 10.0 and full.tier(trial.STRESS_ROLE)[1] == 20.0


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "split"}, "adjustment=all"),
        ({"runtime_authority": True}, "authority"),
        ({"mix": MIX | {"owner_confirmed": False}}, "owner-confirmed"),
        ({"mix": MIX | {"cash_buffer_weight": "0.05"}}, "sum to 1"),
        ({"mix": MIX | {"assets": MIX["assets"] + [MIX["assets"][0]]}}, "unique"),
        ({"rebalance": {"band_abs": "0.05", "calendar_frequency": "weekly", "calendar_min_drift": "0.01"}}, "calendar_frequency"),
        ({"cost_tiers": [{"name": "primary", "round_trip_bps": 10, "role": "decision"}]}, "stress tier"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))


def test_trigger_matches_the_portfolio_manager_semantics():
    s = spec()
    january, march, april = date(2024, 1, 2), date(2024, 3, 28), date(2024, 4, 1)
    on_band = np.array([0.63 - 0.58, 0.35 - 0.40, 0.0])  # exactly 5 points in floating point
    assert trial.decide_trigger(s, on_band, january, march) == "none"
    assert trial.decide_trigger(s, np.array([0.0501, -0.0501, 0.0]), january, march) == "band"
    assert trial.decide_trigger(s, np.array([0.0, 0.0, 0.0501]), january, march) == "band"  # the cash buffer counts
    assert trial.decide_trigger(s, np.array([0.02, -0.02, 0.0]), january, march) == "none"  # same quarter
    assert trial.decide_trigger(s, np.array([0.02, -0.02, 0.0]), january, april) == "calendar"
    assert trial.decide_trigger(s, np.array([0.01, -0.01, 0.0]), january, april) == "none"  # on the minimum drift
    assert trial.decide_trigger(s, np.array([0.06, -0.06, 0.0]), january, april) == "band"  # band is reported first
    assert trial.decide_trigger(s, np.array([0.02, -0.02, 0.0]), april, date(2024, 6, 28)) == "none"


def _panel(n, *, jump_at=None, jump=1.0, drift=0.0):
    dates = daily_development_dates(small_contract())[:n]
    index = [d.isoformat() for d in dates]
    aaa = 100 * (1 + drift) ** np.arange(n)
    if jump_at is not None:
        aaa = np.where(np.arange(n) >= jump_at, aaa * jump, aaa)
    flat = np.full(n, 50.0)
    bil = 100 * 1.0001 ** np.arange(n)
    frame = lambda values: pd.DataFrame({"open": values, "close": values}, index=index)  # noqa: E731
    return {"AAA": frame(aaa), "BBB": frame(flat), "SPY": frame(aaa), "BIL": frame(bil)}, dates


def test_band_breach_rebalances_every_asset_to_target_at_the_next_open():
    s = spec()
    panel, dates = _panel(40, jump_at=10, jump=1.5)  # AAA weight goes from 0.58 to about 0.67 at the close of session 10
    frame = trial.simulate(panel, dates, s)
    assert frame.trigger.tolist()[:12] == ["initial"] + ["none"] * 10 + ["band"]
    assert frame.traded.iloc[0] == pytest.approx(0.98) and (frame.traded.iloc[1:11] == 0).all()
    equity = 0.58 * 1.5 + 0.40 + 0.02
    drifts = [abs(0.58 * 1.5 / equity - 0.58), abs(0.40 / equity - 0.40), abs(0.02 / equity - 0.02)]
    assert frame.max_abs_drift.iloc[10] == pytest.approx(max(drifts))
    assert frame.traded.iloc[11] == pytest.approx(drifts[0] + drifts[1])  # both assets trade; the cash move is not a trade
    assert frame.max_abs_drift.iloc[11] == pytest.approx(0.0, abs=1e-12)
    assert (frame.trigger.iloc[12:] == "none").all()
    # The decision uses closes through session 10 only: later prices cannot move it.
    later = {k: v.copy() for k, v in panel.items()}
    later["AAA"].iloc[12:] *= 3
    assert trial.simulate(later, dates, s).trigger.tolist()[:12] == frame.trigger.tolist()[:12]
    trial.validate_sessions(frame, small_contract())


def test_calendar_rebalance_needs_a_new_quarter_and_drift_above_the_minimum():
    s = spec()
    panel, dates = _panel(130, drift=0.0012)  # slow rise: past 1 point before April, inside the band throughout
    frame = trial.simulate(panel, dates, s)
    fired = frame[frame.trigger == "calendar"]
    assert fired.session.tolist() == ["2024-04-01", "2024-07-01"]
    assert (frame.trigger.iloc[1:] != "band").all() and frame.max_abs_drift.max() < 0.05
    quiet = trial.simulate(_panel(130)[0], dates, s)  # no drift at all: the calendar check never trades
    assert (quiet.trigger.iloc[1:] == "none").all() and quiet.traded.iloc[1:].sum() == 0
    assert quiet.never_rebalanced_gross.tolist() == pytest.approx(quiet.strategy_gross.tolist())


def test_cost_accounting_per_tier_charges_traded_weight_and_one_control_entry():
    frame = pd.DataFrame({"strategy_gross": [0.01, 0.02, -0.01], "never_rebalanced_gross": [0.01, 0.03, -0.02],
                          "benchmark_return": [0.02, 0.01, 0.0], "risk_free_return": [0.0001] * 3, "traded": [0.98, 0.1, 0.0]})
    for bps in (6.0, 10.0, 20.0):
        half = bps / 10_000 / 2
        nets = trial.tier_returns(frame, bps, 0.98)
        assert nets["strategy"] == pytest.approx([1.01 * (1 - half * 0.98) - 1, 1.02 * (1 - half * 0.1) - 1, -0.01])
        assert nets["never_rebalanced"] == pytest.approx([1.01 * (1 - half * 0.98) - 1, 0.03, -0.02])
        assert nets["benchmark"] == pytest.approx([1.02 * (1 - half) - 1, 0.01, 0.0])
        assert nets["risk_free"] == pytest.approx([0.0001] * 3)
    assert trial.tier_returns(frame, 0.0, 0.98)["strategy"] == pytest.approx(frame.strategy_gross)
    assert trial.to_periods(np.array([0.1, 0.1, -0.5, 0.0, 0.3]), 2) == pytest.approx([0.21, -0.5])


def test_evaluate_reports_every_tier_and_retires_a_mix_no_safer_than_its_benchmark():
    s = spec()
    panel, dates = _panel(252, jump_at=100, jump=0.6)  # one crash shared by AAA and the benchmark
    report = trial.evaluate(trial.simulate(panel, dates, s), s)
    assert set(report["cost_tiers"]) == {"primary", "stress_high", "stress_low"}
    assert report["rebalances"]["band"] >= 1 and report["annual_turnover"] > 0
    primary, stress = report["cost_tiers"]["primary"], report["cost_tiers"]["stress_high"]
    assert primary["drawdown_ratio_vs_benchmark"] < 0.85  # 58 percent of the crash
    assert stress["strategy"]["cagr"] < primary["strategy"]["cagr"]
    assert report["criteria"]["adequate_support"] is True and report["criteria"]["drawdown_ratio_vs_benchmark"] is True
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning"}
    assert report["decision"] == ("eligible_for_untouched_test_planning" if all(report["criteria"].values()) else "retired")
    json.dumps(report, allow_nan=False)
    short = trial.evaluate(trial.simulate(*_panel(15), s), s)
    assert short["decision"] == "inconclusive_budget_consumed"


# --- end-to-end with a fake data client; synthetic prices only -------------------------------------

class _FakeClient:
    def __init__(self, dates, *, extra=None, skip=None):
        self.dates, self.extra, self.skip, self.requests = dates, extra or [], skip or {}, []

    def get_stock_bars(self, request):
        self.requests.append(request)
        symbol = request.symbol_or_symbols
        rng = np.random.default_rng(sum(map(ord, symbol)))
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.0001 if symbol == "BIL" else 0.012, len(self.dates))))
        bars = [SimpleNamespace(timestamp=datetime(d.year, d.month, d.day, 5, tzinfo=UTC), open=c * 0.999,
                                high=c * 1.01, low=c * 0.99, close=c, volume=1000.0)
                for d, c in zip(self.dates, closes) if d not in self.skip.get(symbol, ())]
        bars += [SimpleNamespace(timestamp=t, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0) for t in self.extra]
        return SimpleNamespace(data={symbol: bars})


def _register(root, payload=None):
    payload = payload or small_contract()
    (root / "config").mkdir(exist_ok=True)
    (root / "config/trial.json").write_text(json.dumps(payload))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = root / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return state_path, _FakeClient(daily_development_dates(payload))


@pytest.fixture
def registered(tmp_path):
    return (tmp_path, *_register(tmp_path))


def _run(root, mode, client):
    return trial.run(root, mode=mode, contract_path=Path("config/trial.json"), client_factory=lambda: client)


def test_stages_never_claim_and_fit_claims_exactly_once(registered, monkeypatch):
    root, state_path, client = registered
    acquired = _run(root, "acquire", client)
    assert acquired["status"] == "acquisition_written" and acquired["trial_claimed"] is False
    assert sorted(r.symbol_or_symbols for r in client.requests) == ["AAA", "BBB", "BIL", "SPY"]
    assert {str(r.adjustment) for r in client.requests} == {"Adjustment.ALL"}
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", client)

    monkeypatch.setattr(trial.daily, "load_panel", lambda *a: pytest.fail("prices read by the support audit"))
    audit = _run(root, "support-audit", client)
    assert audit["status"] == trial.SUPPORT_OK and audit["prices_read_for_analysis"] is False
    assert audit["complete_sessions"] == 252 and audit["trial_claimed"] is False
    assert not (state_path.parent / "preflight.json").exists()
    monkeypatch.undo()

    preflight = _run(root, "preflight", client)
    assert preflight["status"] == "preflight_passed" and preflight["trial_claimed"] is False
    assert json.loads(state_path.read_text())["trials"] == []
    assert not (state_path.parent / "report.json").exists()

    report = _run(root, "fit", client)
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning"}
    assert report["holdout_evaluated"] is False and report["orders_sent"] == 0 and report["sessions"] == 252
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    monkeypatch.setattr(trial.daily, "load_sessions", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for mode in ("fit", "preflight", "support-audit", "acquire"):
        assert _run(root, mode, client)["status"] == "budget_already_claimed"


def test_output_hashes_are_deterministic_across_independent_runs(tmp_path):
    hashes = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        _, client = _register(root)
        _run(root, "acquire", client)
        report = _run(root, "fit", client)
        directory = root / f"artifacts/{CAMPAIGN}"
        hashes.append((report["signature"], report["sessions_sha256"],
                       hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()))
    assert hashes[0] == hashes[1] and all(len(value) == 64 for value in hashes[0])


def test_holdout_and_2026_rows_are_refused_before_anything_is_written_or_read(registered):
    root, state_path, client = registered
    client.extra = [datetime(2026, 9, 10, 4, tzinfo=UTC)]
    with pytest.raises(ValueError, match="outside development"):
        _run(root, "acquire", client)
    assert not (state_path.parent / "data").exists()
    client.extra = []
    _run(root, "acquire", client)
    manifest_path = state_path.parent / "data/acquisition.json"
    manifest = json.loads(manifest_path.read_text())
    source = root / manifest["sources"]["AAA"]["path"]
    source.write_text(source.read_text() + "2026-10-01,1,1,1,1,1\n")
    manifest["sources"]["AAA"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for mode in ("support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="outside development"):
            _run(root, mode, client)
    assert json.loads(state_path.read_text())["trials"] == []
    assert not (state_path.parent / "support_audit.json").exists()


def test_support_audit_blocks_a_late_listed_symbol_without_consuming_the_trial(registered):
    root, state_path, client = registered
    client.skip = {"BBB": set(client.dates[:60])}
    _run(root, "acquire", client)
    audit = _run(root, "support-audit", client)
    assert audit["status"] == "blocked_sampling_support" and audit["missing_sessions"] == 60
    assert set(audit["blocking_reasons"]) == {"sessions_below_minimum", "supported_years_below_minimum", "missing_sessions_above_maximum"}
    assert _run(root, "fit", client)["status"] == "blocked_sampling_support"
    assert json.loads(state_path.read_text())["trials"] == []


def test_changed_source_is_refused_and_a_crash_after_the_claim_consumes_the_budget(registered, monkeypatch):
    root, state_path, client = registered
    _run(root, "acquire", client)
    source = state_path.parent / "data/SPY.csv"
    original = source.read_text()
    source.write_text(original + "\n")
    with pytest.raises(ValueError, match="source changed"):
        _run(root, "preflight", client)
    source.write_text(original)

    def crash(*args, **kwargs):
        raise RuntimeError("simulation crashed")

    monkeypatch.setattr(trial, "simulate", crash)
    with pytest.raises(RuntimeError):
        _run(root, "fit", client)
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    assert _run(root, "fit", client)["status"] == "budget_already_claimed"


def test_fail_closed_without_registration_reset_entry_or_original_ledger(registered):
    root, state_path, _ = registered
    refuse = lambda path: trial.run(root, mode="acquire", contract_path=Path(path),  # noqa: E731
                                    client_factory=lambda: pytest.fail("fetched"))
    (root / "config/draft.json").write_text(json.dumps(small_contract(status="draft_not_registered")))
    with pytest.raises(ValueError, match="not registered"):
        refuse("config/draft.json")
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": []}))
    with pytest.raises(ValueError, match="registered_trials"):
        refuse("config/trial.json")
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    (root / "config/trial.json").write_text(json.dumps(small_contract(hypothesis="changed after registration")))
    with pytest.raises(ValueError, match="immutable"):
        refuse("config/trial.json")
    (root / "config/trial.json").write_text(json.dumps(small_contract()))
    state_path.unlink()
    with pytest.raises(ValueError, match="registration is required"):
        refuse("config/trial.json")
