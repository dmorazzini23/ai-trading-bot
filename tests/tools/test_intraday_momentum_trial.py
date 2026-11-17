from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import intraday_momentum_trial as trial
from ai_trading.tools.registered_strategy_trial import bootstrap_indices

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/intraday_momentum_trial_campaign.json"
CAMPAIGN = "intraday_momentum_trial_test"
EARLY_CLOSES = ("2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24")


def small_contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.pop("draft_notice", None)
    payload.update(
        campaign_id=CAMPAIGN, status="registered", development_start="2023-01-03", development_end="2024-12-31",
        acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
        support_thresholds={"minimum_eligible_days": 400, "minimum_supported_years": 2, "minimum_eligible_days_per_year": 200,
                            "minimum_traded_days": 100},
        acceptance=payload["acceptance"] | {"minimum_positive_years": 2},
        bootstrap={"unit": "circular_blocks_of_eligible_days", "block_days": 5, "replicates": 200, "seed": 7,
                   "lower_quantile": 0.0125})
    return payload | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


CALENDAR = trial.session_calendar(small_contract())


def synthetic_bars(calendar=CALENDAR, seed=5):
    """Random half-hour bars on the calendar's regular grid; each bar opens where the last one closed."""
    rng = np.random.default_rng(seed)
    rows, price = [], 400.0
    for session, stamps in zip(calendar.sessions, calendar.stamps):
        price *= float(np.exp(rng.normal(0, 0.004)))  # overnight gap
        for stamp in stamps:
            close = price * float(np.exp(rng.normal(0, 0.002)))
            rows.append(dict(session=session, time=stamp, open=price, high=max(price, close) * 1.0005,
                             low=min(price, close) * 0.9995, close=close, volume=1e6))
            price = close
    return pd.DataFrame(rows, columns=trial.BAR_COLUMNS)


def _bar(bars, session, stamp):
    return bars[(bars.session == session) & (bars.time == stamp)].iloc[0]


# --- contract ---------------------------------------------------------------------------------------

def test_repository_contract_is_registered_with_every_parameter_fixed(tmp_path):
    draft = json.loads(REPO_CONTRACT.read_text())
    reset = json.loads((REPO_CONTRACT.parents[1] / trial.RESET_CONFIG).read_text())
    assert draft["status"] == "registered" and draft["campaign_id"] == "intraday_momentum_trial_20261115"
    assert "draft_notice" not in draft and draft["campaign_id"] in reset["registered_trials"]
    assert any(a["campaign_id"] == draft["campaign_id"] and (REPO_CONTRACT.parents[1] / a["record"]).is_file() for a in reset["amendments"])
    assert draft["promotion_authority"] is False and draft["runtime_authority"] is False
    assert (draft["development_start"], draft["development_end"]) == ("2016-01-04", "2025-12-31")
    assert (draft["holdout_start"], draft["holdout_end"]) == (reset["holdout_start"], reset["holdout_end"]) == ("2026-09-09", "2026-12-08")
    assert draft["max_trials"] == 1 and len(draft["hypotheses"]) == 1 and list(draft["versions"]) == ["long_only", "long_short"]
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft | {"status": "draft_not_registered"}))
    for mode in ("acquire", "support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="not registered"):
            trial.run(tmp_path, mode=mode, contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    assert not (tmp_path / "artifacts").exists()
    full = trial.contract_spec(draft)
    assert full.tiers == (("primary", 1.0, "decision"), ("stress_high", 2.0, trial.STRESS_ROLE), ("measured_median_region", 0.5, "reported_only"))
    assert (full.min_days, full.min_years, full.min_per_year, full.min_traded, full.min_positive_years) == (2300, 9, 230, 800, 6)
    assert (full.block, full.replicates, full.seed, full.lower_quantile) == (21, 10000, 20261115, 0.0125)
    assert (full.vol_lookback, full.vol_reference, full.annualization) == (21, 252, 252)
    assert len(trial.FULL_SESSION) == 13 and (trial.FULL_SESSION[0], trial.FULL_SESSION[-1]) == ("09:30", "15:30")
    assert set(draft["parameter_justifications"]) and len(draft["limitations"]) >= 5


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "raw"}, "adjustment=all"),
        ({"timeframe": "1Min"}, "30Min"),
        ({"runtime_authority": True}, "authority"),
        ({"promotion_authority": True}, "authority"),
        ({"symbols": {"instrument": "QQQ"}}, "only instrument"),
        ({"versions": {"long_short": "", "long_only": ""}}, "versions"),
        ({"signal": {"bar": "10:00", "reference": "prior_session_final_regular_bar_close", "threshold": 0}}, "signal"),
        ({"signal": {"bar": "09:30", "reference": "prior_session_final_regular_bar_close", "threshold": 0.001}}, "no threshold"),
        ({"trade": {"entry_bar": "15:00", "entry_price": "open", "exit_bar": "15:30", "exit_price": "close", "exposure": 1.0,
                    "early_close_sessions": "skip", "sessions_with_missing_bars": "skip"}}, "15:30 bar open"),
        ({"trade": {"entry_bar": "15:30", "entry_price": "open", "exit_bar": "15:30", "exit_price": "close", "exposure": 2.0,
                    "early_close_sessions": "skip", "sessions_with_missing_bars": "skip"}}, "without leverage"),
        ({"trade": {"entry_bar": "15:30", "entry_price": "open", "exit_bar": "15:30", "exit_price": "close", "exposure": 1.0,
                    "early_close_sessions": "trade", "sessions_with_missing_bars": "skip"}}, "must be skipped"),
        ({"cost_tiers": [{"name": "primary", "per_side_bps": 1, "role": "decision"}]}, "stress tier"),
        ({"primary_per_side_cost_bps": 0.5}, "primary_per_side_cost_bps"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))
    disabled = small_contract()
    disabled["acceptance"] = disabled["acceptance"] | {"long_only_positive_mean_vs_always_in_at_primary": False}
    with pytest.raises(ValueError, match="disabled"):
        trial.contract_spec(disabled)


# --- sessions: half days and missing bars -----------------------------------------------------------

def test_half_days_are_never_traded_but_supply_the_next_prior_close():
    bars = synthetic_bars()
    frame, statuses = trial.build_days(bars, CALENDAR, spec())
    by_session = dict(zip(CALENDAR.sessions, statuses))
    assert [s for s, status in by_session.items() if status == "early_close"] == list(EARLY_CLOSES)
    assert by_session["2023-01-03"] == "no_prior_session" and not set(EARLY_CLOSES) & set(frame.session)
    assert len(frame) == len(CALENDAR.dates) - 1 - len(EARLY_CLOSES)
    after = frame[frame.session == "2024-07-05"].iloc[0]
    assert after.prior_session == "2024-07-03" and after.prior_close == _bar(bars, "2024-07-03", "12:30").close
    assert trial.support_summary(spec(), CALENDAR, statuses)["skipped_sessions"] == {"early_close": 5, "no_prior_session": 1}


def test_sessions_with_missing_bars_are_skipped_and_counted():
    bars = synthetic_bars()
    gone = (((bars.session == "2024-03-05") & (bars.time == "11:00"))  # a midday bar: that session only
            | ((bars.session == "2024-03-12") & (bars.time == "15:30")))  # a final bar: that session and the next
    frame, statuses = trial.build_days(bars[~gone], CALENDAR, spec())
    by_session = dict(zip(CALENDAR.sessions, statuses))
    assert by_session["2024-03-05"] == "missing_bars" and by_session["2024-03-06"] == trial.ELIGIBLE
    assert by_session["2024-03-12"] == "missing_bars" and by_session["2024-03-13"] == "prior_close_missing"
    assert by_session["2024-03-14"] == trial.ELIGIBLE
    assert not {"2024-03-05", "2024-03-12", "2024-03-13"} & set(frame.session)
    summary = trial.support_summary(spec(), CALENDAR, statuses)
    assert summary["skipped_sessions"] == {"early_close": 5, "missing_bars": 2, "no_prior_session": 1, "prior_close_missing": 1}
    assert summary["eligible_days"] == len(frame) and summary["status"] == trial.SUPPORT_OK
    thin = trial.support_summary(spec(support_thresholds={"minimum_eligible_days": 600, "minimum_supported_years": 2,
                                                          "minimum_eligible_days_per_year": 249, "minimum_traded_days": 1}),
                                 CALENDAR, statuses)
    assert thin["status"] == "blocked_sampling_support"
    assert thin["blocking_reasons"] == ["eligible_days_below_minimum", "supported_years_below_minimum"]


# --- look-ahead -------------------------------------------------------------------------------------

def test_signal_uses_nothing_after_ten_and_the_trade_is_the_final_bar_open_to_close():
    bars = synthetic_bars()
    frame, _ = trial.build_days(bars, CALENDAR, spec())
    day, before = "2024-05-15", "2024-05-14"
    row = frame[frame.session == day].iloc[0]
    assert row.prior_session == before and row.prior_close == _bar(bars, before, "15:30").close
    assert row.signal_price == _bar(bars, day, "09:30").close
    assert row.signal_return == pytest.approx(_bar(bars, day, "09:30").close / _bar(bars, before, "15:30").close - 1)
    assert row.entry_price == _bar(bars, day, "15:30").open and row.exit_price == _bar(bars, day, "15:30").close
    assert row.trade_return == pytest.approx(row.exit_price / row.entry_price - 1)
    assert row.second_return == pytest.approx(_bar(bars, day, "15:00").close / _bar(bars, day, "14:30").close - 1)

    # Everything from 10:00 on the day, and every later session, is rewritten: the signal and the flags do not move.
    later = bars.copy()
    future = ((later.session == day) & (later.time > "09:30")) | (later.session > day)
    later.loc[future, trial.PRICE_COLUMNS[:4]] *= 1.5
    changed = trial.build_days(later, CALENDAR, spec())[0]
    again = changed[changed.session == day].iloc[0]
    assert again.signal_return == row.signal_return and again.high_volatility == row.high_volatility
    assert again.trailing_volatility == pytest.approx(row.trailing_volatility, nan_ok=True)
    assert again.entry_price == pytest.approx(row.entry_price * 1.5)
    pd.testing.assert_frame_equal(changed[changed.session < day], frame[frame.session < day])

    # The open of the 09:30 bar, and every bar between 10:00 and 15:30, play no part in the plain rule.
    middle = bars.copy()
    unused = (middle.session == day) & (middle.time > "09:30") & (middle.time < "15:30")
    middle.loc[unused, trial.PRICE_COLUMNS[:4]] *= 0.5
    middle.loc[(middle.session == day) & (middle.time == "09:30"), ["open", "high", "low"]] *= 0.9
    middle.loc[(middle.session == day) & (middle.time == "15:30"), ["high", "low"]] *= 1.1
    same = trial.build_days(middle, CALENDAR, spec())[0]
    kept = ["signal_return", "entry_price", "exit_price", "trade_return"]
    assert same.loc[same.session == day, kept].iloc[0].tolist() == row[kept].tolist()

    # The signal moves with the 10:00 price and with the prior close, and with nothing else on those bars.
    moved = bars.copy()
    moved.loc[(moved.session == day) & (moved.time == "09:30"), "close"] *= 1.01
    assert trial.build_days(moved, CALENDAR, spec())[0].set_index("session").at[day, "signal_return"] == pytest.approx(
        (1 + row.signal_return) * 1.01 - 1)


def test_positions_follow_the_sign_and_a_zero_signal_is_flat():
    frame = pd.DataFrame(dict(signal_return=[0.01, -0.01, 0.0, 0.002, -0.003], second_return=[0.001, 0.002, 0.001, -0.001, -0.001],
                              high_volatility=[True, True, True, False, False]))
    assert trial.positions(frame, "long_only").tolist() == [1, 0, 0, 1, 0]
    assert trial.positions(frame, "long_short").tolist() == [1, -1, 0, 1, -1]
    assert trial.positions(frame, "long_short", variant="both_predictors_agree").tolist() == [1, 0, 0, 0, -1]
    assert trial.positions(frame, "long_only", variant="high_volatility").tolist() == [1, 0, 0, 0, 0]
    assert trial.positions(frame, "long_short", variant="high_volatility").tolist() == [1, -1, 0, 0, 0]
    with pytest.raises(ValueError, match="unknown"):
        trial.positions(frame, "short_only")


# --- costs and scoring ------------------------------------------------------------------------------

def test_cost_is_two_sides_on_every_traded_day_and_nothing_on_flat_days():
    position, trade_return = np.array([1.0, -1.0, 0.0]), np.array([0.001, 0.001, 0.05])
    assert trial.net_returns(position, trade_return, 1).tolist() == pytest.approx([0.001 - 0.0002, -0.001 - 0.0002, 0.0])
    assert trial.net_returns(position, trade_return, 2).tolist() == pytest.approx([0.001 - 0.0004, -0.001 - 0.0004, 0.0])
    frame, _ = trial.build_days(synthetic_bars(), CALENDAR, spec())
    report = trial.evaluate(frame, spec())
    for version in trial.VERSIONS:
        figures, tiers = report["versions"][version], report["versions"][version]["cost_tiers"]
        share = figures["traded_days"] / figures["eligible_days"]
        assert tiers["primary"]["net_per_eligible_day_bps"] == pytest.approx(figures["gross_per_eligible_day_bps"] - 2 * share)
        assert tiers["stress_high"]["net_per_eligible_day_bps"] == pytest.approx(figures["gross_per_eligible_day_bps"] - 4 * share)
        assert tiers["measured_median_region"]["net_per_traded_day_bps"] == pytest.approx(tiers["primary"]["net_per_traded_day_bps"] + 1)
        assert tiers["primary"]["always_in_net_per_eligible_day_bps"] == pytest.approx(
            report["reported_only"]["last_half_hour_gross_per_eligible_day_bps"] - 2)
    long_only, long_short = report["versions"]["long_only"], report["versions"]["long_short"]
    assert long_only["short_days"] == 0 and long_only["long_days"] == long_short["long_days"]
    assert long_short["traded_days"] == long_short["long_days"] + long_short["short_days"] == len(frame)


def test_bootstrap_means_match_the_shared_circular_block_resampler():
    values = np.random.default_rng(1).normal(0, 1, 137)
    for block in (1, 5, 21, 68):
        indices = bootstrap_indices(len(values), block=block, replicates=300, seed=11)
        means = trial.bootstrap_means(values, block=block, replicates=300, seed=11)
        np.testing.assert_allclose(means, values[indices].mean(axis=1), atol=1e-12)
    assert trial.bootstrap_means(values, block=69, replicates=300, seed=11) is None


def _days(signal, trade_return):
    frame = trial.build_days(synthetic_bars(), CALENDAR, spec())[0].iloc[:len(signal)].copy()
    return frame.assign(signal_return=signal, trade_return=trade_return)


def _noise(n, seed=2):
    return np.random.default_rng(seed).normal(0, 0.0002, n)


def test_long_short_cannot_rescue_a_failed_long_only():
    n = 480
    signal = np.where(np.arange(n) % 2 == 0, 0.004, -0.004)
    # The last half hour falls a little after an up morning and falls hard after a down morning.
    frame = _days(signal, np.where(signal > 0, -0.0003, -0.003) + _noise(n))
    report = trial.evaluate(frame, spec())
    long_only, long_short = report["versions"]["long_only"], report["versions"]["long_short"]
    assert long_short["decision"] == "eligible_for_untouched_test_planning" and all(long_short["criteria"].values())
    assert long_only["decision"] == "retired" and long_only["criteria"]["positive_mean_vs_always_in_at_primary"] is True
    assert long_only["criteria"]["positive_lower_bound_vs_cash_at_primary"] is False
    assert report["decision"] == "retired" and report["secondary_decision"] == "eligible_for_untouched_test_planning"
    assert "positive_mean_vs_always_in_at_primary" not in long_short["criteria"]
    json.dumps(report, allow_nan=False)


def test_long_only_needs_every_criterion_including_the_timing_comparison():
    n = 480
    signal = np.where(np.arange(n) % 2 == 0, 0.004, -0.004)
    works = trial.evaluate(_days(signal, np.where(signal > 0, 0.002, -0.002) + _noise(n)), spec())
    assert works["decision"] == works["secondary_decision"] == "eligible_for_untouched_test_planning"
    assert works["versions"]["long_only"]["lower_bounds"]["vs_cash_bps"] > 0 and works["versions"]["long_only"]["positive_years"] == 2
    assert "decision" not in works["reported_only_variants"]["high_volatility"]["long_only"]
    assert "criteria" not in works["reported_only_variants"]["both_predictors_agree"]["long_short"]

    # The last half hour drifts up every day whatever the morning did: profitable, but the timing adds nothing.
    drift = trial.evaluate(_days(signal, 0.002 + _noise(n)), spec())
    criteria = drift["versions"]["long_only"]["criteria"]
    assert [name for name, met in criteria.items() if not met] == ["positive_mean_vs_always_in_at_primary"]
    assert drift["decision"] == "retired" and drift["versions"]["long_short"]["decision"] == "retired"

    # Pays at the decision cost, not at the stress cost: 3 bps a traded day before costs.
    thin = trial.evaluate(_days(signal, np.where(signal > 0, 0.0003, -0.0003) + _noise(n) / 100), spec())
    failed = [name for name, met in thin["versions"]["long_only"]["criteria"].items() if not met]
    assert failed == ["positive_at_stress"] and thin["decision"] == "retired"

    # One good year and one losing year fail the year rule even when the average is positive.
    first_year = np.arange(n) < 249
    uneven = trial.evaluate(_days(signal, np.where(signal > 0, np.where(first_year, 0.006, -0.0005), -0.002) + _noise(n)), spec())
    assert uneven["versions"]["long_only"]["positive_years"] == 1 and uneven["decision"] == "retired"
    assert uneven["versions"]["long_only"]["criteria"]["consistent_years"] is False


def test_inadequate_support_after_the_claim_is_inconclusive_not_a_pass():
    n = 480
    rare = np.where(np.arange(n) % 10 == 0, 0.004, 0.0)  # the signal is zero on nine days in ten: 48 traded days
    few = trial.evaluate(_days(rare, np.where(rare > 0, 0.002, 0.0) + _noise(n)), spec())
    assert few["versions"]["long_only"]["traded_days"] == 48 and few["reported_only"]["zero_signal_days"] == 432
    assert few["versions"]["long_only"]["criteria"]["adequate_support"] is False
    assert few["decision"] == few["secondary_decision"] == "inconclusive_budget_consumed"
    signal = np.where(np.arange(8) % 2 == 0, 0.004, -0.004)
    tiny = trial.evaluate(_days(signal, np.where(signal > 0, 0.002, -0.002)), spec())
    assert tiny["decision"] == "inconclusive_budget_consumed"
    assert tiny["versions"]["long_only"]["uncertainty_status"] == "unavailable_fewer_than_two_blocks"


# --- end-to-end with a fake client; synthetic prices only -------------------------------------------

class _FakeData:
    """Serves the synthetic bars plus one pre-market and one after-hours bar per session, as the provider does."""

    def __init__(self, *, extra=None, drop=None):
        self.extra, self.drop, self.requests = extra or [], drop or (lambda session, stamp: False), []
        self.bars = synthetic_bars()

    def get_stock_bars(self, request):
        self.requests.append(request)
        first, last = (t.astimezone(trial.NEW_YORK).date().isoformat() for t in (request.start, request.end))
        served = []
        for session, group in self.bars[(self.bars.session >= first) & (self.bars.session <= last)].groupby("session"):
            rows = [dict(time="08:00", open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0), *group.to_dict("records"),
                    dict(time="16:00", open=2.0, high=2.0, low=2.0, close=2.0, volume=1.0)]
            for row in rows:
                if self.drop(session, row["time"]):
                    continue
                stamp = datetime.fromisoformat(f"{session}T{row['time']}").replace(tzinfo=trial.NEW_YORK).astimezone(UTC)
                served.append(SimpleNamespace(timestamp=stamp, **{c: row[c] for c in trial.PRICE_COLUMNS}))
        served += [SimpleNamespace(timestamp=t, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0) for t in self.extra]
        return SimpleNamespace(data={request.symbol_or_symbols: served})


def _register(root, payload=None):
    payload = payload or small_contract()
    (root / "config").mkdir(exist_ok=True)
    (root / "config/trial.json").write_text(json.dumps(payload))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = root / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return state_path, trial.Clients(trading=None, data=_FakeData())


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    pauses = []
    monkeypatch.setattr(trial, "sleep", pauses.append)
    return pauses


@pytest.fixture
def registered(tmp_path):
    return (tmp_path, *_register(tmp_path))


def _run(root, mode, clients, path="config/trial.json"):
    return trial.run(root, mode=mode, contract_path=Path(path), client_factory=lambda: clients)


def test_acquisition_is_written_once_and_the_stages_claim_exactly_once(registered, monkeypatch):
    root, state_path, clients = registered
    acquired = _run(root, "acquire", clients)
    assert acquired["status"] == "acquisition_written" and acquired["trial_claimed"] is False
    assert [r.start.astimezone(trial.NEW_YORK).date().isoformat() for r in clients.data.requests] == ["2023-01-03", "2024-01-01"]
    assert [r.end.astimezone(trial.NEW_YORK).date().isoformat() for r in clients.data.requests] == ["2023-12-31", "2024-12-31"]
    assert {(str(r.timeframe), str(r.feed), str(r.adjustment), r.symbol_or_symbols) for r in clients.data.requests} == {
        ("30Min", "DataFeed.SIP", "Adjustment.ALL", "SPY")}
    sessions = len(CALENDAR.dates)
    counts = acquired["counts"]
    assert counts["extended_hours_dropped"] == 2 * sessions and counts["regular_bars"] == 13 * sessions - 6 * len(EARLY_CLOSES)
    assert counts["incomplete_full_sessions"] == 0 and counts["early_close_sessions"] == len(EARLY_CLOSES)
    manifest = json.loads(Path(acquired["manifest"]).read_text())
    source = manifest["sources"]["SPY"]
    stored = pd.read_csv(root / source["path"])
    assert hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert set(stored.time) == set(trial.FULL_SESSION) and stored.session.max() <= "2024-12-31"
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", clients)
    assert len(clients.data.requests) == 2

    monkeypatch.setattr(trial, "load_bars", lambda *a: pytest.fail("prices read by the support audit"))
    audit = _run(root, "support-audit", clients)
    assert audit["status"] == trial.SUPPORT_OK and audit["prices_read_for_analysis"] is False and audit["trial_claimed"] is False
    assert audit["eligible_days"] == sessions - 1 - len(EARLY_CLOSES) and audit["strategy_outcomes_computed"] is False
    assert not (state_path.parent / "preflight.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(trial, "sleep", lambda seconds: None)

    preflight = _run(root, "preflight", clients)
    assert preflight["status"] == "preflight_passed" and json.loads(state_path.read_text())["trials"] == []
    assert not (state_path.parent / "report.json").exists() and not (state_path.parent / "days.csv").exists()

    report = _run(root, "fit", clients)
    assert report["decision"] == report["versions"]["long_only"]["decision"]
    assert report["secondary_decision"] == report["versions"]["long_short"]["decision"]
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning"}
    assert report["holdout_evaluated"] is False and report["orders_sent"] == 0 and report["promotion_authority"] is False
    assert report["skipped_sessions"] == {"early_close": 5, "no_prior_session": 1}
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    assert report["output_sha256"]["days.csv"] == hashlib.sha256((state_path.parent / "days.csv").read_bytes()).hexdigest()
    monkeypatch.setattr(trial, "load_stamps", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for mode in ("fit", "preflight", "support-audit", "acquire"):
        assert _run(root, mode, clients)["status"] == "budget_already_claimed"
    assert len(json.loads(state_path.read_text())["trials"]) == 1 and len(clients.data.requests) == 2


def test_the_ledger_records_the_long_only_decision_when_only_long_short_passes(registered, monkeypatch):
    root, state_path, clients = registered
    _run(root, "acquire", clients)
    built = trial.build_days

    def losers_only(bars, calendar, spec_):
        frame, statuses = built(bars, calendar, spec_)
        signal = np.where(np.arange(len(frame)) % 2 == 0, 0.004, -0.004)
        return frame.assign(signal_return=signal, trade_return=np.where(signal > 0, -0.0003, -0.003) + _noise(len(frame))), statuses

    monkeypatch.setattr(trial, "build_days", losers_only)
    report = _run(root, "fit", clients)
    assert (report["decision"], report["secondary_decision"]) == ("retired", "eligible_for_untouched_test_planning")
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["retired"]
    assert _run(root, "fit", clients)["status"] == "budget_already_claimed"


def test_a_short_outage_is_retried_and_a_lasting_one_writes_nothing(registered, _no_retry_pause):
    root, _, clients = registered
    manifest = root / small_contract()["acquisition_manifest"]
    served = clients.data.get_stock_bars

    def down(request):
        raise OSError("provider unavailable")

    clients.data.get_stock_bars = down
    with pytest.raises(OSError):
        _run(root, "acquire", clients)
    assert len(_no_retry_pause) == trial.FETCH_ATTEMPTS - 1 and not manifest.exists() and not list(manifest.parent.glob("*.csv"))
    outages = iter([True, True, False, True])

    def flaky(request):
        if next(outages, False):
            raise OSError("provider unavailable")
        return served(request)

    clients.data.get_stock_bars = flaky
    assert _run(root, "acquire", clients)["status"] == "acquisition_written"
    assert len(_no_retry_pause) == trial.FETCH_ATTEMPTS - 1 + 3


def test_a_degraded_download_is_refused_and_leaves_a_retry_possible(registered):
    root, _, clients = registered
    manifest = root / small_contract()["acquisition_manifest"]
    clients.data.drop = lambda session, stamp: stamp == "15:30" and session[5:7] == "03"  # a month of final bars missing
    with pytest.raises(ValueError, match="degraded download"):
        _run(root, "acquire", clients)
    clients.data.drop = lambda session, stamp: session >= "2024-01-01"  # the provider returned nothing for a year
    with pytest.raises(ValueError, match="no bars returned:SPY:2024"):
        _run(root, "acquire", clients)
    clients.data.drop = lambda session, stamp: False
    clients.data.extra = [datetime(2024, 5, 15, 14, 15, tzinfo=trial.NEW_YORK)]  # not a half-hour stamp
    with pytest.raises(ValueError, match="off the half-hour grid"):
        _run(root, "acquire", clients)
    clients.data.extra = [datetime(2024, 5, 15, 14, 0, tzinfo=trial.NEW_YORK)]
    with pytest.raises(ValueError, match="duplicated"):
        _run(root, "acquire", clients)
    clients.data.extra = [datetime(2024, 3, 29, 11, 0, tzinfo=trial.NEW_YORK)]  # Good Friday: the exchange is closed
    with pytest.raises(ValueError, match="calendar lacks:2024-03-29T11:00"):
        _run(root, "acquire", clients)
    assert not manifest.exists() and not list(manifest.parent.glob("*.csv"))
    clients.data.extra = []
    clients.data.drop = lambda session, stamp: session == "2024-03-05" and stamp == "11:00"  # one bar: tolerated and counted
    acquired = _run(root, "acquire", clients)
    assert acquired["status"] == "acquisition_written" and acquired["counts"]["incomplete_full_sessions"] == 1
    assert _run(root, "support-audit", clients)["skipped_sessions"]["missing_bars"] == 1


def test_holdout_and_2026_bars_are_refused_at_every_stage(registered):
    root, state_path, clients = registered
    manifest_path = state_path.parent / "data/acquisition.json"
    for stamp in (datetime(2026, 9, 10, 15, 30, tzinfo=trial.NEW_YORK), datetime(2026, 1, 2, 15, 30, tzinfo=trial.NEW_YORK),
                  datetime(2022, 12, 30, 15, 30, tzinfo=trial.NEW_YORK)):
        clients.data.extra = [stamp]
        with pytest.raises(ValueError, match="outside development"):
            _run(root, "acquire", clients)
        assert not manifest_path.exists()
    clients.data.extra = []
    _run(root, "acquire", clients)
    manifest = json.loads(manifest_path.read_text())
    source = root / manifest["sources"]["SPY"]["path"]
    source.write_text(source.read_text() + "2026-10-01,15:30,1,1,1,1,1\n")
    manifest["sources"]["SPY"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for mode in ("support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="outside development"):
            _run(root, mode, clients)
    assert json.loads(state_path.read_text())["trials"] == [] and not (state_path.parent / "support_audit.json").exists()


def test_changed_sources_are_refused_and_a_crash_after_the_claim_consumes_the_budget(registered, monkeypatch):
    root, state_path, clients = registered
    _run(root, "acquire", clients)
    path = state_path.parent / "data/SPY_30Min.csv"
    original = path.read_text()
    path.write_text(original.replace("400.", "401.", 1))
    for mode in ("support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="source changed"):
            _run(root, mode, clients)
    path.write_text(original)
    assert json.loads(state_path.read_text())["trials"] == []

    def crash(*args, **kwargs):
        raise RuntimeError("scoring crashed")

    monkeypatch.setattr(trial, "evaluate", crash)
    with pytest.raises(RuntimeError):
        _run(root, "fit", clients)
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    assert _run(root, "fit", clients)["status"] == "budget_already_claimed"


def test_invalid_prices_block_the_preflight_without_a_claim(registered):
    root, state_path, clients = registered
    bad = clients.data.bars.session.str[5:7].isin(["02", "03", "04", "05", "06", "07", "08"]) & (clients.data.bars.time == "12:00")
    clients.data.bars.loc[bad, "low"] = clients.data.bars.loc[bad, "high"] * 2  # low above high: 0.6 percent of bars
    clients.data.drop = lambda session, stamp: False
    _run(root, "acquire", clients)
    assert _run(root, "support-audit", clients)["status"] == trial.SUPPORT_OK  # stamps alone look complete
    preflight = _run(root, "preflight", clients)
    assert preflight["status"] == "blocked_price_validated_support" and preflight["rejection_counts"]["invalid_prices"] == int(bad.sum())
    assert _run(root, "fit", clients)["status"] == "blocked_price_validated_support"
    assert json.loads(state_path.read_text())["trials"] == [] and not (state_path.parent / "report.json").exists()


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


def test_output_hashes_are_deterministic_across_independent_runs(tmp_path):
    results = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        _, clients = _register(root)
        _run(root, "acquire", clients)
        report = _run(root, "fit", clients)
        results.append((report["signature"], report["output_sha256"],
                        hashlib.sha256((root / f"artifacts/{CAMPAIGN}/report.json").read_bytes()).hexdigest()))
    assert results[0] == results[1]


def test_the_rule_based_fallback_calendar_is_refused(monkeypatch):
    from ai_trading.utils import market_calendar

    monkeypatch.setattr(market_calendar, "_get_calendar", lambda: None)
    with pytest.raises(ValueError, match="exchange calendar unavailable"):
        trial.session_calendar(small_contract())
