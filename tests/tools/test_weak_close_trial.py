from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import weak_close_trial as trial

REPO = Path(__file__).resolve().parents[2]
REPO_CONTRACT = REPO / "config/weak_close_trial_campaign.json"
CAMPAIGN = "weak_close_trial_test"
EARLY_CLOSES = ("2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24")


def small_contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.pop("draft_notice", None)
    payload.update(
        campaign_id=CAMPAIGN, status="registered", development_start="2023-01-03", development_end="2024-12-31",
        acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
        support_thresholds={"minimum_observations": 400, "minimum_supported_years": 2, "minimum_observations_per_year": 200,
                            "minimum_snapshot_sessions_per_symbol": 450, "minimum_signal_sessions_per_symbol": 30},
        acceptance=payload["acceptance"] | {"minimum_positive_years": 2},
        bootstrap={"unit": "circular_blocks_of_observations", "block_days": 5, "replicates": 200, "seed": 7,
                   "lower_quantile": 0.0125})
    return payload | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


CALENDAR = trial.session_calendar(small_contract())


@cache
def _synthetic(seed=5):
    """Random fifteen-minute bars on the calendar's regular grid and the daily bars they make; synthetic only."""
    rng = np.random.default_rng(seed)
    bars, days = [], []
    for symbol, price in (("SPY", 400.0), ("QQQ", 300.0)):
        for session, stamps in zip(CALENDAR.sessions, CALENDAR.stamps):
            price *= float(np.exp(rng.normal(0, 0.004)))  # overnight gap
            first, high, low = price, -np.inf, np.inf
            for stamp in stamps:
                close = price * float(np.exp(rng.normal(0, 0.002)))
                bar = dict(symbol=symbol, session=session, time=stamp, open=price, high=max(price, close) * 1.0005,
                           low=min(price, close) * 0.9995, close=close, volume=1e5)
                bars.append(bar)
                high, low, price = max(high, bar["high"]), min(low, bar["low"]), close
            days.append(dict(symbol=symbol, session=session, open=first, high=high, low=low, close=price, volume=1e6))
    return pd.DataFrame(bars, columns=trial.BAR_COLUMNS), pd.DataFrame(days, columns=trial.DAILY_COLUMNS)


def synthetic():
    bars, days = _synthetic()
    return bars.copy(), days.copy()


def _mask(frame, symbol, session, time=None):
    hit = (frame.symbol == symbol) & (frame.session == session)
    return hit & (frame.time == time) if time is not None else hit


def _row(frame, session):
    return frame[frame.session == session].iloc[0]


# --- contract ---------------------------------------------------------------------------------------

def test_repository_contract_is_an_unregistered_draft_with_every_parameter_fixed(tmp_path):
    draft = json.loads(REPO_CONTRACT.read_text())
    reset = json.loads((REPO / trial.RESET_CONFIG).read_text())
    assert draft["status"] == "draft_not_registered" and draft["draft_notice"]
    assert draft["campaign_id"] == "weak_close_trial_20261120" and draft["campaign_id"] not in reset["registered_trials"]
    assert not any(a["campaign_id"] == draft["campaign_id"] for a in reset["amendments"])
    assert not (REPO / "artifacts" / draft["campaign_id"]).exists()
    assert draft["promotion_authority"] is False and draft["runtime_authority"] is False
    assert (draft["development_start"], draft["development_end"]) == ("2016-01-04", "2025-12-31")
    assert (draft["holdout_start"], draft["holdout_end"]) == (reset["holdout_start"], reset["holdout_end"]) == ("2026-09-09", "2026-12-08")
    assert draft["max_trials"] == 1 and len(draft["hypotheses"]) == 1 and list(draft["versions"]) == ["long_only"]
    assert draft["runner"] == "ai_trading/tools/weak_close_trial.py"
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft))
    for mode in ("acquire", "support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="not registered"):
            trial.run(tmp_path, mode=mode, contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    assert not (tmp_path / "artifacts").exists()
    full = trial.contract_spec(draft)
    assert full.threshold == 0.2 and trial.SLOT_WEIGHT == 0.5
    assert full.tiers == (("primary", 1.0, "decision"), ("stress_high", 2.0, trial.STRESS_ROLE),
                          ("half", 0.5, "reported_only"), ("five", 5.0, "reported_only"))
    assert (full.min_observations, full.min_years, full.min_per_year, full.min_snapshot_sessions, full.min_signal_sessions,
            full.min_positive_years) == (2300, 9, 230, 2300, 300, 6)
    assert (full.block, full.replicates, full.seed, full.lower_quantile) == (21, 10000, 20261120, 0.0125)
    assert len(trial.FULL_SESSION) == 26 and trial.SNAPSHOT_BARS[0] == "09:30" and trial.SNAPSHOT_BARS[-1] == "15:30"
    assert len(trial.SNAPSHOT_BARS) == 25 and trial.PRICE_BAR == "15:30"
    assert set(draft["parameter_justifications"]) and len(draft["limitations"]) >= 5


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "raw"}, "adjustment=all"),
        ({"timeframes": {"signal": "5Min", "prices": "1Day"}}, "15Min"),
        ({"runtime_authority": True}, "authority"),
        ({"promotion_authority": True}, "authority"),
        ({"symbols": {"instruments": ["SPY", "IWM"]}}, "instruments"),
        ({"versions": {"long_only": "", "long_short": ""}}, "only version"),
        ({"signal": {"bar_minutes": 15, "snapshot_time": "15:55", "threshold": 0.2, "comparison": "strictly_below",
                     "zero_range": "no_signal"}}, "15:45"),
        ({"signal": {"bar_minutes": 15, "snapshot_time": "15:45", "threshold": 0.2, "comparison": "at_or_below",
                     "zero_range": "no_signal"}}, "strictly below"),
        ({"signal": {"bar_minutes": 15, "snapshot_time": "15:45", "threshold": 1.5, "comparison": "strictly_below",
                     "zero_range": "no_signal"}}, "threshold"),
        ({"trade": {"entry": "next_open", "exit": "market_on_close_next_session", "price": "daily_bar_close",
                    "repeat_signal": "hold_without_trading", "early_close_sessions": "no_entry", "slot_weight": 0.5}}, "closing auction"),
        ({"trade": {"entry": "market_on_close_same_session", "exit": "market_on_close_next_session", "price": "daily_bar_close",
                    "repeat_signal": "hold_without_trading", "early_close_sessions": "no_entry", "slot_weight": 1.0}}, "slot"),
        ({"cost_tiers": [{"name": "primary", "per_side_bps": 1, "role": "decision"}]}, "stress tier"),
        ({"primary_per_side_cost_bps": 0.5}, "primary_per_side_cost_bps"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))


@pytest.mark.parametrize("criterion", ["positive_lower_bound_vs_cash_at_primary", "positive_lower_bound_vs_exposure_matched_at_primary",
                                       "positive_paired_mean_at_stress", "each_symbol_positive_paired_mean_at_primary"])
def test_no_acceptance_criterion_can_be_disabled(criterion):
    disabled = small_contract()
    disabled["acceptance"] = disabled["acceptance"] | {criterion: False}
    with pytest.raises(ValueError, match="disabled"):
        trial.contract_spec(disabled)


# --- the signal -------------------------------------------------------------------------------------

def test_ibs_is_strictly_below_the_threshold_and_a_zero_range_never_signals():
    ibs = trial.internal_bar_strength(np.array([100.0, 101.0, 102.0, 105.0, 100.0, np.nan]),
                                      np.array([110.0, 110.0, 110.0, 110.0, 100.0, 110.0]),
                                      np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
    np.testing.assert_allclose(ibs, [0.0, 0.1, 0.2, 0.5, np.nan, np.nan])
    assert trial.signal_weights(ibs, 0.2).tolist() == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    assert trial.signal_weights(np.array([0.19999, 0.20001]), 0.2).tolist() == [0.5, 0.0]


def test_the_signal_is_read_from_the_bars_up_to_15_45_only():
    bars, days = synthetic()
    day = "2024-05-15"
    snapshot = _mask(bars, "SPY", day) & bars.time.isin(trial.SNAPSHOT_BARS)
    bars.loc[snapshot, ["open", "close"]] = 105.0
    bars.loc[snapshot, "high"], bars.loc[snapshot, "low"] = 110.0, 100.0
    bars.loc[_mask(bars, "SPY", day, "15:30"), "close"] = 101.9  # IBS at 15:45 is 0.19
    bars.loc[_mask(bars, "SPY", day, "15:45"), ["open", "high", "low", "close"]] = [101.9, 120.0, 101.0, 119.0]
    days.loc[_mask(days, "SPY", day), ["high", "low", "close"]] = [120.0, 100.0, 119.0]  # the final IBS is 0.95
    frame, statuses = trial.build_days(bars, days, CALENDAR, spec())
    after = _row(frame, "2024-05-16")
    assert after.prior_session == day and after.SPY_signal_ibs == pytest.approx(0.19) and after.SPY_held == 0.5
    assert after.SPY_textbook_signal_ibs == pytest.approx(0.95) and after.SPY_textbook_held == 0.0
    assert _row(frame, day).SPY_after == 0.5
    bars.loc[_mask(bars, "SPY", day, "15:30"), "close"] = 102.0  # exactly 0.2: no signal
    assert _row(trial.build_days(bars, days, CALENDAR, spec())[0], "2024-05-16").SPY_held == 0.0
    assert statuses["SPY"][CALENDAR.sessions.index(day)] == trial.SIGNAL_POSSIBLE


def test_early_closes_missing_bars_and_the_last_session_never_open_a_position():
    bars, days = synthetic()
    gone = _mask(bars, "QQQ", "2024-03-05", "11:00") | _mask(bars, "SPY", "2024-03-12", "15:45")
    frame, statuses = trial.build_days(bars[~gone], days, CALENDAR, spec())
    by_session = dict(zip(CALENDAR.sessions, statuses["QQQ"]))
    assert [s for s, status in by_session.items() if status == "early_close"] == list(EARLY_CLOSES)
    assert by_session["2024-03-05"] == "missing_snapshot_bars" and by_session["2024-12-31"] == "last_session"
    assert dict(zip(CALENDAR.sessions, statuses["SPY"]))["2024-03-12"] == trial.SIGNAL_POSSIBLE  # the 15:45 bar is not needed
    for session in (*EARLY_CLOSES, "2024-03-05", "2024-12-31"):
        assert _row(frame, session).QQQ_after == 0.0
    assert _row(frame, "2024-03-06").QQQ_held == 0.0 and _row(frame, "2024-12-31").SPY_after == 0.0
    assert len(frame) == len(CALENDAR.dates) - 1 and frame.session.iloc[0] == "2023-01-04"
    support = trial.support_summary(spec(), CALENDAR, statuses, {s: set(CALENDAR.sessions) for s in trial.SYMBOLS})
    assert support["observations"] == len(frame) and support["status"] == trial.SUPPORT_OK
    assert support["sessions_without_signal"]["QQQ"] == {"early_close": 5, "last_session": 1, "missing_snapshot_bars": 1}
    thin = trial.support_summary(spec(), CALENDAR, statuses, {"SPY": set(CALENDAR.sessions[:300]), "QQQ": set(CALENDAR.sessions)})
    assert thin["status"] == "blocked_sampling_support"
    assert thin["blocking_reasons"] == ["observations_below_minimum", "supported_years_below_minimum"]


def test_nothing_after_the_15_45_snapshot_moves_a_decision():
    bars, days = synthetic()
    frame, _ = trial.build_days(bars, days, CALENDAR, spec())
    day = "2024-05-15"
    later = bars.copy()
    future = (later.session > day) | ((later.session == day) & (later.time >= trial.SNAPSHOT_TIME))
    later.loc[future, trial.PRICE_COLUMNS[:4]] *= 0.7
    later_days = days.copy()
    later_days.loc[later_days.session >= day, ["open", "high", "low", "close"]] *= 1.3  # the final daily bar of the day too
    changed, _ = trial.build_days(later, later_days, CALENDAR, spec())
    decided = [f"{s}_{k}" for s in trial.SYMBOLS for k in ("held", "before", "signal_ibs")]
    pd.testing.assert_frame_equal(changed.loc[changed.prior_session <= day, decided], frame.loc[frame.prior_session <= day, decided])
    after = [f"{s}_after" for s in trial.SYMBOLS]
    pd.testing.assert_frame_equal(changed.loc[changed.session <= day, after], frame.loc[frame.session <= day, after])
    pd.testing.assert_frame_equal(changed[changed.session < day], frame[frame.session < day])


# --- entry, exit and costs --------------------------------------------------------------------------

def test_a_position_earns_the_daily_close_to_the_next_daily_close():
    bars, days = synthetic()
    frame, _ = trial.build_days(bars, days, CALENDAR, spec())
    held = frame[frame.SPY_held > 0].iloc[3]
    days.loc[_mask(days, "SPY", held.session), "close"] *= 1.001  # the closing auction differs from the last bar
    moved = _row(trial.build_days(bars, days, CALENDAR, spec())[0], held.session)
    closes = days[days.symbol == "SPY"].set_index("session").close
    assert moved.SPY_return == pytest.approx(closes[held.session] / closes[held.prior_session] - 1)
    assert moved.SPY_overnight == pytest.approx(days[days.symbol == "SPY"].set_index("session").open[held.session]
                                                / closes[held.prior_session] - 1)
    assert moved.SPY_return == pytest.approx((1 + held.SPY_return) * 1.001 - 1)
    trial.validate_days(frame, small_contract())


def _frame(held, returns, *, start=0):
    """Accounting rows from per-interval weights and returns; the textbook columns copy the snapshot ones."""
    sessions = CALENDAR.sessions[start:start + len(next(iter(returns.values()))) + 1]
    columns = dict(session=sessions[1:], prior_session=sessions[:-1], year=[s[:4] for s in sessions[1:]])
    for symbol in trial.SYMBOLS:
        weight, ret = np.asarray(held[symbol], dtype=float), np.asarray(returns[symbol], dtype=float)
        for prefix in ("", "textbook_"):
            columns[f"{symbol}_{prefix}held"] = weight
            columns[f"{symbol}_{prefix}before"] = np.concatenate([[0.0], weight[:-1]])
            columns[f"{symbol}_{prefix}after"] = np.concatenate([weight[1:], [0.0]])
            columns[f"{symbol}_{prefix}signal_ibs"] = np.where(weight > 0, 0.1, 0.5)
        close = 100 * np.cumprod(np.concatenate([[1.0], 1 + ret]))
        columns.update({f"{symbol}_prior_close": close[:-1], f"{symbol}_open": close[1:], f"{symbol}_close": close[1:],
                        f"{symbol}_return": ret, f"{symbol}_overnight": np.zeros(len(ret))})
    return pd.DataFrame(columns)


def test_each_round_trip_pays_two_sides_once_and_a_repeated_signal_trades_nothing():
    held = np.array([0, 0.5, 0.5, 0, 0.5, 0])
    ret = np.array([0.01, 0.02, -0.01, 0.03, 0.04, 0.05])
    frame = _frame({"SPY": held, "QQQ": np.zeros(6)}, {"SPY": ret, "QQQ": ret})
    trial.validate_days(frame, small_contract())
    assert trial.symbol_net(frame, "SPY", 1).tolist() == pytest.approx(
        [0.0, 0.01 - 0.00005, -0.005 - 0.00005, 0.0, 0.02 - 0.0001, 0.0])
    gross, net = trial.symbol_net(frame, "SPY", 0), trial.symbol_net(frame, "SPY", 2)
    assert (gross - net).sum() == pytest.approx(2 * 2 * 0.5 * 2 / 10_000)  # two round trips, two sides, half a slot, 2 bps
    assert trial.round_trip_lengths(held) == [2, 1] and trial.round_trip_lengths(np.array([0.5, 0.5])) == [2]
    assert trial.symbol_net(frame, "QQQ", 1).tolist() == [0.0] * 6
    broken = frame.copy()
    broken.loc[2, "SPY_before"] = 0.0
    with pytest.raises(ValueError, match="chain"):
        trial.validate_days(broken, small_contract())
    unclosed = frame.copy()
    unclosed.loc[5, ["SPY_held", "SPY_after"]] = [0.0, 0.5]
    with pytest.raises(ValueError, match="chain"):
        trial.validate_days(unclosed, small_contract())


def test_the_exposure_matched_control_holds_the_same_average_weight_every_interval():
    held = np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0])
    ret = np.linspace(-0.01, 0.02, 8)
    frame = _frame({"SPY": held, "QQQ": np.zeros(8)}, {"SPY": ret, "QQQ": ret * 2})
    np.testing.assert_allclose(trial.matched_control(frame, "SPY"), 0.125 * ret)
    np.testing.assert_allclose(trial.matched_control(frame, "QQQ"), 0.0)
    net, control = trial.portfolio(frame, 0.0)
    assert control.mean() == pytest.approx(held.mean() * ret.mean())
    assert (net - control).mean() == pytest.approx(np.mean(held * ret) - held.mean() * ret.mean())  # timing only


# --- acceptance ---------------------------------------------------------------------------------------

N = len(CALENDAR.dates) - 1
SIGNALS = (np.arange(N) % 5 == 2) & (np.arange(N) < N - 1)  # one interval in five, never the last


def _noise(seed=2, scale=0.0002):
    return np.random.default_rng(seed).normal(0, scale, N)


def _evaluate(spy_on, spy_off, qqq_on=None, qqq_off=None, *, noise=1.0, signals=SIGNALS):
    held = np.where(signals, 0.5, 0.0)
    qqq_on = spy_on if qqq_on is None else qqq_on
    qqq_off = spy_off if qqq_off is None else qqq_off
    returns = {"SPY": np.where(signals, spy_on, spy_off) + noise * _noise(2),
               "QQQ": np.where(signals, qqq_on, qqq_off) + noise * _noise(3)}
    return trial.evaluate(_frame({"SPY": held, "QQQ": held}, returns), spec())


def test_a_timing_edge_in_both_etfs_passes_every_criterion():
    works = _evaluate(0.004, -0.0002)
    assert works["decision"] == "eligible_for_untouched_test_planning" and all(works["criteria"].values())
    assert works["lower_bounds"]["vs_cash_bps"] > 0 and works["lower_bounds"]["vs_exposure_matched_bps"] > 0
    assert works["signal_sessions"] == {"SPY": int(SIGNALS.sum()), "QQQ": int(SIGNALS.sum())}
    assert works["round_trips"] == works["signal_sessions"] and works["positive_years"] == 2
    assert works["reported_only"]["sessions_with_both"] == int(SIGNALS.sum())
    assert works["reported_only"]["average_exposure"] == pytest.approx(SIGNALS.mean())
    json.dumps(works, allow_nan=False)


def test_drift_alone_beats_cash_but_not_the_exposure_matched_control():
    drift = _evaluate(0.001, 0.001)  # every interval rises the same: no timing skill
    failed = [name for name, met in drift["criteria"].items() if not met]
    assert drift["criteria"]["positive_lower_bound_vs_cash_at_primary"] is True
    assert "positive_lower_bound_vs_exposure_matched_at_primary" in failed and drift["decision"] == "retired"


def test_one_etf_cannot_carry_the_other():
    lopsided = _evaluate(0.008, -0.0002, qqq_on=-0.0005, qqq_off=0.0)
    assert lopsided["paired_by_symbol_bps"]["SPY"] > 0 > lopsided["paired_by_symbol_bps"]["QQQ"]
    assert [name for name, met in lopsided["criteria"].items() if not met] == ["each_symbol_positive_paired_mean_at_primary"]
    assert lopsided["decision"] == "retired"


def test_an_edge_that_pays_at_one_bp_but_not_at_two_fails_the_stress_tier():
    thin = _evaluate(0.0003, 0.0, noise=0.0)  # 1.2 bps of timing per round trip against 1 or 2 bps of cost
    assert [name for name, met in thin["criteria"].items() if not met] == ["positive_paired_mean_at_stress"]
    tiers = thin["cost_tiers"]
    assert tiers["primary"]["paired_vs_exposure_matched_bps"] > 0 > tiers["stress_high"]["paired_vs_exposure_matched_bps"]
    assert tiers["half"]["net_per_observation_bps"] > tiers["primary"]["net_per_observation_bps"] > tiers["five"]["net_per_observation_bps"]


def test_one_good_year_is_not_enough():
    first_year = np.array([s < "2024" for s in CALENDAR.sessions[1:]])
    uneven = _evaluate(np.where(first_year, 0.01, -0.001), -0.0002)
    assert uneven["positive_years"] == 1 and uneven["criteria"]["consistent_years"] is False
    assert uneven["decision"] == "retired"


def test_too_few_signals_after_the_claim_is_inconclusive_not_a_pass():
    rare = (np.arange(N) % 25 == 3) & (np.arange(N) < N - 1)  # 20 signals per ETF against a minimum of 30
    few = _evaluate(0.004, -0.0002, signals=rare)
    assert few["signal_sessions"]["SPY"] < 30 and few["criteria"]["adequate_support"] is False
    assert few["decision"] == "inconclusive_budget_consumed"
    tiny = trial.evaluate(_frame({"SPY": [0, 0.5, 0, 0], "QQQ": [0, 0.5, 0, 0]},
                                 {"SPY": [0.0, 0.01, 0.0, 0.0], "QQQ": [0.0, 0.01, 0.0, 0.0]}), spec())
    assert tiny["decision"] == "inconclusive_budget_consumed" and tiny["uncertainty_status"] == "unavailable_fewer_than_two_blocks"


# --- end to end with a fake client; synthetic prices only -------------------------------------------

class _FakeData:
    """Serves the synthetic bars, plus one pre-market and one after-hours bar per session for intraday requests."""

    def __init__(self):
        self.bars, self.days = synthetic()
        self.extra, self.requests = [], []
        self.drop = lambda symbol, session, stamp: False
        self.drop_day = lambda symbol, session: False

    def get_stock_bars(self, request):
        self.requests.append(request)
        symbol = request.symbol_or_symbols
        first, last = (t.astimezone(trial.NEW_YORK).date().isoformat() for t in (request.start, request.end))
        daily_request = str(request.timeframe) == "1Day"
        served = []

        def bar(session, stamp, row):
            when = datetime.fromisoformat(f"{session}T{stamp}").replace(tzinfo=trial.NEW_YORK).astimezone(UTC)
            return SimpleNamespace(timestamp=when, **{c: row[c] for c in trial.PRICE_COLUMNS})

        if daily_request:
            part = self.days[(self.days.symbol == symbol) & (self.days.session >= first) & (self.days.session <= last)]
            served = [bar(r["session"], "00:00", r) for r in part.to_dict("records") if not self.drop_day(symbol, r["session"])]
        else:
            part = self.bars[(self.bars.symbol == symbol) & (self.bars.session >= first) & (self.bars.session <= last)]
            flat = dict(open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
            for session, group in part.groupby("session"):
                for row in [dict(flat, time="08:00"), *group.to_dict("records"), dict(flat, time="16:00")]:
                    if not self.drop(symbol, session, row["time"]):
                        served.append(bar(session, row["time"], row))
        served += [SimpleNamespace(timestamp=t, **dict(open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
                   for kind, s, t in self.extra if s == symbol and kind == ("daily" if daily_request else "intraday")]
        return SimpleNamespace(data={symbol: served})


def _register(root, payload=None):
    payload = payload or small_contract()
    (root / "config").mkdir(exist_ok=True)
    (root / "config/trial.json").write_text(json.dumps(payload))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = root / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return state_path, SimpleNamespace(trading=None, data=_FakeData())


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
    kinds = [(r.symbol_or_symbols, str(r.timeframe)) for r in clients.data.requests]
    assert kinds == [("SPY", "15Min"), ("SPY", "15Min"), ("QQQ", "15Min"), ("QQQ", "15Min"), ("SPY", "1Day"), ("QQQ", "1Day")]
    assert {(str(r.feed), str(r.adjustment)) for r in clients.data.requests} == {("DataFeed.SIP", "Adjustment.ALL")}
    sessions = len(CALENDAR.dates)
    for symbol in trial.SYMBOLS:
        counts = acquired["counts"][symbol]
        assert counts["extended_hours_dropped"] == 2 * sessions and counts["regular_bars"] == 26 * sessions - 12 * len(EARLY_CLOSES)
        assert counts["incomplete_full_sessions"] == 0 and counts["early_close_sessions"] == len(EARLY_CLOSES)
    manifest = json.loads(Path(acquired["manifest"]).read_text())
    assert sorted(manifest["sources"]) == ["QQQ_15Min", "QQQ_1Day", "SPY_15Min", "SPY_1Day"]
    for source in manifest["sources"].values():
        assert hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() == source["sha256"]
        assert source["last_session"] <= "2024-12-31"
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", clients)
    assert len(clients.data.requests) == 6

    monkeypatch.setattr(trial, "load_prices", lambda *a: pytest.fail("prices read by the support audit"))
    audit = _run(root, "support-audit", clients)
    assert audit["status"] == trial.SUPPORT_OK and audit["prices_read_for_analysis"] is False and audit["trial_claimed"] is False
    assert audit["observations"] == sessions - 1 and audit["strategy_outcomes_computed"] is False
    assert audit["snapshot_sessions"] == {s: sessions - 1 - len(EARLY_CLOSES) for s in trial.SYMBOLS}
    assert not (state_path.parent / "preflight.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(trial, "sleep", lambda seconds: None)

    preflight = _run(root, "preflight", clients)
    assert preflight["status"] == "preflight_passed" and preflight["reasons"] == []
    assert preflight["daily_close_mismatch_sessions"] == {"SPY": 0, "QQQ": 0}
    assert json.loads(state_path.read_text())["trials"] == []
    assert not (state_path.parent / "report.json").exists() and not (state_path.parent / "days.csv").exists()

    report = _run(root, "fit", clients)
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning", "inconclusive_budget_consumed"}
    assert report["holdout_evaluated"] is False and report["orders_sent"] == 0 and report["promotion_authority"] is False
    assert report["observations"] == sessions - 1
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    assert report["output_sha256"]["days.csv"] == hashlib.sha256((state_path.parent / "days.csv").read_bytes()).hexdigest()

    # A second fit, or any other stage, never re-reads data or re-scores once the trial is claimed.
    monkeypatch.setattr(trial, "load_stamps", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for mode in ("fit", "preflight", "support-audit", "acquire"):
        assert _run(root, mode, clients)["status"] == "budget_already_claimed"
    assert len(json.loads(state_path.read_text())["trials"]) == 1 and len(clients.data.requests) == 6


def test_a_crash_after_the_claim_consumes_the_budget(registered, monkeypatch):
    root, state_path, clients = registered
    _run(root, "acquire", clients)

    def crash(*args, **kwargs):
        raise RuntimeError("scoring crashed")

    monkeypatch.setattr(trial, "evaluate", crash)
    with pytest.raises(RuntimeError):
        _run(root, "fit", clients)
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    assert _run(root, "fit", clients)["status"] == "budget_already_claimed"


def test_holdout_and_2026_bars_are_refused_at_every_stage(registered):
    root, state_path, clients = registered
    manifest_path = state_path.parent / "data/acquisition.json"
    for kind, stamp in (("intraday", datetime(2026, 9, 10, 15, 30, tzinfo=trial.NEW_YORK)),
                        ("intraday", datetime(2026, 1, 2, 15, 30, tzinfo=trial.NEW_YORK)),
                        ("daily", datetime(2026, 10, 1, 0, 0, tzinfo=trial.NEW_YORK)),
                        ("daily", datetime(2022, 12, 30, 0, 0, tzinfo=trial.NEW_YORK))):
        clients.data.extra = [(kind, "QQQ", stamp)]
        with pytest.raises(ValueError, match="outside development"):
            _run(root, "acquire", clients)
        assert not manifest_path.exists() and not list(manifest_path.parent.glob("*.csv"))
    clients.data.extra = []
    _run(root, "acquire", clients)
    manifest = json.loads(manifest_path.read_text())
    for name, line in (("SPY_15Min", "SPY,2026-10-01,15:30,1,1,1,1,1\n"), ("QQQ_1Day", "QQQ,2026-10-01,1,1,1,1,1\n")):
        source = root / manifest["sources"][name]["path"]
        original, original_hash = source.read_text(), manifest["sources"][name]["sha256"]
        source.write_text(original + line)
        manifest["sources"][name]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        for mode in ("support-audit", "preflight", "fit"):
            with pytest.raises(ValueError, match="outside development"):
                _run(root, mode, clients)
        source.write_text(original)
        manifest["sources"][name]["sha256"] = original_hash
    assert json.loads(state_path.read_text())["trials"] == [] and not (state_path.parent / "support_audit.json").exists()


def test_a_degraded_download_is_refused_and_writes_nothing(registered):
    root, _, clients = registered
    manifest = root / small_contract()["acquisition_manifest"]
    cases = [
        (lambda: setattr(clients.data, "drop", lambda sym, session, stamp: sym == "QQQ" and stamp == "15:30" and session[5:7] == "03"),
         "degraded download:QQQ"),
        (lambda: setattr(clients.data, "drop_day", lambda sym, session: sym == "SPY" and session == "2024-05-15"),
         "daily bars missing:SPY:1"),
        (lambda: setattr(clients.data, "extra", [("intraday", "SPY", datetime(2024, 5, 15, 14, 20, tzinfo=trial.NEW_YORK))]),
         "off the fifteen-minute grid"),
        (lambda: setattr(clients.data, "extra", [("intraday", "SPY", datetime(2024, 3, 29, 11, 0, tzinfo=trial.NEW_YORK))]),
         "calendar lacks:2024-03-29T11:00"),
        (lambda: setattr(clients.data, "extra", [("daily", "SPY", datetime(2024, 3, 29, 0, 0, tzinfo=trial.NEW_YORK))]),
         "daily bar on a date the exchange calendar lacks:SPY:2024-03-29"),
        (lambda: setattr(clients.data, "extra", [("daily", "SPY", datetime(2024, 5, 15, 0, 0, tzinfo=trial.NEW_YORK))]),
         "duplicate daily bars:SPY"),
    ]
    for arrange, match in cases:
        clients.data.extra, clients.data.drop = [], (lambda sym, session, stamp: False)
        clients.data.drop_day = lambda sym, session: False
        arrange()
        with pytest.raises(ValueError, match=match):
            _run(root, "acquire", clients)
        assert not manifest.exists() and not list(manifest.parent.glob("*.csv"))
    clients.data.extra, clients.data.drop_day = [], (lambda sym, session: False)
    clients.data.drop = lambda sym, session, stamp: sym == "SPY" and session == "2024-03-05" and stamp == "11:00"
    acquired = _run(root, "acquire", clients)
    assert acquired["status"] == "acquisition_written" and acquired["counts"]["SPY"]["incomplete_full_sessions"] == 1
    assert _run(root, "support-audit", clients)["sessions_without_signal"]["SPY"]["missing_snapshot_bars"] == 1


def test_changed_sources_invalid_prices_and_a_wrong_daily_close_block_without_a_claim(registered):
    root, state_path, clients = registered
    shifted = clients.data.days.symbol.eq("SPY") & clients.data.days.session.str[5:7].isin(["02", "03"])
    clients.data.days.loc[shifted, "close"] *= 1.02  # about 8 percent of sessions: not the closing price
    clients.data.days.loc[shifted, "high"] *= 1.02
    _run(root, "acquire", clients)
    path = state_path.parent / "data/SPY_15Min.csv"
    original = path.read_text()
    path.write_text(original.replace("100000.0\n", "100001.0\n", 1))
    for mode in ("support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="source changed"):
            _run(root, mode, clients)
    path.write_text(original)
    assert _run(root, "support-audit", clients)["status"] == trial.SUPPORT_OK  # stamps alone look complete
    preflight = _run(root, "preflight", clients)
    assert preflight["status"] == "blocked_price_validated_support" and preflight["reasons"] == ["daily_close_mismatch:SPY"]
    assert preflight["daily_close_mismatch_sessions"]["SPY"] == int(shifted.sum())
    assert _run(root, "fit", clients)["status"] == "blocked_price_validated_support"
    assert json.loads(state_path.read_text())["trials"] == [] and not (state_path.parent / "report.json").exists()


def test_invalid_daily_bars_block_the_preflight(tmp_path):
    state_path, clients = _register(tmp_path)
    bad = clients.data.days.symbol.eq("QQQ") & clients.data.days.session.eq("2024-06-03")
    clients.data.days.loc[bad, "low"] = clients.data.days.loc[bad, "high"] * 2
    _run(tmp_path, "acquire", clients)
    preflight = _run(tmp_path, "preflight", clients)
    assert preflight["status"] == "blocked_price_validated_support" and "invalid_daily_bars" in preflight["reasons"]
    assert preflight["rejection_counts"]["invalid_daily_bars"] == 1
    assert json.loads(state_path.read_text())["trials"] == []


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


def test_the_rule_based_fallback_calendar_is_refused(monkeypatch):
    from ai_trading.utils import market_calendar

    monkeypatch.setattr(market_calendar, "_get_calendar", lambda: None)
    with pytest.raises(ValueError, match="exchange calendar unavailable"):
        trial.session_calendar(small_contract())
