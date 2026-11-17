from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import opening_range_trial as trial
from ai_trading.tools import swing_reversal_trial as swing

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/opening_range_trial_campaign.json"
CAMPAIGN = "opening_range_trial_test"
NAMES = [f"S{chr(65 + i)}" for i in range(8)]
EARLY_CLOSE = "2024-07-03"
SPLIT_AT = 60  # session index of SPLT's two-for-one split


def small_contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.pop("draft_notice", None)
    payload.update(
        campaign_id=CAMPAIGN, status="registered", development_start="2024-01-02", development_end="2024-07-31",
        acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
        signal=payload["signal"] | {"top_names": 3},
        support_thresholds={"minimum_eligible_days": 100, "minimum_supported_years": 1, "minimum_eligible_days_per_year": 100,
                            "minimum_filled_trades": 50, "minimum_median_selected_names": 2},
        acceptance=payload["acceptance"] | {"minimum_positive_years": 1},
        bootstrap={"unit": "circular_blocks_of_eligible_days", "block_days": 5, "replicates": 200, "seed": 7,
                   "lower_quantile": 0.0125})
    return payload | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


CALENDAR = trial.session_calendar(small_contract())
SESSIONS = CALENDAR.sessions


@pytest.fixture(autouse=True)
def _no_pauses(monkeypatch):
    pauses = []
    monkeypatch.setattr(trial, "sleep", pauses.append)
    monkeypatch.setattr(swing, "sleep", pauses.append)
    return pauses


# --- contract ---------------------------------------------------------------------------------------

def test_repository_contract_is_registered_with_the_papers_parameters_fixed(tmp_path):
    draft = json.loads(REPO_CONTRACT.read_text())
    assert draft["status"] == "registered" and "draft_notice" not in draft and draft["campaign_id"] == "opening_range_trial_20261117"
    reset = json.loads((REPO_CONTRACT.parents[1] / trial.RESET_CONFIG).read_text())
    assert draft["campaign_id"] in reset["registered_trials"]
    assert any(a["campaign_id"] == draft["campaign_id"] and (REPO_CONTRACT.parents[1] / a["record"]).is_file() for a in reset["amendments"])
    assert (draft["holdout_start"], draft["holdout_end"]) == (reset["holdout_start"], reset["holdout_end"]) == ("2026-09-09", "2026-12-08")
    assert draft["promotion_authority"] is False and draft["runtime_authority"] is False and draft["max_trials"] == 1
    assert "2026-11-17" in draft["authorization"] and draft["owner_decisions"]["date"] == "2026-11-17"
    full = trial.contract_spec(draft)
    assert (full.min_open_price, full.min_avg_volume, full.min_atr, full.lookback) == (5.0, 1_000_000.0, 0.5, 14)
    assert (full.min_rel_volume, full.top_names, full.stop_atr_fraction) == (1.0, 20, 0.1)
    assert (full.risk_fraction, full.max_leverage, full.commission) == (0.01, 4.0, 0.0035)
    assert (full.stale_minutes, full.stale_adverse_atr, full.ssr_decline, full.first_index) == (30, 1.0, 0.1, 15)
    assert [(n, c) for n, c, _ in full.tiers] == [("base", 5.0), ("stress", 7.5), ("double", 10.0), ("half", 2.5),
                                                  ("commission_only_as_published", 0.0)]
    assert full.tier("decision")[0] == "base" and full.tier(trial.STRESS_ROLE)[0] == "stress"
    assert [r for _, _, r in full.tiers].count("reported_only") == 3
    assert trial.FILL_RULES["deciding_model"] == trial.DECIDING_MODEL == "bar_executable"
    assert set(draft["fills"]["models"]) == set(trial.FILL_MODELS) and draft["fills"]["reported_only_models"] == list(trial.FILL_MODELS[1:])
    assert not any("conservative" in key for key in draft["acceptance"]) and "required_positive_model" not in draft["fills"]
    assert "permits nothing" in draft["versions"]["long_short"]
    assert (full.min_days, full.min_years, full.min_per_year, full.min_trades, full.min_selected) == (2300, 9, 230, 5000, 10)
    assert (full.block, full.replicates, full.seed, full.lower_quantile, full.min_positive_years) == (21, 10000, 20261117, 0.0125, 6)
    assert (draft["development_start"], draft["development_end"], draft["adjustment"]) == ("2016-01-04", "2025-12-31", "raw")
    assert list(draft["versions"]) == ["long_only", "long_short"] and {k: draft["fills"][k] for k in trial.FILL_RULES} == trial.FILL_RULES
    assert draft["survivorship"]["label"] == trial.PROBE_STATUSES[draft["survivorship"]["probe_status"]]
    assert "PENDING" not in json.dumps(draft) and len(draft["limitations"]) >= 6 and len(draft["departures_from_the_paper"]) >= 8
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft | {"status": "draft_not_registered"}))
    for mode in ("acquire", "support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="not registered"):
            trial.run(tmp_path, mode=mode, contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "all"}, "raw SIP"),
        ({"timeframes": {"opening_range": "15Min", "path": "1Min", "filters": "1Day"}}, "raw SIP"),
        ({"runtime_authority": True}, "authority"),
        ({"promotion_authority": True}, "authority"),
        ({"versions": {"long_short": "", "long_only": ""}}, "versions"),
        ({"cost_tiers": [{"name": "base", "per_side_bps": 5, "role": "decision"}]}, "stress tier"),
        ({"cost_tiers": [{"name": "base", "per_side_bps": 5, "role": "decision"},
                         {"name": "stress", "per_side_bps": 5, "role": "required_positive_at_stress"}]}, "cost more"),
        ({"primary_per_side_cost_bps": 2}, "primary_per_side_cost_bps"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))


@pytest.mark.parametrize(
    ("section", "change", "match"),
    [
        ("signal", {"opening_range_minutes": 15}, "first 5-minute bar"),
        ("signal", {"doji": "buy"}, "doji"),
        ("signal", {"top_names": 0}, "top_names"),
        ("trade", {"exit": "next_open"}, "session close"),
        ("trade", {"early_close_sessions": "skip"}, "early closes"),
        ("trade", {"stop_atr_fraction": 0}, "stop_atr_fraction"),
        ("fills", {"deciding_model": "paper_exact"}, "fill models"),
        ("fills", {"deciding_model": "lower_bound"}, "fill models"),
        ("fills", {"reported_only_models": ["paper_exact", "lower_bound"]}, "fill models"),
        ("fills", {"commission_per_share": -1}, "commission_per_share"),
        ("sizing", {"max_leverage": 8}, "max_leverage"),
        ("sizing", {"risk_fraction_of_slot": True}, "risk_fraction_of_slot"),
        ("shorts", {"require_current_shortable_and_easy_to_borrow": False}, "shortable"),
        ("survivorship", {"probe_status": "not_run"}, "probe_status"),
        ("acceptance", {"positive_at_stress": False}, "disabled"),
        ("acceptance", {"long_only_positive_lower_bound_vs_exposure_matched_market_at_primary": False}, "disabled"),
        ("bootstrap", {"replicates": 150}, "multiple of 100"),
        ("universe", {"minimum_open_price": "5"}, "minimum_open_price"),
    ],
)
def test_every_rule_section_is_checked(section, change, match):
    contract = small_contract()
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(contract | {section: contract[section] | change})


# --- calendar ---------------------------------------------------------------------------------------

def test_calendar_has_early_closes_and_the_fallback_calendar_is_refused(monkeypatch):
    closes = dict(zip(SESSIONS, CALENDAR.closes))
    assert closes[EARLY_CLOSE] == "13:00" and closes["2024-07-02"] == "16:00"
    assert CALENDAR.regular_minutes(SESSIONS.index(EARLY_CLOSE)) == 210 and CALENDAR.regular_minutes(0) == 390
    assert "2024-01-15" not in closes and "2024-06-19" not in closes  # holidays
    from ai_trading.utils import market_calendar

    monkeypatch.setattr(market_calendar, "_get_calendar", lambda: None)
    with pytest.raises(ValueError, match="exchange calendar unavailable"):
        trial.session_calendar(small_contract())


def test_a_session_needs_history_and_a_complete_market_record():
    full = Counter({session: CALENDAR.regular_minutes(i) for i, session in enumerate(SESSIONS)})
    statuses = trial.session_statuses(CALENDAR, full, spec())
    assert statuses[:15] == ["insufficient_history"] * 15 and set(statuses[15:]) == {trial.ELIGIBLE}
    assert statuses[SESSIONS.index(EARLY_CLOSE)] == trial.ELIGIBLE  # 210 minutes is complete for an early close
    gappy = full.copy()
    gappy[SESSIONS[40]] -= 1
    gappy[EARLY_CLOSE] = 390  # bars after 13:00 would be a wrong count, not a pass
    statuses = trial.session_statuses(CALENDAR, gappy, spec())
    assert statuses[40] == "market_incomplete" and statuses[SESSIONS.index(EARLY_CLOSE)] == "market_incomplete"
    selected = Counter({session: 3 for session in SESSIONS})
    summary = trial.support_summary(spec(), CALENDAR, statuses, selected, selected)
    assert summary["skipped_sessions"] == {"insufficient_history": 15, "market_incomplete": 2}
    assert summary["eligible_days"] == len(SESSIONS) - 17 and summary["status"] == trial.SUPPORT_OK
    thin = trial.support_summary(spec(), CALENDAR, statuses, Counter({session: 1 for session in SESSIONS}), selected)
    assert thin["status"] == "blocked_sampling_support" and thin["blocking_reasons"] == ["selected_names_below_minimum"]
    short = small_contract()
    short["support_thresholds"] = short["support_thresholds"] | {"minimum_eligible_days": 500, "minimum_eligible_days_per_year": 200}
    blocked = trial.support_summary(trial.contract_spec(short), CALENDAR, statuses, selected, selected)
    assert blocked["blocking_reasons"] == ["eligible_days_below_minimum", "supported_years_below_minimum"]


# --- daily filters: price, volume, range; prior sessions only ---------------------------------------

def daily_bars(symbol="AAA", *, price=50.0, band=0.02, volume=2e6, n=40, seed=1):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[price], close[:-1]]) * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame(dict(session=SESSIONS[:n], symbol=symbol, open=open_, high=np.maximum(open_, close) * (1 + band),
                             low=np.minimum(open_, close) * (1 - band), close=close, volume=volume))


def test_daily_filters_match_a_hand_computation_and_use_only_prior_sessions():
    bars = daily_bars()
    features = trial.daily_features(bars, SESSIONS, [], spec()).set_index("idx")
    assert features.index.tolist() == list(range(15, 40))  # the first 15 sessions only supply history
    t = 30
    window = bars.iloc[t - 14:t]
    before = bars.close.shift(1).iloc[t - 14:t]
    true_range = np.maximum(window.high - window.low, np.maximum((window.high - before).abs(), (window.low - before).abs()))
    assert features.at[t, "atr"] == pytest.approx(true_range.mean()) and features.at[t, "avg_volume"] == pytest.approx(2e6)
    assert features.at[t, "open"] == bars.open.iat[t] and features.at[t, "prior_close"] == bars.close.iat[t - 1]

    # Everything on session t except its open, and everything after t, is rewritten: the row for t does not move.
    later = bars.copy()
    later.loc[later.index >= t, ["high", "low", "close"]] *= 3.0
    later.loc[later.index >= t, "volume"] *= 50
    later.loc[later.index > t, "open"] *= 3.0
    changed = trial.daily_features(later, SESSIONS, [], spec()).set_index("idx")
    pd.testing.assert_series_equal(changed.loc[t], features.loc[t])
    pd.testing.assert_frame_equal(changed.loc[:t], features.loc[:t])

    # Each filter can fail on its own, and each uses the paper's comparison.
    cheap = bars.copy()
    cheap.loc[t, "open"] = 5.0  # "above $5"
    assert t not in trial.daily_features(cheap, SESSIONS, [], spec()).idx.tolist()
    assert trial.daily_features(daily_bars(volume=999_999), SESSIONS, [], spec()).empty
    assert len(trial.daily_features(daily_bars(volume=1_000_000), SESSIONS, [], spec())) == 25  # "at least 1,000,000"
    quiet = daily_bars(price=6.0, band=0.001)
    assert trial.daily_features(quiet, SESSIONS, [], spec()).empty  # range of a few cents: ATR under $0.50


def test_a_missing_daily_bar_or_a_recent_split_makes_a_name_ineligible():
    bars = daily_bars()
    gap = trial.daily_features(bars.drop(index=20), SESSIONS, [], spec()).idx.tolist()
    # Session 20 has no open; sessions 21 to 35 have session 20 inside the 15 bars they need.
    assert gap == [15, 16, 17, 18, 19, 36, 37, 38, 39]
    split = trial.daily_features(bars, SESSIONS, [("AAA", SESSIONS[22]), ("OTHER", SESSIONS[30])], spec()).idx.tolist()
    assert split == [15, 16, 17, 18, 19, 20, 21, 37, 38, 39]  # the split session and the 14 after it


def test_split_events_are_the_sessions_where_the_adjustment_ratio_moves():
    raw = daily_bars(n=30)
    raw.loc[raw.index < 12, ["open", "high", "low", "close"]] *= 2.0  # two-for-one effective on session 12
    adjusted = raw.copy()
    adjusted.loc[adjusted.index < 12, ["open", "high", "low", "close"]] /= 2.0
    other = daily_bars("BBB", n=30, seed=2)
    events = trial.split_events(pd.concat([raw, other]), pd.concat([adjusted, other]))
    assert events == [("AAA", SESSIONS[12])]


def test_short_sale_restriction_carries_from_a_ten_percent_fall_the_day_before():
    bars = daily_bars()
    bars.loc[25, "low"] = bars.close.iat[24] * 0.89
    features = trial.daily_features(bars, SESSIONS, [], spec()).set_index("idx")
    assert features.ssr_carry.to_dict() == {i: i == 26 for i in range(15, 40)}


# --- relative volume ranking: nothing after 09:35, nothing from later sessions ----------------------

def opening_bars(symbols, *, n=40, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in symbols:
        for i in range(n):
            open_ = 50.0 + i * 0.1
            close = open_ * (1 + 0.002 * rng.choice([-1, 1]))
            rows.append(dict(session=SESSIONS[i], symbol=symbol, open=open_, high=max(open_, close) + 0.05, low=min(open_, close) - 0.05,
                             close=close, volume=float(rng.integers(50_000, 150_000))))
    return pd.DataFrame(rows)


def feature_rows(symbols, sessions_idx, *, atr=2.0, prior_close=50.0):
    return pd.DataFrame([dict(idx=i, symbol=s, open=50.0, atr=atr, avg_volume=2e6, prior_close=prior_close, ssr_carry=False)
                         for s in symbols for i in sessions_idx])


def test_relative_volume_is_todays_first_bar_over_the_mean_of_the_prior_fourteen():
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    opening = opening_bars(symbols)
    t = 30
    opening.loc[(opening.symbol == "CCC") & (opening.session == SESSIONS[t]), "volume"] = 900_000.0
    opening.loc[(opening.symbol == "EEE") & (opening.session == SESSIONS[t]), "volume"] = 10.0  # far below its average
    selection = trial.build_selection(feature_rows(symbols, range(15, 40)), opening, SESSIONS, frozenset(), spec())
    day = selection[selection.session == SESSIONS[t]].reset_index(drop=True)
    volume = opening.pivot(index="session", columns="symbol", values="volume")
    expected = (volume.iloc[t] / volume.iloc[t - 14:t].mean()).sort_values(ascending=False)
    expected = expected[expected >= 1.0].head(3)
    assert day.symbol.tolist() == expected.index.tolist() and day.symbol.iat[0] == "CCC" and "EEE" not in day.symbol.tolist()
    assert day["rank"].tolist() == list(range(1, len(day) + 1)) and len(day) <= 3
    assert day.rel_volume.tolist() == pytest.approx(expected.tolist())
    assert (selection.groupby("session").size() <= 3).all() and (selection.rel_volume >= 1.0).all()
    assert selection.session.min() >= SESSIONS[15]

    # Every later session is rewritten, and so is today's bar apart from its volume: the ranking for t does not move.
    later = opening.copy()
    later.loc[later.session > SESSIONS[t], "volume"] *= 40
    later.loc[later.session > SESSIONS[t], ["open", "high", "low", "close"]] *= 2
    again = trial.build_selection(feature_rows(symbols, range(15, 40)), later, SESSIONS, frozenset(), spec())
    keep = ["session", "symbol", "rank", "rel_volume", "direction", "trigger"]
    pd.testing.assert_frame_equal(again[again.session <= SESSIONS[t]][keep].reset_index(drop=True),
                                  selection[selection.session <= SESSIONS[t]][keep].reset_index(drop=True))

    # A name missing one of its 14 prior opening bars has no relative volume and is not ranked.
    holed = opening[~((opening.symbol == "CCC") & (opening.session == SESSIONS[t - 5]))]
    without = trial.build_selection(feature_rows(symbols, range(15, 40)), holed, SESSIONS, frozenset(), spec())
    assert "CCC" not in without[without.session == SESSIONS[t]].symbol.tolist()
    # A name that fails the daily filters that session is not ranked however high its volume.
    filtered = feature_rows(symbols, range(15, 40))
    filtered = filtered[~((filtered.symbol == "CCC") & (filtered.idx == t))]
    unfiltered = trial.build_selection(filtered, opening, SESSIONS, frozenset(), spec())
    assert "CCC" not in unfiltered[unfiltered.session == SESSIONS[t]].symbol.tolist()


def test_ties_break_by_symbol_and_the_threshold_is_at_least_one():
    symbols = ["DDD", "AAA", "CCC", "BBB"]
    opening = pd.DataFrame([dict(session=SESSIONS[i], symbol=s, open=50.0, high=50.3, low=49.9, close=50.2, volume=100_000.0)
                            for s in symbols for i in range(20)])
    selection = trial.build_selection(feature_rows(symbols, range(15, 20)), opening, SESSIONS, frozenset(), spec())
    assert selection[selection.session == SESSIONS[17]].symbol.tolist() == ["AAA", "BBB", "CCC"]  # all exactly 1.0
    assert selection.rel_volume.eq(1.0).all()


def test_direction_trigger_doji_and_the_short_rule():
    symbols = ["DOJI", "DOWN", "FELL", "HTB", "UP"]
    opening = pd.DataFrame([dict(session=SESSIONS[i], symbol=s, open=50.0, high=50.4, low=49.7, close=50.0, volume=100_000.0 + i)
                            for s in symbols for i in range(20)])
    t = 17
    today = opening.session == SESSIONS[t]
    opening.loc[today & (opening.symbol == "UP"), "close"] = 50.2
    opening.loc[today & opening.symbol.isin(["DOWN", "FELL", "HTB"]), "close"] = 49.8
    features = feature_rows(symbols, [t])
    features.loc[features.symbol == "FELL", "prior_close"] = 56.0  # the sell trigger, 49.7, is more than 10% below
    contract = small_contract()
    wide = trial.contract_spec(contract | {"signal": contract["signal"] | {"top_names": 5}})
    day = trial.build_selection(features, opening, SESSIONS, frozenset({"HTB"}), wide).set_index("symbol")
    assert day.direction.to_dict() == {"DOJI": 0, "DOWN": -1, "FELL": -1, "HTB": -1, "UP": 1}
    assert day.at["UP", "trigger"] == 50.4 and day.at["DOWN", "trigger"] == 49.7 and np.isnan(day.at["DOJI", "trigger"])
    assert day.stop_distance.eq(0.2).all()  # 10 percent of an ATR of 2.0
    assert day.short_ok.to_dict() == {"DOJI": False, "DOWN": True, "FELL": False, "HTB": False, "UP": False}
    carried = features.assign(ssr_carry=True)
    assert not trial.build_selection(carried, opening, SESSIONS, frozenset(), wide).short_ok.any()
    assert set(day.index) == set(symbols)  # the doji keeps its rank and simply places no order


# --- one order: entry, stop and exit ----------------------------------------------------------------

def sim(bars, *, direction=1, trigger=100.0, stop=0.5, start="09:35", close="16:00", stale=30, adverse=5.0, model="lower_bound"):
    times = [trial._stamp(trial._minute(start) + i) for i in range(len(bars))]
    columns = [np.array(column, dtype=float) for column in zip(*bars)]
    return trial.simulate_trade(times, *columns, direction=direction, trigger=trigger, stop_distance=stop, close_stamp=close,
                                stale_minutes=stale, stale_adverse=adverse, model=model)


def price(trade):
    return trade["entry_time"], trade["entry_price"], trade["exit_time"], trade["exit_price"], trade["exit_reason"]


# The deciding model: enter at the next bar's open, stop measured from that fill, stop-outs at the stop price.

def test_the_deciding_model_enters_at_the_next_bars_open_whatever_it_is():
    assert trial.DECIDING_MODEL == "bar_executable"
    up = sim([QUIET, (99.7, 100.2, 99.6, 100.1), (100.3, 100.6, 100.2, 100.5), (100.5, 100.9, 100.4, 100.8)], close="09:39", model="bar_executable")
    assert price(up) == ("09:37", 100.3, "09:38", 100.8, "session_end") and up["entry_at_open"] is True
    down = sim([QUIET, (99.7, 100.2, 99.6, 99.9), (99.8, 99.95, 99.7, 99.85)], close="09:38", model="bar_executable")
    assert price(down) == ("09:37", 99.8, "09:37", 99.85, "session_end")  # the next open is below the trigger: that is still the fill
    # The trigger bar's own low is before the fill and cannot stop the trade; a gap through the trigger is not filled at the gap.
    wide = sim([(99.9, 100.6, 99.0, 100.5), (100.9, 103.0, 100.8, 103.0)], close="09:37", model="bar_executable")
    assert price(wide) == ("09:36", 100.9, "09:36", 103.0, "session_end")
    gap = sim([(100.4, 100.7, 100.1, 100.6), (100.6, 100.8, 100.5, 100.7)], close="09:37", model="bar_executable")
    assert price(gap) == ("09:36", 100.6, "09:36", 100.7, "session_end") and gap["gap_entry"] is True
    # A trigger on the session's last bar has no next bar: no fill, in this model and in the conservative one.
    assert sim([QUIET, (99.7, 100.3, 99.6, 100.2)], close="09:37", model="bar_executable") is None
    assert sim([QUIET, (99.7, 100.3, 99.6, 100.2)], close="09:37", model="conservative") is None
    assert sim([QUIET, QUIET], close="09:37", model="bar_executable") is None


def test_the_deciding_model_measures_the_stop_from_the_fill_and_exits_at_the_stop_or_the_gapped_open():
    start = [(99.7, 100.2, 99.6, 99.9), (100.3, 100.4, 100.2, 100.3)]  # filled at 100.3, so the stop is 99.8, not 99.5
    held = sim([*start, (100.3, 100.4, 99.8, 99.9), (99.0, 99.1, 98.9, 99.0)], close="09:39", model="bar_executable")
    assert price(held) == ("09:36", 100.3, "09:37", 99.8, "stop")  # the stop price, though the next bar opens far lower
    careful = sim([*start, (100.3, 100.4, 99.8, 99.9), (99.0, 99.1, 98.9, 99.0)], close="09:39", model="conservative")
    assert price(careful) == ("09:36", 100.3, "09:37", 99.0, "stop")  # the conservative model takes the worse next open
    kept = sim([*start, (100.3, 100.4, 99.8, 99.9), (100.1, 100.2, 99.9, 100.0)], close="09:39", model="conservative")
    assert kept["exit_price"] == 99.8  # ... but never better than the stop
    above = sim([*start, (100.3, 100.4, 99.81, 99.9)], close="09:38", model="bar_executable")
    assert price(above)[3:] == (99.9, "session_end")  # one cent above the stop is not a touch
    gapped = sim([*start, (99.1, 99.5, 99.0, 99.4), (99.4, 99.9, 99.3, 99.8)], close="09:39", model="bar_executable")
    assert price(gapped) == ("09:36", 100.3, "09:37", 99.1, "stop_gap")  # opened through the stop: the open, not the stop
    same_bar = sim([(99.7, 100.2, 99.6, 99.9), (100.3, 100.4, 99.7, 100.0), (100.0, 101.0, 99.9, 101.0)], close="09:38", model="bar_executable")
    assert price(same_bar) == ("09:36", 100.3, "09:36", 99.8, "stop")  # entered at the bar's open, so its low came after the fill


def test_the_deciding_model_uses_the_next_bar_that_exists_when_minutes_are_missing():
    # No trade printed from 09:37 to 09:40 and from 09:43 to 09:49: a missing minute is not a bar.
    times = ["09:35", "09:36", "09:41", "09:42", "09:50", "09:51"]
    bars = [QUIET, (99.7, 100.2, 99.6, 100.1), (100.6, 100.9, 100.5, 100.8), (100.8, 100.9, 100.3, 100.4), (99.0, 99.2, 98.9, 99.1),
            (99.1, 99.3, 99.0, 99.2)]
    columns = [np.array(column, dtype=float) for column in zip(*bars)]
    kwargs = dict(direction=1, trigger=100.0, stop_distance=0.5, close_stamp="09:52", stale_minutes=30, stale_adverse=5.0)
    trade = trial.simulate_trade(times, *columns, model="bar_executable", **kwargs)
    assert price(trade) == ("09:41", 100.6, "09:50", 99.0, "stop_gap")  # filled five minutes after the trigger; stop 100.1
    careful = trial.simulate_trade(times, *columns, model="conservative", **kwargs)
    assert price(careful) == price(trade)
    touched = [*bars[:3], (100.8, 100.9, 100.0, 100.4), *bars[4:]]
    columns = [np.array(column, dtype=float) for column in zip(*touched)]
    assert price(trial.simulate_trade(times, *columns, model="bar_executable", **kwargs)) == ("09:41", 100.6, "09:42", 100.1, "stop")
    assert price(trial.simulate_trade(times, *columns, model="conservative", **kwargs)) == ("09:41", 100.6, "09:42", 99.0, "stop")
    # The record ending early is the stale rule's business: last bar 09:36 with a 16:00 close.
    assert trial.simulate_trade(times[:2], *(column[:2] for column in columns), model="bar_executable", **(kwargs | {"close_stamp": "16:00"})) is None
    stale = trial.simulate_trade(times[:3], *(column[:3] for column in columns), model="bar_executable", **(kwargs | {"close_stamp": "16:00"}))
    assert price(stale) == ("09:41", 100.6, "09:41", pytest.approx(95.8), "stale_last_print")


def test_the_papers_fills_are_exactly_the_trigger_and_the_stop():
    bars = [(100.4, 100.7, 99.0, 100.6), (100.6, 100.8, 100.5, 100.7), (99.0, 99.2, 98.0, 98.5), (98.5, 105.0, 98.4, 105.0)]
    paper = sim(bars, close="09:39", model="paper_exact")
    assert price(paper) == ("09:35", 100.0, "09:37", 99.5, "stop") and paper["entry_at_open"] is False  # gaps ignored on both sides
    kept = sim(bars[:2], close="09:37", model="paper_exact")
    assert price(kept)[3:] == (100.7, "session_end")  # the trigger bar's low of 99.0 does not stop it
    last = sim([QUIET, (99.7, 100.3, 99.6, 100.2)], close="09:37", model="paper_exact")
    assert price(last) == ("09:36", 100.0, "09:36", 100.2, "session_end")
    with pytest.raises(ValueError, match="unknown fill model"):
        sim(bars, model="optimistic")


def test_every_model_mirrors_for_a_sell_and_the_models_are_ordered():
    rng = np.random.default_rng(5)
    seen = Counter()
    for _ in range(300):
        count = int(rng.integers(3, 40))
        close = 99.6 + np.cumsum(rng.normal(0.02, 0.25, count))
        open_ = np.concatenate([[99.6], close[:-1]]) + rng.normal(0, 0.1, count)
        high, low = np.maximum(open_, close) + rng.uniform(0, 0.2, count), np.minimum(open_, close) - rng.uniform(0, 0.2, count)
        bars = list(zip(open_, high, low, close))
        result = {}
        for model in trial.FILL_MODELS:
            end = trial._stamp(575 + count)
            buy, sell = sim(bars, close=end, model=model), sim(mirror(bars), direction=-1, close=end, model=model)
            assert (buy is None) == (sell is None)
            if buy is None:
                continue
            assert price(sell)[::2] + (price(sell)[4],) == price(buy)[::2] + (price(buy)[4],)
            assert sell["entry_price"] == pytest.approx(200 - buy["entry_price"]) and sell["exit_price"] == pytest.approx(200 - buy["exit_price"])
            assert buy["entry_time"] >= "09:35" and buy["entry_time"] <= buy["exit_time"]
            result[model] = buy["exit_price"] - buy["entry_price"]
            seen[model, buy["exit_reason"]] += 1
        if "bar_executable" in result:
            assert result["conservative"] <= result["bar_executable"] + 1e-12  # same entry, exits never better
        if len(result) == 4:
            assert result["lower_bound"] <= result["paper_exact"] + 1e-9
    assert {reason for model, reason in seen if model == "bar_executable"} == {"session_end", "stop", "stop_gap"}


def driftless_days(n_days, *, seed, k_vol=1.0, sub=30, minutes=390):
    """A walk with no edge: volatile at the open, direction-aligned to its first five minutes, in units of the daily range."""
    rng = np.random.default_rng(seed)
    t = np.arange(minutes)
    profile = 1 + 2.0 * np.exp(-t / 15) + 0.5 * np.exp(-(minutes - 1 - t) / 20)
    share = profile ** 2 / (profile ** 2).sum()
    step = np.repeat(0.55 * k_vol * np.sqrt(share / sub), sub)
    for _ in range(n_days):
        path = np.cumsum(rng.normal(0.0, 1.0, minutes * sub) * step)
        direction = np.sign(path[5 * sub - 1] - path[0])
        if direction == 0:
            continue
        path = 50.0 + direction * path
        bars = path[5 * sub:].reshape(minutes - 5, sub)
        yield float(path[:5 * sub].max()), bars[:, 0], bars.max(axis=1), bars.min(axis=1), bars[:, -1]


def test_zero_drift_calibration_the_deciding_model_scores_no_edge_as_zero_and_the_lower_bound_as_a_loss():
    """A fill model that turns a process with no edge into a loss decides the trial by itself. This must stay near zero.

    Measured with this seed, in planned losses per fill before costs (standard error 0.03): bar_executable +0.04,
    conservative -0.06, paper_exact +0.10, lower_bound -0.21. The deciding model's small positive figure is the
    two-second steps of this walk jumping past the stop while the model sells at the stop price; other seeds give
    +0.04 to +0.05. The tolerance is 0.10, about three standard errors.
    """
    times = [trial._stamp(575 + m) for m in range(385)]
    results = {model: [] for model in trial.FILL_MODELS}
    for trigger, *columns in driftless_days(8000, seed=20261117):
        for model in trial.FILL_MODELS:
            trade = trial.simulate_trade(times, *columns, direction=1, trigger=trigger, stop_distance=0.1, close_stamp="16:00",
                                         stale_minutes=30, stale_adverse=1.0, model=model)
            if trade is not None:
                results[model].append((trade["exit_price"] - trade["entry_price"]) / 0.1)
    mean = {model: float(np.mean(values)) for model, values in results.items()}
    error = float(np.std(results["bar_executable"]) / np.sqrt(len(results["bar_executable"])))
    assert len(results["bar_executable"]) > 7000 and error < 0.035
    assert abs(mean["bar_executable"]) < 0.10  # in planned losses per fill, before costs
    assert mean["lower_bound"] < -0.15 and mean["lower_bound"] < mean["bar_executable"] - 0.15
    assert mean["lower_bound"] < mean["conservative"] < mean["bar_executable"] < mean["paper_exact"]


# The lower-bound model (reported only): the harsh rules of the first draft.


def mirror(bars, pivot=200.0):
    return [(pivot - o, pivot - low, pivot - h, pivot - c) for o, h, low, c in bars]


QUIET = (99.0, 99.5, 98.8, 99.2)


def test_an_order_that_never_reaches_its_trigger_is_no_trade():
    assert sim([QUIET, (99.2, 99.99, 99.0, 99.9)], close="09:37") is None
    assert sim([QUIET, (99.2, 150.0, 99.0, 120.0)], direction=-1, trigger=98.0, close="09:37") is None  # a sell never triggers on a rise
    assert sim(mirror([QUIET, (99.2, 99.99, 99.0, 99.9)]), direction=-1, close="09:37") is None


def test_entry_is_the_worse_of_the_trigger_and_the_next_bars_open():
    ran = sim([QUIET, (99.7, 100.2, 99.6, 100.1), (100.3, 100.6, 100.2, 100.5), (100.5, 100.9, 100.4, 100.8)], close="09:39")
    assert (ran["entry_time"], ran["entry_price"], ran["gap_entry"]) == ("09:36", 100.3, False)  # the price ran: pay the next open
    assert (ran["exit_time"], ran["exit_price"], ran["exit_reason"]) == ("09:38", 100.8, "session_end")
    back = sim([QUIET, (99.7, 100.2, 99.6, 99.9), (99.8, 99.95, 99.7, 99.85)], close="09:38")
    assert (back["entry_price"], back["exit_price"], back["exit_reason"]) == (100.0, 99.85, "session_end")  # fell back: still pay the trigger
    touch = sim([(99.7, 100.0, 99.6, 99.9), (99.9, 99.95, 99.8, 99.9)], close="09:37")
    assert touch["entry_price"] == 100.0  # a high exactly at the trigger fills


def test_a_gap_through_the_trigger_fills_at_the_gapped_open():
    trade = sim([(100.4, 100.7, 100.1, 100.6), (100.6, 100.8, 100.5, 100.7)], close="09:37")
    assert (trade["entry_time"], trade["entry_price"], trade["gap_entry"]) == ("09:35", 100.4, True)
    assert (trade["exit_price"], trade["exit_reason"]) == (100.7, "session_end")
    stopped = sim([(100.4, 100.7, 99.9, 100.6), (100.6, 100.8, 100.5, 100.7)], close="09:37")
    assert (stopped["exit_reason"], stopped["exit_price"]) == ("stop_in_entry_bar", 99.9)  # 100.4 - 0.5, touched in the same bar


def test_a_bar_that_touches_entry_and_stop_is_a_stop_out():
    # Trigger 100.0, stop distance 0.5; the trigger bar's low is 99.4, so the stop is assumed hit after entry.
    kind = sim([(99.9, 100.1, 99.4, 99.6), (99.7, 99.9, 99.6, 99.8), (99.8, 105.0, 99.8, 105.0)], close="09:38")
    assert (kind["entry_price"], kind["exit_time"], kind["exit_price"], kind["exit_reason"]) == (100.0, "09:35", 99.5, "stop_in_entry_bar")
    worse = sim([(99.9, 100.1, 99.4, 99.6), (99.2, 99.9, 99.1, 99.8)], close="09:37")
    assert (worse["entry_price"], worse["exit_price"]) == (100.0, 99.2)  # the next open is below the stop: that is the exit
    # Even when the price then ran away upward, a trigger bar that reached the stop distance below the trigger is a full loss.
    ran = sim([(99.9, 100.6, 99.4, 100.5), (100.9, 103.0, 100.8, 103.0)], close="09:37")
    assert (ran["entry_price"], ran["exit_price"], ran["exit_reason"]) == (100.0, 99.5, "stop_in_entry_bar")
    # One cent less and the bar did not touch the stop: the entry is the dearer next open and the stop moves up with it.
    alive = sim([(99.9, 100.6, 99.51, 100.5), (100.9, 103.0, 100.8, 103.0), (103.0, 103.1, 100.4, 100.5)], close="09:38")
    assert (alive["entry_price"], alive["exit_time"], alive["exit_price"], alive["exit_reason"]) == (100.9, "09:37", 100.4, "stop")


def test_stop_loss_exits_at_the_worse_of_the_stop_and_the_next_open_and_gaps_fill_at_the_open():
    start = [(99.7, 100.2, 99.6, 99.9), (99.9, 100.4, 99.8, 100.3)]  # entry 100.0, stop 99.5
    held = sim([*start, (100.3, 100.4, 99.5, 99.7), (99.8, 99.9, 99.7, 99.8)], close="09:39")
    assert (held["exit_time"], held["exit_price"], held["exit_reason"]) == ("09:37", 99.5, "stop")  # low exactly at the stop
    slipped = sim([*start, (100.3, 100.4, 99.3, 99.4), (99.1, 99.2, 99.0, 99.1)], close="09:39")
    assert (slipped["exit_price"], slipped["exit_reason"]) == (99.1, "stop")
    gapped = sim([*start, (98.0, 98.5, 97.9, 98.4), (98.4, 99.9, 98.3, 99.8)], close="09:39")
    assert (gapped["exit_time"], gapped["exit_price"], gapped["exit_reason"]) == ("09:37", 98.0, "stop_gap")
    last = sim([*start, (100.3, 100.4, 99.3, 99.4)], close="09:38")
    assert (last["exit_price"], last["exit_reason"]) == (99.4, "stop")  # no next bar: the worse of the stop and that bar's close
    survived = sim([*start, (100.3, 100.4, 99.51, 99.6)], close="09:38")
    assert (survived["exit_price"], survived["exit_reason"]) == (99.6, "session_end")  # one cent above the stop is not a touch


def test_a_trigger_on_the_last_bar_cannot_profit_and_a_stale_record_is_marked_against_the_position():
    last = sim([QUIET, (99.7, 100.3, 99.6, 100.2)], close="09:37")
    assert (last["entry_price"], last["exit_price"], last["exit_reason"]) == (100.2, 100.2, "session_end")
    bars = [(99.7, 100.2, 99.6, 99.9), (99.9, 101.0, 99.8, 100.9)]
    fresh = sim(bars, start="15:28")  # last bar 15:29, 30 minutes before the final 15:59 bar
    assert (fresh["exit_price"], fresh["exit_reason"]) == (100.9, "session_end")
    stale = sim(bars, start="15:27")  # last bar 15:28: the record ends more than 30 minutes early
    assert (stale["exit_price"], stale["exit_reason"]) == (pytest.approx(95.9), "stale_last_print")
    early = sim(bars, start="12:40", close="13:00")
    assert early["exit_reason"] == "session_end"  # measured against the early close, not 16:00


def test_a_stale_exit_never_goes_below_one_cent_so_scoring_cannot_crash_after_the_claim():
    cheap = [(5.5, 5.7, 5.45, 5.6), (5.6, 5.8, 5.55, 5.7), (5.7, 5.8, 5.6, 5.7)]
    for model in trial.FILL_MODELS:
        trade = sim(cheap, trigger=5.65, stop=0.6, adverse=6.0, model=model)  # last print 5.70 less an ATR of 6.00 is below zero
        assert (trade["exit_reason"], trade["exit_price"]) == ("stale_last_print", 0.01)
        short = sim(mirror(cheap, 11.0), direction=-1, trigger=5.35, stop=0.6, adverse=6.0, model=model)
        assert short["exit_reason"] == "stale_last_print" and short["exit_price"] == pytest.approx(5.3 + 6.0)
    full = trial.contract_spec(json.loads(REPO_CONTRACT.read_text()))
    calendar = trial.SessionCalendar(dates=(date(2024, 1, 5),), closes=("16:00",))
    trial.validate_trades(trade_rows([SESSIONS[3]]).assign(exit_price=0.01, exit_reason="stale_last_print"), calendar, full)
    # The floor belongs to the stale rule alone: an ordinary exit is never moved (a synthetic walk may sit below zero).
    below = sim([(-1.0, -0.5, -1.1, -0.6), (-0.6, -0.55, -0.7, -0.65), (-0.65, -0.6, -2.0, -1.9)], trigger=-0.5, close="09:38",
                model="bar_executable")
    assert price(below) == ("09:36", -0.6, "09:37", -1.1, "stop")


def test_a_sell_order_is_the_mirror_image_of_a_buy_order():
    cases = [
        [QUIET, (99.7, 100.2, 99.6, 100.1), (100.3, 100.6, 100.2, 100.5), (100.5, 100.9, 100.4, 100.8)],
        [(100.4, 100.7, 100.1, 100.6), (100.6, 100.8, 100.5, 100.7)],
        [(99.9, 100.1, 99.4, 99.6), (99.2, 99.9, 99.1, 99.8)],
        [(99.9, 100.6, 99.51, 100.5), (100.9, 103.0, 100.8, 103.0), (103.0, 103.1, 100.4, 100.5)],
        [(99.7, 100.2, 99.6, 99.9), (99.9, 100.4, 99.8, 100.3), (98.0, 98.5, 97.9, 98.4), (98.4, 99.9, 98.3, 99.8)],
        [(99.7, 100.2, 99.6, 99.9), (99.9, 101.0, 99.8, 100.9)],
    ]
    for bars in cases:
        buy = sim(bars, close="15:59")
        sell = sim(mirror(bars), direction=-1, close="15:59")
        assert (sell["entry_time"], sell["exit_time"], sell["exit_reason"], sell["gap_entry"]) == (
            buy["entry_time"], buy["exit_time"], buy["exit_reason"], buy["gap_entry"])
        assert sell["entry_price"] == pytest.approx(200.0 - buy["entry_price"]) and sell["exit_price"] == pytest.approx(200.0 - buy["exit_price"])
    short = sim(mirror([(99.7, 100.2, 99.6, 99.9), (99.9, 100.4, 99.8, 100.3), (100.3, 100.4, 99.3, 99.4), (99.1, 99.2, 99.0, 99.1)]),
                direction=-1, close="09:39")
    assert short["exit_price"] == pytest.approx(100.9) and short["exit_price"] > short["entry_price"]  # a short stop-out buys back higher


def test_the_pessimistic_fill_is_never_better_than_filling_exactly_at_the_order_prices():
    """Against the paper's model (entry at the trigger, stop at the stop price, the trigger bar never stops the trade)."""
    rng = np.random.default_rng(42)
    seen = Counter()
    for _ in range(400):
        count = int(rng.integers(2, 40))
        close = 99.6 + np.cumsum(rng.normal(0.02, 0.25, count))
        open_ = np.concatenate([[99.6], close[:-1]]) + rng.normal(0, 0.1, count)
        high, low = np.maximum(open_, close) + rng.uniform(0, 0.2, count), np.minimum(open_, close) - rng.uniform(0, 0.2, count)
        for direction, columns in ((1, (open_, high, low, close)), (-1, (200 - open_, 200 - low, 200 - high, 200 - close))):
            o, h, lo, c = columns
            trade = trial.simulate_trade([trial._stamp(575 + i) for i in range(count)], o, h, lo, c, direction=direction, trigger=100.0,
                                         stop_distance=0.5, close_stamp=trial._stamp(575 + count), stale_minutes=30, stale_adverse=5.0,
                                         model="lower_bound")
            reached = np.flatnonzero(h >= 100.0) if direction > 0 else np.flatnonzero(lo <= 100.0)
            assert (trade is None) == (len(reached) == 0)
            if trade is None:
                continue
            k = int(reached[0])
            later_low = (lo[k + 1:] if direction > 0 else -h[k + 1:])
            stop_level = 99.5 if direction > 0 else -100.5
            naive = -0.5 if (later_low <= stop_level).any() else direction * (c[-1] - 100.0)
            pessimistic = direction * (trade["exit_price"] - trade["entry_price"])
            assert pessimistic <= naive + 1e-9 and direction * (trade["entry_price"] - 100.0) >= -1e-12
            assert trade["entry_time"] == trial._stamp(575 + k) and trade["entry_time"] <= trade["exit_time"]
            if trade["exit_reason"] != "session_end":
                assert pessimistic <= -0.5 + 1e-9  # a stop-out never loses less than the planned loss
            seen[trade["exit_reason"]] += 1
    assert set(seen) == {"session_end", "stop", "stop_gap", "stop_in_entry_bar"} and min(seen.values()) >= 10


def test_impossible_trades_are_refused_before_a_report_is_written():
    full = trial.contract_spec(json.loads(REPO_CONTRACT.read_text()))
    calendar = trial.SessionCalendar(dates=(date(2024, 7, 3),), closes=("13:00",))
    good = trade_rows([EARLY_CLOSE]).assign(exit_time="12:59")
    trial.validate_trades(good, calendar, full)
    for change, match in ((dict(entry_time="09:34"), "outside 09:35"), (dict(exit_time="13:00"), "outside 09:35"),
                          (dict(exit_time="09:39"), "outside 09:35"), (dict(gross=np.nan), "nonfinite"),
                          (dict(exit_price=0.0), "nonpositive exit"), (dict(weight=6.0), "leverage cap")):
        with pytest.raises(ValueError, match=match):
            trial.validate_trades(good.assign(**change), calendar, full)
    with pytest.raises(ValueError, match="duplicate orders"):
        trial.validate_trades(pd.concat([good, good]), calendar, full)


# --- sizing and costs -------------------------------------------------------------------------------

def test_sizing_risks_one_percent_of_a_slot_and_is_capped_at_four_times_the_slot():
    full = trial.contract_spec(json.loads(REPO_CONTRACT.read_text()))
    slot = 1 / 20
    by_risk = trial.shares_per_equity(50.0, 0.2, full)
    assert by_risk == pytest.approx(0.01 * slot / 0.2) and by_risk * 0.2 == pytest.approx(0.0005)  # a stop-out loses 0.05% of equity
    assert by_risk * 50.0 == pytest.approx(0.125)  # 2.5 times the slot
    capped = trial.shares_per_equity(50.0, 0.05, full)
    assert capped == pytest.approx(4 * slot / 50.0) and capped * 50.0 == pytest.approx(0.2)  # 4 times the slot, 20% of equity
    assert capped * 0.05 < 0.0005  # under the cap a stop-out loses less than the planned amount
    assert 20 * capped * 50.0 == pytest.approx(4.0)  # twenty capped names are exactly 4x equity


def trade_rows(sessions, *, direction=1, gross=0.0004, short_ok=False, matched=0.0, weight=0.2, status="filled"):
    return pd.DataFrame(dict(session=sessions, year=[s[:4] for s in sessions], symbol="AAA", rank=1, rel_volume=2.0, direction=direction,
                             short_ok=short_ok, status=status, entry_time="09:40", entry_price=50.0, exit_time="15:59", exit_price=50.1,
                             exit_reason="session_end", gap_entry=False, entry_at_open=True, shares=weight / 50.0, weight=weight,
                             cap_scale=1.6, gross=gross,
                             commission=2 * weight / 50.0 * 0.0035, turnover=2 * weight, matched_market=matched, r_multiple=0.5))


def test_costs_are_commission_per_share_plus_basis_points_on_both_fills_and_idle_days_are_zero():
    sessions = SESSIONS[15:25]
    part = trade_rows(sessions[:4])
    part.loc[1, ["gross", "commission", "turnover"]] = [-0.0005, 0.00003, 0.41]
    net, matched = trial.day_returns(part, sessions, 5.0)
    assert net[0] == pytest.approx(0.0004 - 2 * 0.004 * 0.0035 - 0.0005 * 0.4)  # 0.004 shares per dollar, two fills, 5 bps of 0.4
    assert net[1] == pytest.approx(-0.0005 - 0.00003 - 0.0005 * 0.41)
    assert net[4:].tolist() == [0.0] * 6 and matched.tolist() == [0.0] * 10
    free, _ = trial.day_returns(part, sessions, 0.0)
    double, _ = trial.day_returns(part, sessions, 10.0)
    assert free[0] == pytest.approx(0.0004 - 0.000028) and (free[:4] - net[:4]) == pytest.approx(net[:4] - double[:4])
    two = pd.concat([part, trade_rows(sessions[:1], gross=0.001).assign(symbol="BBB")])
    assert trial.day_returns(two, sessions, 5.0)[0][0] == pytest.approx(net[0] + 0.001 - 0.000028 - 0.0002)  # a day sums its trades


def test_build_trades_accounts_for_each_order_and_stops_at_the_early_close():
    session = EARLY_CLOSE
    full = trial.contract_spec(json.loads(REPO_CONTRACT.read_text()))
    order = dict(session=session, rel_volume=3.0, atr=2.0, stop_distance=0.2, prior_close=50.0, or_open=50.0, or_close=50.2)
    selection = pd.DataFrame([
        dict(order, symbol="BUY", rank=1, direction=1, trigger=50.4, short_ok=False),
        dict(order, symbol="SELL", rank=2, direction=-1, trigger=49.6, short_ok=True),
        dict(order, symbol="DOJI", rank=3, direction=0, trigger=np.nan, short_ok=False),
        dict(order, symbol="FLAT", rank=4, direction=1, trigger=60.0, short_ok=False),
        dict(order, symbol="GONE", rank=5, direction=1, trigger=50.4, short_ok=False),
        dict(order, symbol="LATE", rank=6, direction=1, trigger=50.4, short_ok=False),
        dict(order, session=SESSIONS[3], symbol="BUY", rank=1, direction=1, trigger=50.4, short_ok=False),
    ], columns=trial.SELECTION_COLUMNS)
    minutes = [trial._stamp(m) for m in range(trial._minute("09:35"), trial._minute("13:02"))]
    rise = np.linspace(50.0, 52.0, len(minutes))

    def path(symbol, prices, times=minutes):
        return pd.DataFrame(dict(session=session, symbol=symbol, time=times, open=prices, high=prices + 0.01, low=prices - 0.01,
                                 close=prices, volume=100.0))

    minute = pd.concat([path("BUY", rise), path("SELL", 100.0 - rise), path("FLAT", rise),
                        path("LATE", np.array([70.0, 70.0]), ["13:00", "13:01"]),  # only bars at or after the 13:00 close
                        path("BUY", rise).assign(session=SESSIONS[3])], ignore_index=True)
    market_minutes = [trial._stamp(m) for m in range(trial._minute("09:30"), trial._minute("13:00"))]
    market = pd.DataFrame(dict(session=session, time=market_minutes, open=np.linspace(499.9, 504.9, len(market_minutes)), high=506.0,
                               low=499.0, close=np.linspace(500.0, 505.0, len(market_minutes)), volume=1.0))
    calendar = trial.SessionCalendar(dates=(date(2024, 1, 5), date.fromisoformat(session)), closes=("16:00", "13:00"))
    trades = trial.build_trades(selection, minute, market, calendar, {session}, full).set_index("symbol")
    assert trades.status.to_dict() == {"BUY": "filled", "SELL": "filled", "DOJI": "doji", "FLAT": "not_triggered",
                                       "GONE": "no_minute_bars", "LATE": "no_minute_bars"}
    assert len(trades) == 6  # the ineligible session's order is not scored
    buy = trades.loc["BUY"]
    assert buy.entry_time >= "09:35" and buy.exit_time == "12:59" and buy.exit_reason == "session_end"
    exit_price = rise[minutes.index("12:59")]
    assert buy.exit_price == pytest.approx(exit_price) and buy.entry_price >= 50.4 and bool(buy.entry_at_open)
    trigger_bar = int(np.flatnonzero(rise + 0.01 >= 50.4)[0])
    assert buy.entry_time == minutes[trigger_bar + 1] and buy.entry_price == pytest.approx(rise[trigger_bar + 1])  # the next bar's open
    assert buy.cap_scale == pytest.approx(4 / 20 / 50.4 / buy.shares)
    shares = trial.shares_per_equity(50.4, 0.2, full)
    assert buy.shares == pytest.approx(shares) and buy.weight == pytest.approx(shares * buy.entry_price)
    assert buy.gross == pytest.approx(shares * (exit_price - buy.entry_price)) and buy.commission == pytest.approx(2 * shares * 0.0035)
    assert buy.turnover == pytest.approx(shares * (buy.entry_price + exit_price))
    assert buy.r_multiple == pytest.approx((exit_price - buy.entry_price) / 0.2)
    spy, spy_open = dict(zip(market.time, market.close)), dict(zip(market.time, market.open))
    assert buy.matched_market == pytest.approx(buy.weight * (spy["12:59"] / spy_open[buy.entry_time] - 1))  # from the open of the entry minute
    sell = trades.loc["SELL"]
    assert sell.gross == pytest.approx(sell.shares * (sell.entry_price - sell.exit_price)) and sell.gross > 0
    assert sell.matched_market == pytest.approx(-sell.weight * (spy["12:59"] / spy_open[sell.entry_time] - 1)) and sell.matched_market < 0
    exact = trial.build_trades(selection, minute, market, calendar, {session}, full, model="paper_exact").set_index("symbol")
    assert exact.at["BUY", "entry_price"] == 50.4 and exact.at["SELL", "entry_price"] == 49.6 and not exact.at["BUY", "entry_at_open"]
    assert exact.at["BUY", "matched_market"] == pytest.approx(exact.at["BUY", "weight"] * (spy["12:59"] / spy[exact.at["BUY", "entry_time"]] - 1))
    assert exact.at["BUY", "gross"] > buy.gross  # the paper's fill is the better price on a rising path
    trial.validate_trades(trades.reset_index(), calendar, full)
    assert trial.version_trades(trades.reset_index(), "long_only").symbol.tolist() == ["BUY"]
    assert trial.version_trades(trades.reset_index(), "long_short").symbol.tolist() == ["BUY", "SELL"]
    blocked = trades.reset_index().assign(short_ok=False)
    assert trial.version_trades(blocked, "long_short").symbol.tolist() == ["BUY"]
    assert trial.version_trades(blocked, trial.AS_PUBLISHED).symbol.tolist() == ["BUY", "SELL"]


# --- the decision -----------------------------------------------------------------------------------

def noisy(sessions, mean, seed=11):
    return mean + np.random.default_rng(seed).normal(0, 0.0002, len(sessions))


def judge(frame, sessions, chosen, *, conservative=None):
    tables = {model: frame for model in trial.FILL_MODELS}
    if conservative is not None:
        tables["conservative"] = conservative
    return trial.evaluate(tables, sessions, chosen)


def decision_spec(**support):
    contract = small_contract()
    contract["support_thresholds"] = contract["support_thresholds"] | {"minimum_filled_trades": 100} | support
    return trial.contract_spec(contract)


def test_long_only_needs_every_criterion_including_the_market_comparison():
    sessions = SESSIONS[15:]
    good = trade_rows(sessions, gross=noisy(sessions, 0.0015))
    report = judge(good, sessions, decision_spec())
    version = report["versions"]["long_only"]
    assert report["decision"] == version["decision"] == "eligible_for_untouched_test_planning" and all(version["criteria"].values())
    assert set(version["criteria"]) == {"adequate_support", "positive_lower_bound_vs_cash_at_primary", "positive_at_stress",
                                        "consistent_years", "positive_lower_bound_vs_exposure_matched_market_at_primary"}
    tiers = version["cost_tiers"]
    assert tiers["base"]["net_per_day_bps"] > tiers["stress"]["net_per_day_bps"] > tiers["double"]["net_per_day_bps"] > 0
    assert (tiers["stress"]["per_side_bps"], tiers["stress"]["role"], tiers["double"]["role"]) == (7.5, trial.STRESS_ROLE, "reported_only")
    assert report["deciding_fill_model"] == "bar_executable" and set(report["reported_only"]["fill_models"]) == set(trial.FILL_MODELS[1:])

    # The conservative fill model is reported only: losing money under it changes no decision and no criterion.
    cautious = judge(good, sessions, decision_spec(), conservative=trade_rows(sessions, gross=noisy(sessions, -0.01)))
    assert cautious["decision"] == cautious["secondary_decision"] == "eligible_for_untouched_test_planning"
    assert cautious["versions"] == report["versions"]
    assert cautious["reported_only"]["fill_models"]["conservative"]["long_only"]["cost_tiers"]["base"]["net_per_day_bps"] < 0
    assert "decision" not in cautious["reported_only"]["fill_models"]["conservative"]["long_only"]

    # Equal-notional sizing (every fill at the cap, 1.6 times larger here) is reported and decides nothing.
    sized = report["reported_only"][trial.EQUAL_NOTIONAL]["long_only"]
    assert "decision" not in sized and sized["cost_tiers"]["base"]["net_per_day_bps"] == pytest.approx(1.6 * tiers["base"]["net_per_day_bps"])
    assert version["cost_tiers"]["commission_only_as_published"]["net_per_day_bps"] > version["cost_tiers"]["half"]["net_per_day_bps"]
    assert version["filled_trades"] == len(sessions) and version["benchmark"] == "exposure_matched_market"

    # Positive at 5 bps (cost 2.3 bps of equity a day) but not at 7.5 bps (3.3 bps): retired.
    marginal = judge(trade_rows(sessions, gross=0.0003), sessions, decision_spec())
    criteria = marginal["versions"]["long_only"]["criteria"]
    assert marginal["decision"] == "retired" and criteria["positive_lower_bound_vs_cash_at_primary"] and not criteria["positive_at_stress"]

    # The whole gain is what the market did over the same minutes: beats cash, does not beat the matched market.
    drift = trade_rows(sessions, gross=noisy(sessions, 0.0015))
    drift["matched_market"] = drift.gross
    drifting = judge(drift, sessions, decision_spec())
    criteria = drifting["versions"]["long_only"]["criteria"]
    assert drifting["decision"] == "retired" and criteria["positive_lower_bound_vs_cash_at_primary"]
    assert not criteria["positive_lower_bound_vs_exposure_matched_market_at_primary"]

    # A noisy positive mean that the bootstrap cannot separate from zero does not pass.
    wild = trade_rows(sessions, gross=0.0004 + np.random.default_rng(3).normal(0, 0.01, len(sessions)))
    assert not judge(wild, sessions, decision_spec())["versions"]["long_only"]["criteria"]["positive_lower_bound_vs_cash_at_primary"]


def test_long_short_cannot_rescue_long_only_and_blocked_shorts_are_left_out():
    sessions = SESSIONS[15:]
    longs = trade_rows(sessions, gross=noisy(sessions, -0.0005))
    shorts = trade_rows(sessions, direction=-1, gross=noisy(sessions, 0.004, 13), short_ok=True).assign(symbol="BBB")
    blocked = trade_rows(sessions, direction=-1, gross=0.05, short_ok=False).assign(symbol="CCC")
    report = judge(pd.concat([longs, shorts, blocked], ignore_index=True), sessions, decision_spec())
    assert report["decision"] == "retired" and report["secondary_decision"] == "eligible_for_untouched_test_planning"
    assert report["secondary_decision_permits"] == "nothing; reported as information only"
    both = report["versions"]["long_short"]
    assert (both["long_trades"], both["short_trades"]) == (len(sessions), len(sessions)) and both["benchmark"] == "cash"
    assert "positive_lower_bound_vs_exposure_matched_market_at_primary" not in both["criteria"]
    published = report["reported_only"][trial.AS_PUBLISHED]
    assert published["short_trades"] == 2 * len(sessions) and "decision" not in published
    assert published["cost_tiers"]["base"]["net_per_day_bps"] > both["cost_tiers"]["base"]["net_per_day_bps"]
    assert report["reported_only"]["shorts_blocked_by_the_short_rule"] == len(sessions)


def test_thin_support_is_inconclusive_and_most_years_must_be_positive():
    sessions = SESSIONS[15:]
    good = trade_rows(sessions, gross=noisy(sessions, 0.0015))
    few = judge(good, sessions, decision_spec(minimum_filled_trades=10_000))
    assert few["decision"] == "inconclusive_budget_consumed" and not few["versions"]["long_only"]["criteria"]["adequate_support"]
    days = judge(good, sessions, decision_spec(minimum_eligible_days=5_000))
    assert days["decision"] == "inconclusive_budget_consumed"
    tiny = judge(good[good.session.isin(sessions[:8])], sessions[:8], decision_spec())
    assert tiny["versions"]["long_only"]["uncertainty_status"] == "unavailable_fewer_than_two_blocks"
    contract = small_contract()
    contract["acceptance"] = contract["acceptance"] | {"minimum_positive_years": 2}
    contract["support_thresholds"] = contract["support_thresholds"] | {"minimum_filled_trades": 100}
    one_year = judge(good, sessions, trial.contract_spec(contract))
    assert one_year["decision"] == "retired" and not one_year["versions"]["long_only"]["criteria"]["consistent_years"]
    assert one_year["versions"]["long_only"]["positive_years"] == 1 and list(one_year["versions"]["long_only"]["by_year"]) == ["2024"]


# --- end to end with a fake provider; synthetic prices only -----------------------------------------

PROFILE = {  # symbol: (price, daily move, intraday band, daily volume)
    **{name: (40.0 + 5 * i, 0.01, 0.015, 3e6) for i, name in enumerate(NAMES)},
    "PENNY": (3.0, 0.01, 0.015, 3e6), "THIN": (50.0, 0.01, 0.015, 2e5), "CALM": (20.0, 0.002, 0.001, 3e6),
    "SPLT": (60.0, 0.01, 0.015, 3e6), "DEAD": (55.0, 0.01, 0.015, 3e6)}
DEAD_SESSIONS = 100


def _bar(stamp, open, high, low, close, volume):  # noqa: A002
    return SimpleNamespace(timestamp=stamp, open=float(open), high=float(high), low=float(low), close=float(close),
                           volume=float(volume), vwap=float(close))


class _World:
    """A deterministic synthetic market served through the provider's request interface."""

    def __init__(self, *, extra_daily=(), drop_market=lambda session, stamp: False, fail=(), fail_opening=(), thin_opening=(),
                 drop_adjusted=None, fail_minute=()):
        self.fail_minute = set(fail_minute)
        self.extra_daily, self.drop_market, self.fail, self.requests = extra_daily, drop_market, set(fail), []
        self.fail_opening, self.thin_opening, self.drop_adjusted = set(fail_opening), set(thin_opening), drop_adjusted
        n = len(SESSIONS)
        self.daily = {}
        for k, (symbol, (price, move, band, volume)) in enumerate(PROFILE.items()):
            rng = np.random.default_rng(100 + k)
            close = price * np.exp(np.cumsum(rng.normal(0, move, n)))
            open_ = np.concatenate([[price], close[:-1]]) * np.exp(rng.normal(0, move / 3, n))
            frame = pd.DataFrame(dict(open=open_, high=np.maximum(open_, close) * (1 + band), low=np.minimum(open_, close) * (1 - band),
                                      close=close, volume=np.full(n, volume), factor=1.0))
            if symbol == "SPLT":  # before the split: twice the price, half the volume
                frame.loc[:SPLIT_AT - 1, ["open", "high", "low", "close"]] *= 2
                frame.loc[:SPLIT_AT - 1, "volume"] /= 2
                frame.loc[:SPLIT_AT - 1, "factor"] = 2.0
            self.daily[symbol] = frame.iloc[:DEAD_SESSIONS] if symbol == "DEAD" else frame

    def opening(self, symbol, i):
        rng = np.random.default_rng([list(PROFILE).index(symbol), i, 1])
        day = self.daily[symbol].iloc[i]
        close = day.open * (1 + 0.002 * rng.choice([-1, 0, 1], p=[0.45, 0.1, 0.45]))
        return dict(open=day.open, high=max(day.open, close) * 1.001, low=min(day.open, close) * 0.999, close=close,
                    volume=0.05 * day.volume * float(np.exp(rng.normal(0, 0.6))))

    def path(self, symbol, i):
        rng = np.random.default_rng([list(PROFILE).index(symbol), i, 2])
        count = CALENDAR.regular_minutes(i) - 5
        close = self.opening(symbol, i)["close"] * np.exp(np.cumsum(rng.normal(0, 0.0012, count)))
        open_ = np.concatenate([[self.opening(symbol, i)["close"]], close[:-1]])
        return open_, np.maximum(open_, close) * 1.0003, np.minimum(open_, close) * 0.9997, close

    def _at(self, session, stamp):
        return datetime.fromisoformat(f"{session}T{stamp}").replace(tzinfo=trial.NEW_YORK).astimezone(UTC)

    def get_stock_bars(self, request):
        self.requests.append(request)
        symbols = request.symbol_or_symbols if isinstance(request.symbol_or_symbols, list) else [request.symbol_or_symbols]
        if self.fail & set(symbols):
            raise ValueError("provider error")
        kind = request.timeframe.value
        start, end = (t.replace(tzinfo=UTC).astimezone(trial.NEW_YORK) for t in (request.start, request.end))
        inside = [i for i, s in enumerate(SESSIONS) if start.date().isoformat() <= s <= end.date().isoformat()]
        data = {}
        for symbol in symbols:
            bars = []
            if symbol == "SPY" and kind == "1Min":
                for i in inside:
                    session, minutes = SESSIONS[i], CALENDAR.regular_minutes(i)
                    price = 500.0 + 0.01 * np.arange(minutes)
                    bars.append(_bar(self._at(session, "08:00"), 1, 1, 1, 1, 1))  # pre-market: discarded
                    for m in range(minutes):
                        stamp = trial._stamp(trial._minute("09:30") + m)
                        if not self.drop_market(session, stamp):
                            bars.append(_bar(self._at(session, stamp), price[m], price[m] + 0.02, price[m] - 0.02, price[m] + 0.01, 1e5))
                    bars.append(_bar(self._at(session, CALENDAR.closes[i]), 2, 2, 2, 2, 1))  # stamped at the close: discarded
            elif symbol in PROFILE:
                frame = self.daily[symbol]
                held = [i for i in inside if i < len(frame)]
                if kind == "1Month":
                    for month in sorted({SESSIONS[i][:7] for i in held}):
                        rows = frame.iloc[[i for i in held if SESSIONS[i][:7] == month]]
                        bars.append(_bar(datetime(int(month[:4]), int(month[5:]), 1, 5, tzinfo=UTC), rows.open.iat[0], rows.high.max(),
                                         rows.low.min(), rows.close.iat[-1], rows.volume.sum()))
                elif kind == "1Day":
                    scale = frame.factor if request.adjustment.value == "split" else pd.Series(1.0, index=frame.index)
                    for i in held:
                        if request.adjustment.value == "split" and self.drop_adjusted == (symbol, i):
                            continue
                        row, d = frame.iloc[i], date.fromisoformat(SESSIONS[i])
                        bars.append(_bar(datetime(d.year, d.month, d.day, 5, tzinfo=UTC), row.open / scale.iat[i], row.high / scale.iat[i],
                                         row.low / scale.iat[i], row.close / scale.iat[i], row.volume * scale.iat[i]))
                    if symbol == symbols[0]:
                        bars += [_bar(stamp, 50, 51, 49, 50, 3e6) for stamp in self.extra_daily]
                elif kind == "5Min":
                    assert end - start < timedelta(minutes=5) and start.strftime("%H:%M") == "09:30"  # nothing after 09:35 is asked for
                    for i in held:
                        if SESSIONS[i] in self.fail_opening:
                            raise ValueError("provider error")
                        if SESSIONS[i] not in self.thin_opening or symbol == "SA":
                            bars.append(_bar(self._at(SESSIONS[i], "09:30"), **self.opening(symbol, i)))
                elif kind == "1Min":
                    assert len(held) <= 1
                    for i in held:
                        if SESSIONS[i] in self.fail_minute:
                            raise ValueError("provider error")
                        columns = self.path(symbol, i)
                        base = self._at(SESSIONS[i], "09:35")
                        bars += [_bar(base + timedelta(minutes=m), o, h, low, c, 100) for m, (o, h, low, c) in enumerate(zip(*columns))]
            if bars:
                data[symbol] = bars
        return SimpleNamespace(data=data)


class _FakeTrading:
    def get_all_assets(self, request):
        asset = lambda symbol, name, exchange, shortable=True: SimpleNamespace(  # noqa: E731
            symbol=symbol, name=name, exchange=SimpleNamespace(value=exchange), shortable=shortable, easy_to_borrow=shortable)
        if request.status.value == "inactive":
            return [asset("DEAD", "Dead Corp Common Stock", "NYSE", False)]
        return ([asset(name, f"{name} Inc. Common Stock", "NASDAQ" if i % 2 else "NYSE", name != "SH") for i, name in enumerate(NAMES)]
                + [asset(s, f"{s} Corp", "NYSE") for s in ("PENNY", "THIN", "CALM", "SPLT")]
                + [asset("QQQX", "Big Index ETF", "NASDAQ"), asset("ARCX", "Arca Listed Corp", "ARCA"),
                   asset("SPY", "SPDR S&P 500 ETF Trust", "ARCA")])


def _clients(**world):
    return trial.Clients(trading=_FakeTrading(), data=_World(**world))


def _register(root, payload=None):
    payload = payload or small_contract()
    (root / "config").mkdir(exist_ok=True)
    (root / "config/trial.json").write_text(json.dumps(payload))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = root / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return state_path


def _run(root, mode, clients=None, path="config/trial.json"):
    factory = (lambda: clients) if clients is not None else (lambda: pytest.fail("fetched"))
    return trial.run(root, mode=mode, contract_path=Path(path), client_factory=factory)


@pytest.fixture(scope="module")
def acquired_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("orb")
    _register(root)
    clients = _clients()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trial, "sleep", lambda seconds: None)
        patch.setattr(swing, "sleep", lambda seconds: None)
        result = _run(root, "acquire", clients)
    return root, result, clients


@pytest.fixture
def acquired(acquired_template, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(acquired_template[0], root)
    return root, root / f"artifacts/{CAMPAIGN}/campaign_state.json"


def _manifest(root):
    return json.loads((root / f"artifacts/{CAMPAIGN}/data/acquisition.json").read_text())


def test_acquisition_freezes_the_filters_the_ranking_and_only_the_bars_the_rules_need(acquired_template):
    root, result, clients = acquired_template
    counts = result["counts"]
    assert result["status"] == "acquisition_written" and result["trial_claimed"] is False
    assert counts["candidates"] == 13 and counts["excluded_name"] == 1 and counts["excluded_exchange"] == 2  # the ETF and the Arca listings
    assert counts["shortlist"] == 11  # THIN never trades 7,000,000 shares in a month; PENNY never trades above $5
    assert counts["names_ever_passing_daily_filters"] == 10 and counts["split_events"] == 1  # CALM fails the range filter
    manifest = _manifest(root)
    assert manifest["failed"] == {"monthly": {}, "daily": {}, "opening": {}, "minute": {}} and manifest["quality_passed"] is True
    coverage = manifest["opening_coverage"]
    assert len(coverage) == len(SESSIONS) - 1 and all(asked == returned and lost == 0 for asked, returned, lost in coverage.values())
    assert sum(asked for asked, _, _ in coverage.values()) == counts["opening_bars_requested"] == counts["opening_bars"]
    assert json.loads((root / manifest["frozen"]["splits"]["path"]).read_text()) == [["SPLT", SESSIONS[SPLIT_AT]]]
    selection = pd.read_csv(root / manifest["frozen"]["selection"]["path"])
    index = selection.session.map({s: i for i, s in enumerate(SESSIONS)})
    assert index.min() == 15 and (selection.groupby("session").size() <= 3).all() and (selection.rel_volume >= 1.0).all()
    assert not set(selection.symbol) & {"PENNY", "THIN", "CALM", "QQQX", "ARCX", "SPY"}
    split_rows = index[selection.symbol == "SPLT"]
    assert len(split_rows) and not split_rows.between(SPLIT_AT, SPLIT_AT + 14).any()  # doubled volume after the split is not "in play"
    assert index[selection.symbol == "DEAD"].max() < DEAD_SESSIONS and counts["selected_inactive_symbols"] == 1
    assert not selection.short_ok[selection.symbol.isin(["DEAD", "SH"])].any() and selection.short_ok.any()

    # One selected row, recomputed from the synthetic market by hand.
    row = selection[index == 80].iloc[0]
    world = clients.data
    k = 80
    volumes = [world.opening(row.symbol, i)["volume"] for i in range(k - 14, k + 1)]
    assert row.rel_volume == pytest.approx(volumes[-1] / np.mean(volumes[:-1]))
    bar = world.opening(row.symbol, k)
    assert row.direction == np.sign(bar["close"] - bar["open"]) and row.or_open == pytest.approx(bar["open"])

    # The ranking's inputs stop at 09:35: opening requests cover 09:30:00-09:34:59, minute requests start at 09:35.
    kinds = Counter(request.timeframe.value for request in world.requests)
    assert set(kinds) == {"1Month", "1Day", "5Min", "1Min"} and kinds["5Min"] == len(SESSIONS) - 1  # the first session is nobody's history
    for request in world.requests:
        start, end = (t.replace(tzinfo=UTC).astimezone(trial.NEW_YORK).strftime("%H:%M:%S") for t in (request.start, request.end))
        if request.timeframe.value == "5Min":
            assert (start, end) == ("09:30:00", "09:34:59")
        elif request.timeframe.value == "1Min" and request.symbol_or_symbols != ["SPY"]:
            assert start == "09:35:00" and len(request.symbol_or_symbols) <= 3
            assert set(request.symbol_or_symbols) <= set(selection.symbol[selection.session == request.start.date().isoformat()])
    minute = pd.concat(pd.read_csv(root / source["path"]) for name, source in manifest["sources"].items() if name.startswith("minute/"))
    assert minute.time.min() == "09:35" and minute.time.max() == "15:59" and minute[minute.session == EARLY_CLOSE].time.max() == "12:59"
    market = pd.concat(pd.read_csv(root / source["path"]) for name, source in manifest["sources"].items() if name.startswith("market/"))
    assert market.time.min() == "09:30" and market.time.max() == "15:59" and len(market) == sum(CALENDAR.regular_minutes(i) for i in range(len(SESSIONS)))
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", clients)


def test_the_stages_run_in_order_and_claim_exactly_once(acquired, monkeypatch):
    root, state_path = acquired
    directory = root / f"artifacts/{CAMPAIGN}"
    monkeypatch.setattr(trial, "rebuild_selection", lambda *a: pytest.fail("prices read by the support audit"))
    audit = _run(root, "support-audit")
    assert audit["status"] == trial.SUPPORT_OK and audit["prices_read_for_analysis"] is False and audit["strategy_outcomes_computed"] is False
    assert audit["eligible_days"] == len(SESSIONS) - 15 and audit["skipped_sessions"] == {"insufficient_history": 15}
    assert audit["median_selected_names"] >= 2 and audit["audit"] == "session_symbol_and_bar_time_only"
    monkeypatch.undo()
    monkeypatch.setattr(trial, "sleep", lambda seconds: None)
    monkeypatch.setattr(trial, "build_trades", lambda *a, **k: pytest.fail("a trade was computed before the claim"))
    preflight = _run(root, "preflight")
    assert preflight["status"] == "preflight_passed" and preflight["trial_claimed"] is False
    assert json.loads(state_path.read_text())["trials"] == [] and not (directory / "report.json").exists()
    monkeypatch.undo()
    report = _run(root, "fit")
    state = json.loads(state_path.read_text())
    assert len(state["trials"]) == 1 and state["trials"][0]["status"] == report["decision"]
    assert report["decision"] in ("eligible_for_untouched_test_planning", "retired", "inconclusive_budget_consumed")
    assert report["survivorship_label"] == "incomplete_direction_unknown" and report["orders_sent"] == 0
    assert report["holdout_evaluated"] is False and report["promotion_authority"] is False and report["signature"] == preflight["signature"]
    assert set(report["output_sha256"]) == {f"trades_{model}.csv" for model in trial.FILL_MODELS}
    assert report["deciding_fill_model"] == "bar_executable" and set(report["reported_only"]["fill_models"]) == set(trial.FILL_MODELS[1:])
    assert set(report["reported_only"][trial.EQUAL_NOTIONAL]) == set(trial.VERSIONS)
    assert not any("conservative" in key for v in trial.VERSIONS for key in report["versions"][v]["criteria"])
    trades = pd.read_csv(directory / "trades_bar_executable.csv")
    filled = trades[trades.status == "filled"]
    assert filled.entry_at_open.all() and (filled.entry_time >= "09:36").all()  # never the trigger bar itself
    harsh = pd.read_csv(directory / "trades_lower_bound.csv")
    assert harsh.gross.sum() < pd.read_csv(directory / "trades_conservative.csv").gross.sum() <= filled.gross.sum()
    assert set(trades.status) >= {"filled", "doji", "not_triggered"} and (filled.entry_time >= "09:35").all()
    assert (filled.exit_time <= "15:59").all() and (filled.weight <= 4 / 3 * 1.05).all()
    assert trades.session.min() >= SESSIONS[15] and (trades.groupby("session").size() <= 3).all()
    versions = report["versions"]
    assert versions["long_only"]["short_trades"] == 0 and versions["long_short"]["short_trades"] > 0
    assert report["reported_only"][trial.AS_PUBLISHED]["short_trades"] > versions["long_short"]["short_trades"]  # SH and DEAD shorts are blocked
    assert versions["long_only"]["eligible_days"] == len(SESSIONS) - 15
    # Costs only ever subtract: each dearer tier is worse.
    tiers = versions["long_short"]["cost_tiers"]
    assert tiers["commission_only_as_published"]["net_per_day_bps"] > tiers["half"]["net_per_day_bps"] > tiers["base"]["net_per_day_bps"] > tiers["double"]["net_per_day_bps"]
    monkeypatch.setattr(trial, "load_stamps", lambda *a: pytest.fail("data read"))
    again = _run(root, "fit")
    assert again["status"] == "budget_already_claimed" and len(again["trials"]) == 1


def test_a_crash_after_the_claim_consumes_the_budget(acquired, monkeypatch):
    root, state_path = acquired

    def crash(*args):
        raise RuntimeError("crash")

    monkeypatch.setattr(trial, "evaluate", crash)
    with pytest.raises(RuntimeError, match="crash"):
        _run(root, "fit")
    assert json.loads(state_path.read_text())["trials"][0]["status"] == "claimed"
    monkeypatch.undo()
    assert _run(root, "fit")["status"] == "budget_already_claimed"


def test_changed_sources_and_a_selection_that_does_not_match_its_bars_are_refused(acquired):
    root, state_path = acquired
    manifest_path = root / f"artifacts/{CAMPAIGN}/data/acquisition.json"
    manifest = _manifest(root)
    selection_path = root / manifest["frozen"]["selection"]["path"]
    original = selection_path.read_text()
    frame = pd.read_csv(selection_path)
    frame.loc[frame.index[5], "symbol"] = "SA" if frame.symbol.iat[5] != "SA" else "SB"
    selection_path.write_text(frame.to_csv(index=False))
    with pytest.raises(ValueError, match="source changed"):
        _run(root, "support-audit")
    manifest["frozen"]["selection"]["sha256"] = trial.digest(selection_path)  # even a re-hashed selection must equal the rebuilt one
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match the selection rebuilt"):
        _run(root, "preflight")
    selection_path.write_text(original)
    manifest["frozen"]["selection"]["sha256"] = trial.digest(selection_path)
    manifest_path.write_text(json.dumps(manifest))
    assert _run(root, "preflight")["status"] == "preflight_passed"
    minute_path = root / manifest["sources"]["minute/2024-03"]["path"]
    minute_path.write_bytes(minute_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source changed"):
        _run(root, "fit")
    manifest_path.write_text(json.dumps(manifest | {"contract_hash": "0" * 64}))
    with pytest.raises(ValueError, match="does not belong"):
        _run(root, "support-audit")
    assert json.loads(state_path.read_text())["trials"] == []


def test_fail_closed_without_registration_reset_entry_or_ledger(acquired):
    root, state_path = acquired
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": []}))
    with pytest.raises(ValueError, match="research reset does not list"):
        _run(root, "fit")
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    (root / "config/draft.json").write_text(json.dumps(small_contract(status="draft_not_registered")))
    with pytest.raises(ValueError, match="not registered"):
        _run(root, "fit", path="config/draft.json")
    state_path.unlink()
    with pytest.raises(ValueError, match="original campaign registration"):
        _run(root, "fit")
    assert not (root / f"artifacts/{CAMPAIGN}/report.json").exists()


def test_an_incomplete_market_session_is_skipped_and_a_degraded_download_is_refused(tmp_path):
    root = tmp_path / "one"
    root.mkdir()
    _register(root)
    skipped = SESSIONS[50]
    result = _run(root, "acquire", _clients(drop_market=lambda session, stamp: session == skipped and stamp == "11:17"))
    assert result["counts"]["incomplete_market_sessions"] == 1
    audit = _run(root, "support-audit")
    assert audit["skipped_sessions"] == {"insufficient_history": 15, "market_incomplete": 1} and audit["eligible_days"] == len(SESSIONS) - 16
    report = _run(root, "fit")
    assert skipped not in set(pd.read_csv(root / f"artifacts/{CAMPAIGN}/trades_bar_executable.csv").session) and report["skipped_sessions"]["market_incomplete"] == 1

    other = tmp_path / "two"
    other.mkdir()
    _register(other)
    with pytest.raises(ValueError, match="degraded download:incomplete_market_sessions=3"):
        _run(other, "acquire", _clients(drop_market=lambda session, stamp: session in SESSIONS[50:53] and stamp == "10:00"))
    assert not (other / f"artifacts/{CAMPAIGN}/data/acquisition.json").exists()  # nothing to trust; a retry is possible


def test_holdout_rows_and_lasting_fetch_failures_are_refused_and_nothing_is_written(tmp_path):
    for name, world, match in (
            # One bad session must not hide among 145 good ones: each session's opening bars are checked on their own.
            ("outage", dict(fail_opening=[SESSIONS[70]]), f"too many opening-bar fetch failures:{SESSIONS[70]}"),
            ("thin", dict(thin_opening=[SESSIONS[70]]), f"opening bars cover too few names:{SESSIONS[70]}:returned=1"),
            # The split guard reads the adjusted series: it must cover exactly the raw bars.
            ("adjusted", dict(drop_adjusted=("SB", 40)), "split-adjusted daily bars do not cover the raw bars"),
            ("holdout", dict(extra_daily=[datetime(2026, 9, 10, 5, tzinfo=UTC)]), "outside development window"),
            ("year", dict(extra_daily=[datetime(2026, 1, 5, 5, tzinfo=UTC)]), "outside development window"),
            ("before", dict(extra_daily=[datetime(2023, 12, 29, 5, tzinfo=UTC)]), "outside development window"),
            ("failed", dict(fail=["SC"]), "too many monthly fetch failures")):
        root = tmp_path / name
        root.mkdir()
        _register(root)
        with pytest.raises(ValueError, match=match):
            _run(root, "acquire", _clients(**world))
        assert not (root / f"artifacts/{CAMPAIGN}/data/acquisition.json").exists()


def test_a_failed_batch_is_retried_symbol_by_symbol_and_every_request_is_paced(_no_pauses):
    world = _World(fail=["SC"])
    start, end = trial._at(SESSIONS[20], "09:30"), trial._at(SESSIONS[20], "09:35", seconds_before=1)
    frame, failed = trial.fetch_frame(world, ["SA", "SB", "SC"], start, end, unit="opening")
    assert failed == {"SC": "ValueError"} and sorted(frame.symbol) == ["SA", "SB"]
    assert frame.session.tolist() == [SESSIONS[20]] * 2 and frame.time.tolist() == ["09:30"] * 2
    assert _no_pauses.count(trial.REQUEST_PAUSE_SECONDS) >= 4  # each provider call is followed by a pause
    days, failed = trial.fetch_frame(world, ["SA"], trial._at(SESSIONS[0], "00:00"), trial._at(SESSIONS[9], "23:59"), unit="day")
    assert failed == {} and days.session.tolist() == SESSIONS[:10] and list(days.columns) == trial.MINUTE_COLUMNS


def test_outputs_are_deterministic_across_independent_acquisitions(acquired_template, tmp_path):
    _register(tmp_path)
    _run(tmp_path, "acquire", _clients())
    first, second = _manifest(acquired_template[0]), _manifest(tmp_path)
    assert first["sources"] == second["sources"] and first["frozen"] == second["frozen"] and first["counts"] == second["counts"]


def test_the_probe_runs_on_a_draft_and_keeps_counts_only(tmp_path):
    draft = small_contract(status="draft_not_registered")
    draft["data_probe"] = {"coverage_symbols": ["SPY", "SA"], "coverage_sessions": [SESSIONS[20], EARLY_CLOSE],
                           "delisted": [{"symbol": "DEAD", "session": SESSIONS[30]}, {"symbol": "GONE", "session": SESSIONS[30]}]}
    (tmp_path / "config").mkdir()
    (tmp_path / "config/trial.json").write_text(json.dumps(draft))
    result = _run(tmp_path, "probe", _clients())
    assert result["status"] == "probe_complete" and result["prices_read_for_analysis"] is False and result["trial_claimed"] is False
    assert (result["delisted_probed"], result["delisted_in_asset_list"], result["delisted_with_minute_bars"]) == (2, 1, 1)
    spy = result["coverage"][f"SPY:{EARLY_CLOSE}"]
    assert (spy["regular_minutes"], spy["minute_bars"], spy["first_minute"], spy["last_minute"]) == (210, 210, "09:30", "12:59")
    stock = result["coverage"][f"SA:{SESSIONS[20]}"]
    assert stock["opening_bar"] is True and stock["daily_bar"] is True and stock["asset_statuses"] == ["active"]
    assert result["delisted"]["DEAD"]["asset_statuses"] == ["inactive"] and result["delisted"]["GONE"]["minute_bars"] == 0
    written = json.loads((tmp_path / f"artifacts/{CAMPAIGN}/data_probe.json").read_text())
    assert not any(key in json.dumps(written) for key in ('"open"', '"close"', '"high"', '"low"'))
    assert not (tmp_path / f"artifacts/{CAMPAIGN}/campaign_state.json").exists()


def test_tickers_that_look_like_missing_values_are_read_as_text(tmp_path):
    frame = pd.DataFrame(dict(session=SESSIONS[20], symbol=["NA", "NAN", "NULL", "AAA"], time="09:35", open=1.0, high=2.0, low=0.5,
                              close=[1.5, 1.5, np.nan, 1.5], volume=10.0))
    source = trial._write_table(tmp_path, tmp_path / "minute.csv.gz", frame)
    back = trial._read(tmp_path, source, trial.MINUTE_COLUMNS)
    assert back.symbol.astype(str).tolist() == ["NA", "NAN", "NULL", "AAA"] and not back.symbol.isna().any()
    assert back.close.isna().tolist() == [False, False, True, False] and back.session.tolist() == [SESSIONS[20]] * 4
    opening = pd.DataFrame([dict(session=SESSIONS[i], symbol="NA", open=50.0, high=50.3, low=49.9, close=50.2, volume=100_000.0 + i)
                            for i in range(20)])
    source = trial._write_table(tmp_path, tmp_path / "opening.csv.gz", opening)
    parts = lambda: trial._opening_parts(tmp_path, {"opening/2024-01": source})  # noqa: E731
    selection = trial.select_from_parts(feature_rows(["NA"], range(15, 20)), parts, SESSIONS, frozenset(), spec())
    assert selection.symbol.tolist() == ["NA"] * 5 and selection.session.tolist() == SESSIONS[15:20]


def test_a_name_with_exactly_fourteen_bars_does_not_borrow_history_from_the_name_before_it():
    # AAA trades the last sessions of the window, BBB the first 14: BBB's 14th bar has only 13 behind it.
    sessions = SESSIONS[:40]
    rows = [dict(session=sessions[i], symbol="AAA", open=50.0, high=50.3, low=49.9, close=50.2, volume=1e5) for i in range(20, 40)]
    rows += [dict(session=sessions[i], symbol="BBB", open=50.0, high=50.3, low=49.9, close=50.2, volume=9e9) for i in range(0, 14)]
    features = pd.concat([feature_rows(["AAA"], range(20, 40)), feature_rows(["BBB"], range(0, 14))])
    selection = trial.build_selection(features, pd.DataFrame(rows), sessions, frozenset(), spec())
    assert set(selection.symbol) == {"AAA"} and selection.session.min() == sessions[34]


def test_the_selection_stays_small_in_memory_at_scale():
    """About a seventh of the full download (500 sessions x 1,000 names); the full-scale figure is in the contract."""
    import tracemalloc

    rng = np.random.default_rng(0)
    n_sessions, n_names = 500, 1000
    sessions = [f"2{i:04d}" for i in range(n_sessions)]
    names = np.array([f"N{j:04d}" for j in range(n_names)], dtype=object)
    present = rng.random((n_sessions, n_names)) < 0.98
    at, who = np.nonzero(present)
    months = at // 21
    price_ = 20 + rng.random(len(at)) * 50
    opening = pd.DataFrame(dict(session=pd.Categorical.from_codes(at, sessions), symbol=pd.Categorical.from_codes(who, names), open=price_,
                                high=price_ + 0.2, low=price_ - 0.2, close=price_ + rng.choice([-0.1, 0.1], len(at)),
                                volume=rng.integers(10_000, 1_000_000, len(at)).astype(float)))
    keep = (rng.random(len(at)) < 0.33) & (at >= 15)
    features = pd.DataFrame(dict(idx=at[keep].astype(np.int32), symbol=names[who[keep]], open=price_[keep], atr=2.0, avg_volume=2e6,
                                 prior_close=price_[keep], ssr_carry=False))
    pieces = [piece for _, piece in opening.groupby(months, sort=True)]
    del opening
    tracemalloc.start()
    selection = trial.select_from_parts(features, lambda: iter(pieces), sessions, frozenset(), spec())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows = sum(len(piece) for piece in pieces)
    assert rows > 480_000 and selection.groupby("session").size().median() == 3 and len(selection) > 1400
    assert peak < 100 * rows, f"peak {peak / rows:.0f} bytes per opening bar"  # measured 64; the pivot table this replaced used about 480
    whole = trial.build_selection(features, pd.concat(pieces, ignore_index=True), sessions, frozenset(), spec())
    pd.testing.assert_frame_equal(whole, selection)


def test_a_thin_opening_on_an_unscored_session_is_recorded_and_only_a_scored_session_refuses(tmp_path):
    """The 2020-03-16 case: a market-wide halt just after the open leaves few opening bars and an incomplete SPY record."""
    halted = SESSIONS[70]
    contract = small_contract()
    contract["support_audit"] = contract["support_audit"] | {"report_opening_coverage_for": [halted, "2020-03-16"]}
    _register(tmp_path, contract)
    world = dict(thin_opening=[halted], drop_market=lambda session, stamp: session == halted and "09:31" <= stamp < "09:46")
    result = _run(tmp_path, "acquire", _clients(**world))
    manifest = _manifest(tmp_path)
    asked, returned, lost = manifest["opening_coverage"][halted]
    assert result["status"] == "acquisition_written" and (returned, lost) == (1, 0) and asked > 2
    assert manifest["opening_shortfall_sessions"] == [halted] and result["counts"]["opening_shortfall_sessions_unscored"] == 1
    audit = _run(tmp_path, "support-audit")
    review = audit["pre_claim_review"]
    assert review["split_events"] == 1 and review["opening_shortfall_sessions"] == [halted]
    assert review["opening_coverage_asked_returned_failed"] == {halted: [asked, 1, 0], "2020-03-16": None}
    assert review["session_status"] == {halted: "market_incomplete"} and audit["prices_read_for_analysis"] is False
    # Names without that session's opening bar are unranked for the next 14 sessions; SA, which had it, still can be.
    selection = pd.read_csv(tmp_path / manifest["frozen"]["selection"]["path"])
    after = selection[selection.session.isin(SESSIONS[70:85])]
    assert set(after.symbol) <= {"SA"} and len(selection[selection.session == SESSIONS[85]]) >= 1
    # The same thin opening with a complete SPY record is a scored session with bad data: refused, after the SPY bars are in.
    scored = tmp_path / "scored"
    scored.mkdir()
    _register(scored)
    clients = _clients(thin_opening=[halted])
    with pytest.raises(ValueError, match=f"opening bars cover too few names:{halted}:returned=1"):
        _run(scored, "acquire", clients)
    assert any(r.timeframe.value == "1Min" and r.symbol_or_symbols == ["SPY"] for r in clients.data.requests)
    assert not (scored / f"artifacts/{CAMPAIGN}/data/acquisition.json").exists()


def test_a_selected_name_whose_minute_request_failed_is_counted_once(acquired_template, tmp_path):
    selection = pd.read_csv(acquired_template[0] / _manifest(acquired_template[0])["frozen"]["selection"]["path"])
    orders = selection[selection.direction != 0]
    per_session = orders.groupby("session").size()
    session = per_session.idxmax()
    lost, limit = int(per_session.max()), 0.01 * len(orders)
    assert lost <= limit < 2 * lost  # inside the contract's 1 percent; counting it twice would have refused
    _register(tmp_path)
    result = _run(tmp_path, "acquire", _clients(fail_minute=[session]))
    assert result["counts"]["orders_without_minute_bars"] == lost == len(_manifest(tmp_path)["failed"]["minute"])
    other = tmp_path / "two"
    other.mkdir()
    _register(other)
    second = per_session.drop(session).idxmax()
    with pytest.raises(ValueError, match="too many selected names without minute bars"):
        _run(other, "acquire", _clients(fail_minute=[session, second]))


# --- the selection against a brute-force reference on gappy data (adapted from the second review) ---

def _gappy_world(seed, n_sessions=90, n_symbols=45):
    rng = np.random.default_rng(seed)
    names = ["NA", "NAN", "NULL", "A", "AA", "AAA", "TRUE", "NONE", "INF"]
    while len(names) < n_symbols:
        name = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=int(rng.integers(1, 6))))
        if name not in names:
            names.append(name)
    bars, opening, splits = {}, {}, []
    for j, name in enumerate(names):
        kind = j % 9
        start = int(rng.integers(1, n_sessions - 20)) if kind == 1 else 0  # late listing
        end = int(rng.integers(20, n_sessions)) if kind == 2 else n_sessions  # delisted
        level = float(rng.choice([4.0, 5.0, 5.2, 8.0, 30.0, 120.0]))
        band, volume = float(rng.choice([0.005, 0.02, 0.06, 0.1])), float(rng.choice([6e5, 1e6, 1.05e6, 3e6, 2e7]))
        split_at = int(rng.integers(16, n_sessions)) if kind == 3 else None
        day, first = {}, {}
        for i in range(start, end):
            if i == split_at:
                level /= 2
                splits.append((name, SESSIONS[i]))
            level = max(0.5, level * (1 + rng.normal(0, 0.02)))
            open_ = 5.0 if rng.random() < 0.05 else round(level * (1 + rng.normal(0, 0.01)), 2)  # sometimes exactly on the threshold
            close = round(level, 2)
            high, low = round(max(open_, close) + band * level * rng.random(), 2), round(max(0.01, min(open_, close) - band * level * rng.random()), 2)
            shares = 1e6 if kind == 4 else float(int(volume * rng.uniform(0.5, 1.6)))  # kind 4: exactly at the volume threshold
            if not (kind == 5 and rng.random() < 0.03) and not (kind == 6 and i % 37 == 5):
                day[i] = (open_, high, low, close, shares)
            if rng.random() >= {6: 0.02, 7: 0.07}.get(kind, 0.0):
                last = round(open_ * (1 + rng.choice([-0.003, 0.0, 0.003])), 2)
                traded = 50_000.0 if kind == 8 else float(int(rng.integers(1, 6)) * 10_000) if kind == 0 else float(int(rng.integers(5_000, 400_000)))
                first[i] = (open_, round(max(open_, last) + 0.05, 2), round(max(0.01, min(open_, last) - 0.05), 2), last,
                            0.0 if rng.random() < 0.01 else traded)
        bars[name], opening[name] = day, first
    for name, lo, hi in (("ZEXA", 0, 14), ("AEXA", n_sessions - 14, n_sessions), ("MEXA", 40, 54), ("MEXB", 40, 55)):  # exactly 14 or 15 bars
        names.append(name)
        bars[name] = {i: (50.0, 52.0, 48.0, 50.0, 5e6) for i in range(n_sessions)}
        opening[name] = {i: (50.0, 50.4, 49.9, 50.2, 9e9 if i == hi - 1 else 1e3) for i in range(lo, hi)}
    return names, bars, opening, splits


def _reference_selection(sessions, names, bars, opening, splits, not_shortable, chosen):
    """The contract's words, one session and one name at a time, with no shared code."""
    split_at = {}
    for name, session in splits:
        split_at.setdefault(name, []).append(sessions.index(session))
    rows = []
    for t in range(len(sessions)):
        candidates = []
        for name in names:
            day, first = bars[name], opening[name]
            if t < 15 or any(i not in day for i in range(t - 15, t + 1)) or any(t - 14 <= at <= t for at in split_at.get(name, [])):
                continue
            ranges = [max(day[i][1] - day[i][2], abs(day[i][1] - day[i - 1][3]), abs(day[i][2] - day[i - 1][3])) for i in range(t - 14, t)]
            atr = float(np.mean(ranges))
            if not (day[t][0] > chosen.min_open_price and sum(day[i][4] for i in range(t - 14, t)) / 14 >= chosen.min_avg_volume
                    and atr > chosen.min_atr) or any(i not in first for i in range(t - 14, t + 1)):
                continue
            mean = sum(first[i][4] for i in range(t - 14, t)) / 14
            if mean > 0 and first[t][4] / mean >= chosen.min_rel_volume:
                candidates.append((-first[t][4] / mean, name, atr))
        for rank, (negative, name, atr) in enumerate(sorted(candidates)[:chosen.top_names], 1):
            open_, high, low, close, _ = opening[name][t]
            direction = int(np.sign(close - open_))
            trigger = high if direction > 0 else low if direction < 0 else np.nan
            prior = bars[name][t - 1][3]
            restricted = bars[name][t - 1][2] <= (1 - chosen.ssr_decline) * bars[name][t - 2][3]
            rows.append(dict(session=sessions[t], symbol=name, rank=rank, rel_volume=-negative, direction=direction, trigger=trigger,
                             atr=atr, stop_distance=chosen.stop_atr_fraction * atr, prior_close=prior,
                             short_ok=bool(direction < 0 and name not in not_shortable and not restricted
                                           and trigger > (1 - chosen.ssr_decline) * prior), or_open=open_, or_close=close))
    return pd.DataFrame(rows, columns=trial.SELECTION_COLUMNS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_selection_equals_a_brute_force_reference_on_gappy_multi_symbol_data(seed):
    chosen = trial.contract_spec(json.loads(REPO_CONTRACT.read_text()))
    sessions = SESSIONS[:90]
    names, bars, opening, splits = _gappy_world(seed)
    rng = np.random.default_rng(seed + 1000)
    not_shortable = frozenset(rng.choice(names, size=8, replace=False).tolist())
    table = lambda source: pd.DataFrame([dict(session=sessions[i], symbol=name, open=b[0], high=b[1], low=b[2], close=b[3], volume=b[4])  # noqa: E731
                                         for name in names for i, b in source[name].items()])
    daily_frame, opening_frame = table(bars), table(opening)
    ordered = sorted(names)
    features = pd.concat([trial.daily_features(daily_frame[daily_frame.symbol.isin(ordered[k:k + 7])].sort_values(["symbol", "session"]),
                                               sessions, [e for e in splits if e[0] in ordered[k:k + 7]], chosen)
                          for k in range(0, len(ordered), 7)], ignore_index=True)  # in symbol chunks, as acquisition does
    pieces = [piece.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).astype({"session": "category", "symbol": "category"})
              for _, piece in opening_frame.groupby(opening_frame.session.str[:7], sort=True)]  # monthly, shuffled, own categories
    got = trial.select_from_parts(features, lambda: iter(pieces), sessions, not_shortable, chosen).sort_values(["session", "rank"], ignore_index=True)
    want = _reference_selection(sessions, names, bars, opening, splits, not_shortable, chosen).sort_values(["session", "rank"], ignore_index=True)
    assert len(got) == len(want) > 300 and {"NA", "NAN", "NULL"} & set(got.symbol)
    for column in ("session", "symbol", "rank", "direction", "short_ok"):
        assert got[column].astype(str).tolist() == want[column].astype(str).tolist(), column
    for column in ("rel_volume", "trigger", "atr", "stop_distance", "prior_close", "or_open", "or_close"):
        np.testing.assert_allclose(got[column].to_numpy(dtype=float), want[column].to_numpy(dtype=float), rtol=1e-12, equal_nan=True, err_msg=column)
    assert not {"ZEXA", "AEXA", "MEXA"} & set(got.symbol)  # 14 bars are history for a 15th that never comes
