from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import swing_reversal_trial as trial
from ai_trading.tools.research_feasibility import daily_development_dates

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/swing_reversal_trial_campaign.json"
CAMPAIGN = "swing_reversal_trial_test"
NAMES = [f"S{chr(65 + i)}" for i in range(16)]
SECTORS = {name: ("Tech" if i % 2 else "Energy") for i, name in enumerate(NAMES)}


def small_contract(**changes):
    payload = json.loads(REPO_CONTRACT.read_text())
    payload.update(
        campaign_id=CAMPAIGN, status="registered", development_start="2024-01-02", development_end="2024-12-31",
        acquisition_manifest=f"artifacts/{CAMPAIGN}/data/acquisition.json",
        universe=payload["universe"] | {"size": 12},
        signal=payload["signal"] | {"minimum_group_size": 3},
        portfolio=payload["portfolio"] | {"slice_names": 3, "entry_quantile": 0.2, "hold_quantile": 0.5},
        survivorship=payload["survivorship"] | {"probe_status": "delisted_symbols_missing_or_partial"},
        support_thresholds={"minimum_periods": 20, "minimum_supported_years": 1, "minimum_periods_per_year": 20,
                            "minimum_eligible_names": 8},
        acceptance=payload["acceptance"] | {"minimum_positive_years": 1},
        bootstrap={"unit": "circular_blocks_of_decision_periods", "block_periods": 2, "replicates": 100, "seed": 7,
                   "lower_quantile": 0.0125})
    return payload | changes


def spec(**changes):
    return trial.contract_spec(small_contract(**changes))


def test_repository_contract_is_an_unregistered_draft_with_the_written_rules(tmp_path):
    draft = json.loads(REPO_CONTRACT.read_text())
    assert draft["status"] == "draft_not_registered" and draft["survivorship"]["probe_status"] == "not_run"
    reset = json.loads((REPO_CONTRACT.parents[1] / trial.RESET_CONFIG).read_text())
    assert draft["campaign_id"] not in reset["registered_trials"]
    with pytest.raises(ValueError, match="probe_status"):
        trial.contract_spec(draft)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="not registered"):
        trial.run(tmp_path, mode="acquire", contract_path=Path("config/draft.json"), client_factory=lambda: pytest.fail("fetched"))
    full = trial.contract_spec(draft | {"survivorship": draft["survivorship"] | {"probe_status": "delisted_symbols_served"}})
    assert (full.universe_size, full.volume_months, full.min_price, full.exchanges) == (300, 3, 5.0, ("NYSE", "NASDAQ", "AMEX"))
    assert (full.lookback, full.min_group, full.slots, full.entry_quantile, full.hold_quantile) == (5, 5, 30, 0.1, 0.5)
    assert (full.every, full.first_decision, full.vol_lookback, full.reference_vol) == (5, 65, 21, 0.2)
    assert (full.exposure_floor, full.exposure_cap, full.exposure_band) == (0.25, 1.0, 0.1)
    assert (full.htb_fraction, full.adverse_return, full.lower_quantile, full.min_positive_years) == (0.1, 0.3, 0.0125, 6)
    assert [(n, c, b) for n, c, b, _ in full.tiers] == [("primary", 5.0, 100.0), ("stress_high", 10.0, 300.0), ("measured_paper_fills", 2.0, 100.0)]
    assert full.tier("decision")[0] == "primary" and full.tier(trial.STRESS_ROLE)[0] == "stress_high"
    assert set(draft["parameter_justifications"]) and list(draft["versions"]) == ["long_only", "long_short"]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"development_end": "2026-01-05"}, "holdout"),
        ({"development_end": "2026-09-10"}, "holdout"),
        ({"adjustment": "split"}, "adjustment=all"),
        ({"runtime_authority": True}, "authority"),
        ({"versions": {"long_short": "", "long_only": ""}}, "versions"),
        ({"rebalance_grid": {"every_sessions": 5, "first_decision_index": 10}}, "lookback"),
        ({"rebalance_grid": {"every_sessions": 3, "first_decision_index": 65}}, "hold_sessions"),
        ({"cost_tiers": [{"name": "measured", "per_side_bps": None, "borrow_annual_bps": 100, "role": "reported_only"}]}, "placeholder"),
        ({"cost_tiers": [{"name": "primary", "per_side_bps": 5, "borrow_annual_bps": 100, "role": "decision"}]}, "stress tier"),
        ({"primary_per_side_cost_bps": 2}, "primary_per_side_cost_bps"),
    ],
)
def test_contract_refusals_happen_before_any_data(changes, match):
    with pytest.raises(ValueError, match=match):
        trial.contract_spec(small_contract(**changes))
    disabled = small_contract()
    disabled["acceptance"] = disabled["acceptance"] | {"positive_under_survivorship_stress": False}
    with pytest.raises(ValueError, match="disabled"):
        trial.contract_spec(disabled)


# --- universe ---------------------------------------------------------------------------------------

def _monthly(months, volumes, closes=None):
    rows = [dict(symbol=s, month=m, close=(closes or {}).get((s, m), 20.0), dollar_volume=float(v))
            for m in months for s, v in volumes.items() if v is not None]
    return pd.DataFrame(rows)


def test_universe_uses_only_completed_earlier_months_and_raw_price():
    s = spec()
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    volumes = {name: 1000 - 10 * i for i, name in enumerate(NAMES)}
    monthly = _monthly(months, volumes, closes={("SA", "2024-03"): 4.99})
    universe = trial.build_universe(monthly, months, s)
    assert universe["2024-01"] == [] and universe["2024-03"] == []  # fewer than three completed months
    assert universe["2024-04"] == NAMES[1:13]  # SA is below the price floor at the end of March
    assert universe["2024-05"] == NAMES[:12]  # its April close is back above the floor
    # Point in time: rewriting April and May cannot change April's members.
    later = monthly.copy()
    later.loc[later.month >= "2024-04", "dollar_volume"] = later.loc[later.month >= "2024-04", "dollar_volume"].to_numpy()[::-1]
    later.loc[later.month >= "2024-04", "close"] = 1.0
    assert trial.build_universe(later, months, s)["2024-04"] == universe["2024-04"]
    # A symbol missing one of the three months does not qualify.
    gap = monthly[~((monthly.symbol == "SB") & (monthly.month == "2024-02"))]
    assert "SB" not in trial.build_universe(gap, months, s)["2024-04"]
    assert trial.previous_months("2024-02", 3) == ["2023-11", "2023-12", "2024-01"]


# --- signal and selection ---------------------------------------------------------------------------

def _tables(n=140, seed=3, names=NAMES):
    dates = daily_development_dates(small_contract())[:n]
    rng = np.random.default_rng(seed)
    closes = pd.DataFrame({name: 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n))) for name in names})
    closes["SPY"] = 400 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = closes.shift(1).fillna(closes.iloc[0]) * (1 + rng.normal(0, 0.003, closes.shape))
    return opens, closes, dates


def _universe(dates, names=NAMES):
    return {month: list(names) for month in sorted({d.isoformat()[:7] for d in dates})}


def test_ranks_are_sector_relative_and_use_two_closes_only():
    s = spec()
    opens, closes, _ = _tables()
    d = 70
    flat = closes.copy()
    flat.iloc[d - 5:d + 1] = 100.0
    energy, tech = [n for n in NAMES if SECTORS[n] == "Energy"], [n for n in NAMES if SECTORS[n] == "Tech"]
    flat.loc[d, energy] = 90.0  # the whole sector fell 10 percent: nobody is a relative loser
    flat.loc[d, "SB"] = 97.0  # one tech name fell 3 percent against a flat sector
    ranks = trial.relative_ranks(flat, d, NAMES, SECTORS, s)
    assert ranks.idxmin() == "SB" and ranks["SB"] == 0.0
    assert ranks[energy].min() > ranks["SB"] and set(ranks.index) == set(NAMES)
    # Unmapped names and sectors below the minimum size are measured against the whole universe.
    lonely = trial.relative_ranks(flat, d, NAMES, {**{n: "Tech" for n in tech}, "SA": "Energy"}, s)
    assert lonely[energy].max() < lonely[[n for n in tech if n != "SB"]].min()
    # Only the closes at d and d-5 matter.
    base = trial.relative_ranks(closes, d, NAMES, SECTORS, s)
    changed = closes.copy()
    changed.iloc[d + 1:] *= 5
    changed.iloc[d - 4:d] *= 0.5
    changed.iloc[:d - 5] *= 2
    pd.testing.assert_series_equal(base, trial.relative_ranks(changed, d, NAMES, SECTORS, s))
    assert "SD" not in trial.relative_ranks(closes.assign(SD=np.nan), d, NAMES, SECTORS, s).index


def test_hold_band_keeps_names_inside_it_and_fills_from_the_extremes():
    s = spec()  # 3 slots, enter at 0.2, hold to 0.5
    ranks = pd.Series({"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.5, "F": 0.6, "G": 0.8, "H": 0.9, "I": 1.0})
    assert trial.select(ranks, [], s, side="long") == ["A", "B", "C"]
    assert trial.select(ranks, ["E", "D"], s, side="long") == ["D", "E", "A"]  # held names inside the band stay, untraded
    assert trial.select(ranks, ["F", "G"], s, side="long") == ["A", "B", "C"]  # outside the band: sold and replaced
    assert trial.select(ranks, ["Z"], s, side="long") == ["A", "B", "C"]  # left the universe
    assert trial.select(ranks.iloc[3:], [], s, side="long") == []  # nobody inside the entry slice: slots stay cash
    assert trial.select(ranks, [], s, side="short") == ["I", "H", "G"]
    assert trial.select(ranks, ["E"], s, side="short", blocked=frozenset({"I"})) == ["E", "H", "G"]
    assert trial.select(ranks, ["I"], s, side="short", blocked=frozenset({"I"})) == ["I", "H", "G"]  # held shorts are not forced out


def test_exposure_is_linear_in_trailing_market_volatility_with_floor_and_cap():
    s = spec()
    calm = 400 * np.exp(np.cumsum(np.tile([0.001, -0.001], 40)))
    sigma, target = trial.target_exposure(calm, 60, s)
    assert sigma == pytest.approx(np.std(np.diff(np.log(calm[39:61])), ddof=1) * np.sqrt(252)) and target == 0.25
    wild = 400 * np.exp(np.cumsum(np.tile([0.03, -0.03], 40)))
    assert trial.target_exposure(wild, 60, s)[1] == 1.0
    mid = 400 * np.exp(np.cumsum(np.tile([0.006, -0.006], 40)))
    sigma, target = trial.target_exposure(mid, 60, s)
    assert 0.25 < target < 1.0 and target == pytest.approx(sigma / 0.2)
    later = mid.copy()
    later[61:] *= 3
    assert trial.target_exposure(later, 60, s) == (sigma, target)


# --- simulation -------------------------------------------------------------------------------------

def _simulate(opens, closes, dates, s, **kwargs):
    kwargs.setdefault("version", "long_short")
    return trial.simulate(opens, closes, dates, kwargs.pop("universe", _universe(dates)), kwargs.pop("sectors", SECTORS),
                          kwargs.pop("not_shortable", frozenset()), s, **kwargs)


def test_entry_is_the_next_open_and_future_prices_cannot_change_past_decisions():
    s = spec()
    opens, closes, dates = _tables()
    frame, positions = _simulate(opens, closes, dates, s)
    trial.validate_periods(frame, small_contract(), s)
    assert (frame.entry_index == frame.decision_index + 1).all() and (frame.exit_index == frame.entry_index + 5).all()
    first = positions[positions.decision_index == frame.decision_index.iloc[0]]
    d = int(frame.decision_index.iloc[0])
    for row in first.itertuples():
        assert row.position_return == pytest.approx(opens[row.symbol].iat[d + 6] / opens[row.symbol].iat[d + 1] - 1)
    assert set(first.side) == {"long", "short"} and (first.groupby("side").size() == 3).all() and not first.kept.any()
    ranks = trial.relative_ranks(closes, d, NAMES, SECTORS, s)
    assert set(first[first.side == "long"].symbol) == set(ranks.nsmallest(3).index)
    # Shift every price after the entry open of period k: decisions up to and including k are unchanged.
    k = 6
    cut = int(frame.entry_index.iloc[k])
    shifted_opens, shifted_closes = opens.copy(), closes.copy()
    shifted_closes.iloc[cut:] *= np.linspace(0.5, 2.0, closes.shape[1])
    shifted_opens.iloc[cut + 1:] *= np.linspace(2.0, 0.5, opens.shape[1])
    changed_frame, changed = _simulate(shifted_opens, shifted_closes, dates, s)
    keys = ["decision_index", "symbol", "side", "weight", "kept", "rank"]
    limit = frame.decision_index.iloc[k]
    pd.testing.assert_frame_equal(positions[positions.decision_index <= limit][keys].reset_index(drop=True),
                                  changed[changed.decision_index <= limit][keys].reset_index(drop=True))
    assert changed_frame.gross_return.iloc[:k].tolist() == frame.gross_return.iloc[:k].tolist()
    assert changed_frame.traded.iloc[:k + 1].tolist() == frame.traded.iloc[:k + 1].tolist()
    assert not changed[keys].equals(positions[keys])  # and later decisions do respond
    # The decision never sees the entry open: only the entry session's own open changing leaves selection alone.
    gap = opens.copy()
    gap.iloc[d + 1] *= 1.05
    assert _simulate(gap, closes, dates, s)[1].query("decision_index == @d").symbol.tolist() == first.symbol.tolist()


def test_traded_weight_charges_entries_keeps_exits_and_exposure_changes():
    s = spec()
    opens, closes, dates = _tables()
    frame, positions = _simulate(opens, closes, dates, s, version="long_only", conditioned=False)
    slot = 1 / 3
    assert frame.traded.iloc[0] == pytest.approx(3 * slot) and (frame.exposure == 1.0).all() and (frame.short_weight == 0).all()
    # Rebuild the second period's traded weight by hand from drifted weights.
    d0, d1 = (int(v) for v in frame.decision_index.iloc[:2])
    before = positions[positions.decision_index == d0].set_index("symbol")
    after = positions[positions.decision_index == d1].set_index("symbol")
    drifted = slot * (1 + before.position_return) / (1 + frame.gross_return.iloc[0])
    expected = sum(abs((slot if name in after.index else 0.0) - drifted.get(name, 0.0)) for name in set(before.index) | set(after.index))
    assert frame.traded.iloc[1] == pytest.approx(expected)
    assert after.kept.tolist() == [name in before.index for name in after.index]
    assert frame.kept_names.iloc[1] == int(after.kept.sum())
    assert frame.traded.iloc[-1] > frame.traded.iloc[1:-1].max() * 0.5  # the final liquidation is charged
    # The conditioned version resizes positions only when exposure moves by the band, and that is charged too.
    conditioned, _ = _simulate(opens, closes, dates, s, version="long_only")
    assert conditioned.exposure.between(0.25, 1.0).all()
    assert ((conditioned.exposure.diff().abs().dropna() == 0) | (conditioned.exposure.diff().abs().dropna() >= 0.1 - 1e-12)).all()
    assert (conditioned.long_weight <= conditioned.exposure + 1e-12).all()
    both, _ = _simulate(opens, closes, dates, s)
    assert (both.short_weight > 0).all() and both.traded.iloc[0] == pytest.approx(2 * both.exposure.iloc[0])


def test_cost_borrow_and_survivorship_stress_accounting():
    s = spec()
    frame = pd.DataFrame({"gross_return": [0.01, -0.02], "traded": [2.0, 0.5], "short_weight": [1.0, 0.5], "long_weight": [1.0, 0.75],
                          "market_return": [0.02, -0.01], "terminal_weight": [0.0, 0.25], "hold_sessions": [5, 5]})
    for _, per_side, borrow, _ in s.tiers:
        nets = trial.tier_returns(frame, per_side, borrow, s)
        expected = [0.01 - per_side / 1e4 * 2.0 - borrow / 1e4 * 1.0 * 5 / 252, -0.02 - per_side / 1e4 * 0.5 - borrow / 1e4 * 0.5 * 5 / 252]
        assert nets["net"] == pytest.approx(expected)
        assert nets["matched_market"] == pytest.approx([0.0, 0.25 * -0.01])
        stressed = trial.tier_returns(frame, per_side, borrow, s, survivorship_stress=True)
        assert stressed["net"] == pytest.approx([expected[0], expected[1] - 0.3 * 0.25])
    assert trial.tier_returns(frame, 0, 0, s)["net"] == pytest.approx(frame.gross_return)


def test_data_ending_mid_hold_forces_an_exit_and_feeds_the_survivorship_stress():
    s = spec()
    opens, closes, dates = _tables()
    frame, positions = _simulate(opens, closes, dates, s, version="long_only", conditioned=False)
    d = int(frame.decision_index.iloc[2])
    victim = positions[(positions.decision_index == d) & (positions.side == "long")].symbol.iloc[0]
    dead_opens, dead_closes = opens.copy(), closes.copy()
    dead_opens.loc[d + 4:, victim] = np.nan  # the last bar is session d+3, inside the hold that exits at d+6
    dead_closes.loc[d + 4:, victim] = np.nan
    dead_frame, dead = _simulate(dead_opens, dead_closes, dates, s, version="long_only", conditioned=False)
    row = dead[(dead.decision_index == d) & (dead.symbol == victim)].iloc[0]
    assert row.forced_exit and row.position_return == pytest.approx(closes[victim].iat[d + 3] / opens[victim].iat[d + 1] - 1)
    period = dead_frame[dead_frame.decision_index == d].iloc[0]
    assert period.terminal_weight == pytest.approx(1 / 3) and dead_frame.terminal_weight.sum() == pytest.approx(1 / 3)
    assert victim not in dead[dead.decision_index > d].symbol.tolist()
    base = trial.tier_returns(dead_frame, 5, 100, s)["net"]
    stressed = trial.tier_returns(dead_frame, 5, 100, s, survivorship_stress=True)["net"]
    assert (base - stressed).sum() == pytest.approx(0.3 / 3)
    # A halt (bars resume later) forces the exit but is not a data end.
    halted_opens = opens.copy()
    halted_opens.loc[d + 6, victim] = np.nan
    halted_frame, halted = _simulate(halted_opens, closes, dates, s, version="long_only", conditioned=False)
    assert halted[(halted.decision_index == d) & (halted.symbol == victim)].forced_exit.iloc[0] and halted_frame.terminal_weight.sum() == 0
    # No bar on the entry session: the order does not fill and the slot stays in cash.
    first = int(frame.decision_index.iloc[0])
    fresh = positions[positions.decision_index == first].symbol.iloc[0]
    unfilled = opens.copy()
    unfilled.loc[first + 1, fresh] = np.nan
    unfilled_frame, unfilled_positions = _simulate(unfilled, closes, dates, s, version="long_only", conditioned=False)
    assert fresh not in unfilled_positions[unfilled_positions.decision_index == first].symbol.tolist()
    assert unfilled_frame.names_long.iloc[0] == 2 and unfilled_frame.long_weight.iloc[0] == pytest.approx(2 / 3)
    assert unfilled_frame.traded.iloc[0] == pytest.approx(2 / 3)


def test_shorts_skip_the_hard_to_borrow_proxy_and_membership_is_point_in_time():
    s = spec()
    opens, closes, dates = _tables()
    wild = closes.copy()
    wild["SF"] = 50 * np.exp(np.cumsum(np.tile([0.2, -0.19], len(closes) // 2)))  # by far the most volatile name
    assert "SF" in trial.high_volatility_names(wild, 70, NAMES, s)
    _, positions = _simulate(opens, wild, dates, s, not_shortable=frozenset({"SH"}))
    shorts = positions[positions.side == "short"]
    assert not shorts[~shorts.kept].symbol.isin(["SF", "SH"]).any()
    assert positions[positions.side == "long"].symbol.isin(["SF", "SH"]).any()
    # A name is tradable only in months whose frozen universe lists it.
    universe = _universe(dates)
    months = sorted(universe)
    universe[months[-1]] = [n for n in NAMES if n != "SC"]
    _, limited = _simulate(opens, closes, dates, s, universe=universe)
    last_month_decisions = [i for i in limited.decision_index.unique() if dates[i].isoformat()[:7] == months[-1]]
    assert last_month_decisions and "SC" not in limited[limited.decision_index.isin(last_month_decisions)].symbol.tolist()
    assert "SC" in limited[~limited.decision_index.isin(last_month_decisions)].symbol.tolist()


def test_evaluate_scores_both_versions_and_cannot_pass_on_noise_without_support():
    s = spec()
    opens, closes, dates = _tables(n=252)
    frames = {}
    for version in trial.VERSIONS:
        frames[version] = _simulate(opens, closes, dates, s, version=version)[0]
        frames[f"{version}_unconditioned"] = _simulate(opens, closes, dates, s, version=version, conditioned=False)[0]
    report = trial.evaluate(frames, s)
    assert report["decision"] == report["versions"]["long_only"]["decision"]
    assert report["secondary_decision"] == report["versions"]["long_short"]["decision"]
    long_only = report["versions"]["long_only"]
    assert set(long_only["cost_tiers"]) == {"primary", "stress_high", "measured_paper_fills"}
    assert long_only["cost_tiers"]["stress_high"]["net_per_period_bps"] < long_only["cost_tiers"]["primary"]["net_per_period_bps"]
    assert "positive_lower_bound_vs_exposure_matched_market_at_primary" in long_only["criteria"]
    assert "positive_lower_bound_vs_exposure_matched_market_at_primary" not in report["versions"]["long_short"]["criteria"]
    assert long_only["benchmark"] == "matched_market" and report["versions"]["long_short"]["benchmark"] == "cash"
    assert "decision" not in report["reported_only_unconditioned"]["long_only"]
    assert long_only["annual_one_way_turnover"] > 5 and 0 < long_only["share_of_names_kept"] < 1
    for version in report["versions"].values():
        assert version["decision"] == ("eligible_for_untouched_test_planning" if all(version["criteria"].values()) else "retired")
    json.dumps(report, allow_nan=False)
    short = {k: v.iloc[:3] for k, v in frames.items()}
    assert trial.evaluate(short, s)["decision"] == "inconclusive_budget_consumed"


# --- end-to-end with fake clients; synthetic prices only --------------------------------------------

class _FakeTrading:
    def __init__(self):
        self.calls = []

    def get_all_assets(self, request):
        self.calls.append(request.status.value)
        asset = lambda symbol, name, exchange, shortable=True: SimpleNamespace(  # noqa: E731
            symbol=symbol, name=name, exchange=SimpleNamespace(value=exchange), shortable=shortable, easy_to_borrow=shortable)
        if request.status.value == "inactive":
            return [asset("DEAD", "Dead Corp Common Stock", "NYSE", False)]
        return ([asset(name, f"{name} Inc. Common Stock", "NASDAQ" if i % 2 else "NYSE") for i, name in enumerate(NAMES)]
                + [asset("QQQX", "Big Index ETF", "NASDAQ"), asset("ARCX", "Arca Listed Corp", "ARCA"),
                   asset("PENNY", "Penny Corp", "NYSE"), asset("ADRX", "Foreign Co American Depositary Shares", "NYSE"),
                   asset("SPY", "SPDR S&P 500 ETF Trust", "ARCA")])


class _FakeData:
    def __init__(self, dates, *, extra=None):
        self.dates, self.extra, self.requests = dates, extra or [], []

    def _closes(self, symbol):
        rng = np.random.default_rng(sum(map(ord, symbol)))
        return (2.0 if symbol == "PENNY" else 60.0) * np.exp(np.cumsum(rng.normal(0.0002, 0.006 if symbol == "SPY" else 0.02, len(self.dates))))

    def get_stock_bars(self, request):
        self.requests.append(request)
        monthly = str(request.timeframe) == "1Month"
        data = {}
        for symbol in request.symbol_or_symbols:
            days = self.dates[:150] if symbol == "DEAD" else self.dates
            closes = self._closes(symbol)[:len(days)]
            volume = 5e6 if symbol in ("DEAD", "QQQX") else 1e6 + 1e4 * sum(map(ord, symbol))
            bars = [SimpleNamespace(timestamp=datetime(d.year, d.month, d.day, 5, tzinfo=UTC), open=c * 0.999, high=c * 1.01,
                                    low=c * 0.99, close=c, volume=volume, vwap=c) for d, c in zip(days, closes)]
            if monthly:
                firsts = {}
                for bar in bars:
                    firsts.setdefault((bar.timestamp.year, bar.timestamp.month), bar)
                bars = [SimpleNamespace(**{**vars(bar), "timestamp": datetime(y, m, 1, 5, tzinfo=UTC)}) for (y, m), bar in firsts.items()]
            data[symbol] = bars + [SimpleNamespace(timestamp=t, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, vwap=1.0) for t in self.extra]
        return SimpleNamespace(data=data)


def _register(root, payload=None):
    payload = payload or small_contract()
    (root / "config").mkdir(exist_ok=True)
    (root / "config/trial.json").write_text(json.dumps(payload))
    (root / "config/research_reset.json").write_text(json.dumps({"registered_trials": [CAMPAIGN]}))
    state_path = root / f"artifacts/{CAMPAIGN}/campaign_state.json"
    trial.register_campaign(state_path, payload)
    return state_path, trial.Clients(trading=_FakeTrading(), data=_FakeData(daily_development_dates(payload)))


@pytest.fixture
def registered(tmp_path):
    return (tmp_path, *_register(tmp_path))


def _run(root, mode, clients, path="config/trial.json"):
    return trial.run(root, mode=mode, contract_path=Path(path), client_factory=lambda: clients,
                     sector_lookup=lambda symbol: SECTORS.get(symbol))


def test_acquisition_freezes_a_point_in_time_universe_and_stages_claim_exactly_once(registered, monkeypatch):
    root, state_path, clients = registered
    acquired = _run(root, "acquire", clients)
    assert acquired["status"] == "acquisition_written" and acquired["trial_claimed"] is False
    counts = acquired["counts"]
    assert counts["inactive_assets"] == 1 and counts["excluded_exchange"] == 2 and counts["excluded_name"] == 2
    assert counts["ever_in_universe_inactive"] == 1 and counts["unmapped_sectors"] == 1
    monthly, daily_requests = clients.data.requests[0], clients.data.requests[-1]
    assert str(monthly.timeframe) == "1Month" and str(monthly.adjustment) == "Adjustment.RAW"
    assert str(daily_requests.timeframe) == "1Day" and str(daily_requests.adjustment) == "Adjustment.ALL"
    assert {str(r.feed) for r in clients.data.requests} == {"DataFeed.SIP"}
    data_dir = state_path.parent / "data"
    universe = json.loads((data_dir / "universe.json").read_text())
    assert universe["2024-03"] == [] and len(universe["2024-04"]) == 12
    assert "DEAD" in universe["2024-04"] and "DEAD" not in universe["2024-12"]  # the delisted name is in while it traded
    members = {s for names in universe.values() for s in names}
    assert not members & {"QQQX", "ARCX", "PENNY", "ADRX", "SPY"}
    assert sorted(p.stem for p in (data_dir / "daily").glob("*.csv")) == sorted(members | {"SPY"})
    with pytest.raises(ValueError, match="immutable"):
        _run(root, "acquire", clients)

    monkeypatch.setattr(trial, "load_prices", lambda *a: pytest.fail("prices read by the support audit"))
    audit = _run(root, "support-audit", clients)
    assert audit["status"] == trial.SUPPORT_OK and audit["prices_read_for_analysis"] is False and audit["trial_claimed"] is False
    assert audit["possible_periods"] == 37 and not (state_path.parent / "preflight.json").exists()
    monkeypatch.undo()

    preflight = _run(root, "preflight", clients)
    assert preflight["status"] == "preflight_passed" and json.loads(state_path.read_text())["trials"] == []
    assert not (state_path.parent / "report.json").exists()

    report = _run(root, "fit", clients)
    assert report["decision"] in {"retired", "eligible_for_untouched_test_planning"}
    assert report["survivorship_bias_label"] == "upward_biased" and report["holdout_evaluated"] is False and report["orders_sent"] == 0
    assert report["versions"]["long_only"]["periods_with_data_ended_positions"] >= 0
    assert len(report["output_sha256"]) == 8
    trials = json.loads(state_path.read_text())["trials"]
    assert len(trials) == 1 and trials[0]["status"] == report["decision"]
    assert trials[0]["report_sha256"] == hashlib.sha256((state_path.parent / "report.json").read_bytes()).hexdigest()
    monkeypatch.setattr(trial, "load_frozen", lambda *a: pytest.fail("data read"))
    monkeypatch.setattr(trial, "evaluate", lambda *a: pytest.fail("re-evaluated"))
    for mode in ("fit", "preflight", "support-audit", "acquire"):
        assert _run(root, mode, clients)["status"] == "budget_already_claimed"


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


def test_holdout_and_2026_rows_are_refused_at_every_stage(registered):
    root, state_path, clients = registered
    clients.data.extra = [datetime(2026, 9, 10, 5, tzinfo=UTC)]
    with pytest.raises(ValueError, match="outside development"):
        _run(root, "acquire", clients)
    assert not (state_path.parent / "data/acquisition.json").exists()
    clients.data.extra = []
    _run(root, "acquire", clients)
    manifest_path = state_path.parent / "data/acquisition.json"
    manifest = json.loads(manifest_path.read_text())
    source = root / manifest["sources"]["SB"]["path"]
    source.write_text(source.read_text() + "2026-10-01,1,1,1,1,1\n")
    manifest["sources"]["SB"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for mode in ("support-audit", "preflight", "fit"):
        with pytest.raises(ValueError, match="outside development"):
            _run(root, mode, clients)
    assert json.loads(state_path.read_text())["trials"] == [] and not (state_path.parent / "support_audit.json").exists()


def test_changed_sources_are_refused_and_a_crash_after_the_claim_consumes_the_budget(registered, monkeypatch):
    root, state_path, clients = registered
    _run(root, "acquire", clients)
    for name in ("daily/SPY.csv", "universe.json"):
        path = state_path.parent / "data" / name
        original = path.read_text()
        path.write_text(original + "\n")
        with pytest.raises(ValueError, match="source changed"):
            _run(root, "preflight", clients)
        path.write_text(original)

    def crash(*args, **kwargs):
        raise RuntimeError("simulation crashed")

    monkeypatch.setattr(trial, "simulate", crash)
    with pytest.raises(RuntimeError):
        _run(root, "fit", clients)
    assert [t["status"] for t in json.loads(state_path.read_text())["trials"]] == ["claimed"]
    assert _run(root, "fit", clients)["status"] == "budget_already_claimed"


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


def test_probe_runs_on_a_draft_and_keeps_dates_and_counts_only(tmp_path):
    draft = small_contract(status="draft_not_registered")
    draft["survivorship"] = draft["survivorship"] | {"probe_symbols": ["DEAD", "GONE"], "probe_status": "not_run"}
    (tmp_path / "config").mkdir()
    (tmp_path / "config/draft.json").write_text(json.dumps(draft))
    clients = trial.Clients(trading=_FakeTrading(), data=_FakeData(daily_development_dates(draft)))
    clients.data.get_stock_bars = lambda request, inner=clients.data.get_stock_bars: SimpleNamespace(
        data={k: v for k, v in inner(request).data.items() if k != "GONE"})
    result = _run(tmp_path, "probe", clients, path="config/draft.json")
    assert result["status"] == "probe_complete" and result["inactive_assets"] == 1 and result["trial_claimed"] is False
    assert result["results"]["DEAD"] == dict(asset_statuses=["inactive"], monthly_bars=8, first_month="2024-01",
                                             last_month="2024-08", request_error=None)
    assert result["results"]["GONE"]["monthly_bars"] == 0 and result["results"]["GONE"]["asset_statuses"] == []
    assert result["probe_symbols_with_bars"] == 1 and result["probe_symbols_listed"] == 1
    written = json.loads((tmp_path / f"artifacts/{CAMPAIGN}/survivorship_probe.json").read_text())
    assert "close" not in json.dumps(written) and not (tmp_path / f"artifacts/{CAMPAIGN}/campaign_state.json").exists()


def test_a_failing_symbol_is_retried_alone_and_recorded_not_hidden():
    dates = daily_development_dates(small_contract())
    data = _FakeData(dates)
    inner = data.get_stock_bars

    def flaky(request):
        if "BAD" in request.symbol_or_symbols:
            raise ValueError("unknown symbol")
        return inner(request)

    data.get_stock_bars = flaky
    found, failed = trial.fetch_bars(data, ["SA", "BAD", "SB"], "2024-01-02", "2024-12-31", monthly=True)
    assert sorted(found) == ["SA", "SB"] and failed == {"BAD": "ValueError"}
    assert len(found["SA"]) == 12 and set(found["SA"][0]) == {"session", "open", "high", "low", "close", "volume", "vwap"}
