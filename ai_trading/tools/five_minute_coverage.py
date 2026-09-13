"""Timestamp-only development audit; no returns, signals or campaign claims."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.research_feasibility import audit_sampling_acquisition
from ai_trading.utils.market_calendar import is_trading_day, session_info


ROOT = Path(__file__).resolve().parents[2]


def count_session(index, opening, closing):
    """Require unique minute bars across inputs, entry, holding and exit."""
    counts = pd.Series(index).value_counts()
    decisions = pd.date_range(opening + pd.Timedelta(minutes=5), closing,
                              freq='5min', inclusive='left')
    result = dict(candidate_slots=0, session_boundary_excluded=0,
                  missing_slots=0, duplicate_slots=0, complete_slots=0)
    for decision in decisions:
        result['candidate_slots'] += 1
        exit_ts = decision + pd.Timedelta(minutes=6)
        if exit_ts >= closing:
            result['session_boundary_excluded'] += 1
            continue
        required = pd.date_range(decision - pd.Timedelta(minutes=5), exit_ts, freq='min')
        values = counts.reindex(required, fill_value=0)
        if (values > 1).any():
            result['duplicate_slots'] += 1
        elif (values == 0).any():
            result['missing_slots'] += 1
        else:
            result['complete_slots'] += 1
    assert result['candidate_slots'] == sum(result[k] for k in result if k != 'candidate_slots')
    return result


def main():
    protocol = json.loads((ROOT / 'config/slower_horizon_campaign.json').read_text())
    # Reuse canonical source identity, hash, quality and holdout guards only.
    # This does not execute the campaign evaluator or claim its budget.
    acquisition = ROOT / 'artifacts/research_foundation/acquisition_sip_repaired.json'
    provenance = audit_sampling_acquisition(protocol, acquisition)
    ledgers = [ROOT / f'artifacts/{name}/campaign_state.json'
               for name in ['research_reset', 'research_foundation']]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ledgers}
    sessions = []
    for symbol, source in provenance['sources'].items():
        path = Path(source['path'])
        raw_before = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_before != source['sha256']:
            raise ValueError('source changed after provenance audit')
        frame = pd.read_csv(path, usecols=['timestamp'], dtype=str)
        index = pd.DatetimeIndex(pd.to_datetime(frame['timestamp'], utc=True))
        if hashlib.sha256(path.read_bytes()).hexdigest() != raw_before:
            raise ValueError('source changed during timestamp read')
        if index.min() < pd.Timestamp('2024-01-01', tz='UTC') or index.max() >= pd.Timestamp('2026-01-01', tz='UTC'):
            raise ValueError('actual timestamps outside development interval')
        for day in pd.date_range('2024-01-01', '2025-12-31'):
            if not is_trading_day(day.date()):
                continue
            session = session_info(day.date())
            subset = index[(index >= session.start_utc) & (index < session.end_utc)]
            sessions.append({'symbol': symbol, 'session': day.date().isoformat(),
                             **count_session(subset, session.start_utc, session.end_utc)})
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ledgers}
    assert before == after
    keys = ['candidate_slots', 'session_boundary_excluded', 'missing_slots', 'duplicate_slots', 'complete_slots']
    totals = {symbol: {key: sum(row[key] for row in sessions if row['symbol'] == symbol)
                       for key in keys} for symbol in protocol['symbols']}
    report = dict(scope='timestamp_coverage_upper_bound_not_strategy_eligibility',
                  decision='completed_five_minute_input', entry_delay_seconds=60,
                  holding_seconds=300, exit_after_decision_seconds=360,
                  required_minutes='decision_minus_5_through_decision_plus_6_inclusive',
                  prices_used=False, returns_computed=False, trial_claimed=False,
                  promotion_authority=False, holdout_evaluated=False,
                  current_model_universe_coverage='unavailable_in_this_etf_dataset',
                  sources=provenance['sources'], acquisition_sha256=provenance['acquisition_sha256'],
                  manifest_sha256=provenance['manifest_sha256'], ledger_hashes=after,
                  totals=totals, sessions=sessions)
    atomic_write_text(ROOT / 'artifacts/research_reset/five_minute_coverage.json',
                      json.dumps(report, indent=2, sort_keys=True) + '\n')
    get_logger(__name__).info('FIVE_MINUTE_COVERAGE_AUDITED', extra={'totals': totals})


if __name__ == '__main__':
    main()
