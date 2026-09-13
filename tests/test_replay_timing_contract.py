from ai_trading.core.bot_engine import _replay_summary_metrics


def test_exact_timing_coverage_does_not_substitute_next_observation():
    order = {'id': 'a', 'submitted_at': '2026-09-08T14:00:00+00:00'}
    event = {'event_type': 'fill', 'order_id': 'a', 'symbol': 'AAPL',
             'side': 'buy', 'ts': '2026-09-08T14:01:00+00:00', 'fill_price': 100}
    rows = [{'symbol': 'AAPL', 'ts': ts, 'close': price} for ts, price in [
        ('2026-09-08T14:02:00+00:00', 101),
        ('2026-09-08T14:05:00+00:00', 102),
        ('2026-09-08T14:05:00+00:00', 102),
        ('2026-09-08T14:07:00+00:00', 103),
    ]]
    result = _replay_summary_metrics({'orders': [order], 'events': [event]}, market_rows=rows)
    diagnostic = result['timing_diagnostics']['rows'][0]
    assert diagnostic['decision_observation_status'] == 'available'
    assert diagnostic['fill_observation_status'] == 'missing_exact_observation'
    assert result['markout_observations'][0]['markout_horizon_seconds'] == 60
    assert result['timing_contract']['qualification_authority'] is False
    rows.append({'symbol': 'AAPL', 'ts': '2026-09-08T14:06:00+00:00', 'close': 103})
    exact = _replay_summary_metrics({'orders': [order], 'events': [event]}, market_rows=rows)
    assert exact['timing_diagnostics']['rows'][0]['fill_observation_status'] == 'available'
    assert exact['markout_observations'][0]['markout_horizon_seconds'] == 60
    rows.append({'symbol': 'AAPL', 'ts': '2026-09-08T14:05:00+00:00', 'close': 104})
    conflict = _replay_summary_metrics({'orders': [order], 'events': [event]}, market_rows=rows)
    assert conflict['timing_diagnostics']['rows'][0]['decision_observation_status'] == 'conflicting_exact_observations'


def test_timing_diagnostics_include_excluded_fills_and_empty_results():
    events = [{'event_type': 'fill', 'order_id': 'a', 'symbol': 'AAPL',
               'side': 'buy', 'ts': ts, 'fill_price': 100} for ts in
              ['2026-09-08T14:05:00+00:00', 'invalid']]
    result = _replay_summary_metrics({'orders': [
        {'id': 'a', 'submitted_at': '2026-09-08T14:00:00+00:00'}], 'events': events})
    diagnostics = result['timing_diagnostics']
    assert result['sample_count'] == 0
    assert diagnostics['fill_count'] == len(diagnostics['rows']) == 2
    assert diagnostics['counts']['fill_at_or_after_decision_target'] == 1
    assert diagnostics['counts']['fill_invalid_anchor'] == 1
    empty = _replay_summary_metrics({'orders': [], 'events': []})
    assert empty['timing_diagnostics'] == {'fill_count': 0, 'counts': {}, 'rows': []}
