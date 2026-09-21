import importlib
import json

import pytest


@pytest.mark.parametrize('name', [
    'decision_receipts_report', 'counterfactual_execution_replay_report',
    'portfolio_edge_control_report', 'symbol_lifecycle_evidence_report',
    'symbol_lifecycle_report',
])
def test_report_cli_writes_consistent_latest_and_dated_evidence(tmp_path, capsys, name):
    module = importlib.import_module('ai_trading.tools.' + name)
    output = tmp_path / 'dated/report.json'
    latest = tmp_path / 'latest/report.json'
    assert module.main(['--report-date', '2026-09-21', '--output-json', str(output),
                        '--latest-json', str(latest)]) == 0
    assert output.read_bytes() == latest.read_bytes()
    report = json.loads(output.read_text())
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {'path': str(output), 'status': report['status']}
    assert report['report_date'] == '2026-09-21'


@pytest.mark.parametrize('invalid', [False, True])
def test_reconciliation_cli_preserves_invalid_input_diagnostics(tmp_path, monkeypatch, invalid):
    from ai_trading.tools import execution_evidence_reconciliation as module
    arguments = ['reconcile']
    for name in ('decisions', 'orders', 'fills', 'tca'):
        path = tmp_path / (name + '.jsonl')
        path.write_text('malformed\n' if invalid and name == 'fills' else '')
        arguments += ['--' + name, str(path)]
    output = tmp_path / 'report.json'
    arguments += ['--output', str(output)]
    monkeypatch.setattr('sys.argv', arguments)
    module.main()
    report = json.loads(output.read_text())
    assert report['invalid_json_rows']['fills'] == int(invalid)
    assert set(report['sources']) == {'decisions', 'orders', 'fills', 'tca'}
    assert all(source['stable_during_read'] for source in report['sources'].values())
    if invalid:
        assert report['status'] == 'invalid_input_rows'


def test_dashboard_missing_evidence_is_pending_with_paper_diagnostics(tmp_path, monkeypatch):
    from ai_trading.tools import research_decision_dashboard as module
    paper = tmp_path / 'paper.json'
    paper.write_text(json.dumps({'status': 'review', 'session_date': '2026-09-21',
        'session_audit': {'accepted_unique_fills': 2, 'counts': {'missing_quote': 3}}}))
    output = tmp_path / 'dashboard.json'
    monkeypatch.setattr('sys.argv', ['dashboard', '--training-report', str(tmp_path / 'absent'),
        '--paper-review', str(paper), '--output', str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert report['status'] == 'evidence_pending'
    assert report['promotion_authority'] is False
    assert report['sources'][0]['missing'] is True
    assert report['execution_diagnostics']['accepted_fills'] == 2
    assert report['execution_diagnostics']['rejection_counts'] == {'missing_quote': 3}
    assert 'Execution diagnostics' in output.with_suffix('.html').read_text()
