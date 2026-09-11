import json
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('codex_usage_audit', Path(__file__).resolve().parents[1] / 'scripts/codex_usage_audit.py')
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
audit_file = audit_module.audit_file
belongs_to = audit_module.belongs_to


def test_usage_deduplicates_responses_and_separates_cached_tokens(tmp_path):
    path = tmp_path / 'session.jsonl'
    usage = {'type': 'token_usage_record', 'payload': {'response_id': 'r1', 'usage': {'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 10, 'reasoning_output_tokens': 4}}}
    rows = [usage, usage, {'type': 'response_item', 'payload': {'type': 'custom_tool_call', 'call_id': 'c1', 'input': 'await tools.write_stdin({session_id:1})'}}, {'type': 'response_item', 'payload': {'type': 'custom_tool_call_output', 'call_id': 'c1', 'output': [{'text': '{"output":""}'}]}}]
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    report = audit_file(path)
    assert report['responses'] == 1
    assert report['uncached_input_tokens'] == 20
    assert report['tokens']['output_tokens'] == 10
    assert report['parseable_empty_poll_calls'] == 1


def test_missing_usage_is_explicit_and_no_prompt_text_exported(tmp_path):
    path = tmp_path / 'session.jsonl'
    path.write_text(json.dumps({'type': 'response_item', 'payload': {'type': 'message', 'content': 'PRIVATE CONTENT'}}))
    report = audit_file(path)
    assert report['usage_source'] == 'unavailable'
    assert 'PRIVATE CONTENT' not in json.dumps(report)


def test_automatic_review_sessions_are_not_counted_as_user_tasks(tmp_path):
    path = tmp_path / 'review.jsonl'
    path.write_text(json.dumps({'type': 'session_meta', 'payload': {'cwd': '/project', 'source': {'subagent': {'other': 'guardian'}}}}))
    assert not belongs_to(path, '/project')
