"""Summarize local Codex telemetry without exporting prompts or tool bodies."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any


def audit_file(path: Path) -> dict[str, Any]:
    usage: Counter[str] = Counter()
    efforts: Counter[str] = Counter()
    reads: Counter[str] = Counter()
    seen = set()
    requests: list[int] = []
    outputs = polls = empty_polls = large_outputs = output_chars = malformed = 0
    poll_ids = set()
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            payload = record.get('payload') or {}
            kind = record.get('type')
            if kind == 'token_usage_record':
                key = payload.get('response_id')
                if not key or key in seen:
                    continue
                seen.add(key)
                values = payload.get('usage') or {}
                for name in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens'):
                    usage[name] += int(values.get(name, 0))
                requests.append(int(values.get('input_tokens', 0)))
            elif kind == 'turn_context':
                efforts[f"{payload.get('model')}:{payload.get('effort')}"] += 1
            elif kind == 'response_item' and payload.get('type') in ('custom_tool_call', 'function_call'):
                body = str(payload.get('input') or payload.get('arguments') or '')
                if 'write_stdin' in body or payload.get('name') == 'write_stdin':
                    polls += 1
                    poll_ids.add(payload.get('call_id'))
                for match in re.finditer(r'cmd\s*:\s*"((?:\\.|[^"\\])*)"', body):
                    command = match.group(1)
                    if command.startswith(('rg ', 'cat ', 'sed ', 'tail ', 'head ')):
                        reads[hashlib.sha256(command.encode()).hexdigest()] += 1
            elif kind == 'response_item' and payload.get('type') in ('custom_tool_call_output', 'function_call_output'):
                raw = payload.get('output', '')
                texts = [str(v.get('text', '')) for v in raw if isinstance(v, dict)] if isinstance(raw, list) else [str(raw)]
                body = '\n'.join(texts)
                outputs += 1
                output_chars += len(body)
                large_outputs += int(len(body) > 16000)
                decoded = []
                for text in texts:
                    try:
                        value = json.loads(text)
                        if isinstance(value, dict) and 'output' in value:
                            decoded.append(value)
                    except ValueError:
                        pass
                if payload.get('call_id') in poll_ids and decoded and all(v['output'] == '' for v in decoded):
                    empty_polls += 1
    return {'session_file': path.name, 'usage_source': 'deduplicated_per_response_token_usage_record' if requests else 'unavailable', 'responses': len(requests), 'tokens': dict(usage), 'uncached_input_tokens': usage['input_tokens'] - usage['cached_input_tokens'], 'cached_input_ratio': usage['cached_input_tokens'] / usage['input_tokens'] if usage['input_tokens'] else None, 'median_input_tokens_per_response': median(requests) if requests else None, 'max_input_tokens_per_response': max(requests) if requests else None, 'model_effort_contexts': dict(efforts), 'tool_outputs': outputs, 'tool_output_characters': output_chars, 'tool_outputs_over_16000_characters': large_outputs, 'poll_calls': polls, 'parseable_empty_poll_calls': empty_polls, 'exact_repeated_read_calls': sum(n - 1 for n in reads.values()), 'malformed_lines': malformed}


def belongs_to(path: Path, cwd: str) -> bool:
    with path.open() as stream:
        for _, line in zip(range(30), stream):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('type') == 'session_meta' and isinstance(row.get('payload', {}).get('source'), dict):
                return False  # Separate automatic review/subagent telemetry from user tasks.
            if row.get('type') in ('session_meta', 'turn_context') and row.get('payload', {}).get('cwd') == cwd:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=Path, default=Path.home() / '.codex/sessions')
    parser.add_argument('--cwd', default=str(Path.cwd()))
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error('--limit must be between 1 and 20')
    paths = sorted(args.sessions.rglob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
    selected = []
    for path in paths:
        if belongs_to(path, args.cwd):
            selected.append(path)
        if len(selected) == args.limit:
            break
    report: dict[str, Any] = {'generated_at': datetime.now(UTC).isoformat(), 'sessions': [audit_file(p) for p in selected], 'limitations': ['Tokens are telemetry, not billed dollars or subscription-limit units.', 'Cached input is included in input totals; reasoning is included in output totals.', 'Tool-output sizes are characters, not tokenizer counts.', 'Empty polls and repeated reads are conservative detectable counts, not proof of waste.', 'No prompt bodies, command text, credentials or reasoning text are exported.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.output), 'sessions': len(selected), 'responses': sum(s['responses'] for s in report['sessions'])}))


if __name__ == '__main__':
    main()
