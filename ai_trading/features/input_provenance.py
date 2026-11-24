"""Versioned, non-authoritative evidence for actual inference inputs."""
from __future__ import annotations

import hashlib
import json
from typing import Any
from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION

VERSION = 'day_input_v1'


def frame_identity(frame: Any) -> str:
    """Hash ordered index, columns and values; changing history changes identity."""
    payload = frame.to_json(orient='split', date_format='iso', date_unit='ns', double_precision=15)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def describe_batch(frame: Any, *, requested_start: Any = None,
                   requested_end: Any = None, grace_seconds: float = 2.0,
                   rth_only: bool = True) -> dict[str, Any]:
    if frame is None:
        return {'version': VERSION, 'row_count': 0, 'status': 'no_batch',
                'input_sha256': None, 'qualification_authority': False}
    attrs = frame.attrs
    evidence = {key: attrs.get(key) for key in (
        'data_provider', 'data_feed', 'fallback_provider', 'fallback_feed',
        'raw_payload_provider', 'raw_payload_feed', 'reference_feed_effective',
        'requested_feed', 'requested_adjustment', 'effective_adjustment', 'adjustment_evidence_basis')}
    contract = {'version': VERSION,
                'feed': attrs.get('data_feed') or attrs.get('reference_feed_effective'),
                'adjustment': attrs.get('effective_adjustment'), 'timeframe': '5Min',
                'session_policy': 'canonical_exchange_regular_session_v1' if rth_only else 'extended_sessions',
                'history_policy': attrs.get('history_policy'),
                'finality_policy': {'bar_label': 'start', 'grace_seconds': grace_seconds},
                'feature_version': DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION}
    return {'version': VERSION, 'row_count': len(frame), 'input_contract': contract,
            'first_bar_start': frame.index[0].isoformat() if len(frame) else None,
            'last_bar_start': frame.index[-1].isoformat() if len(frame) else None,
            'requested_start': requested_start.isoformat() if requested_start is not None else None,
            'requested_end': requested_end.isoformat() if requested_end is not None else None,
            'input_sha256': frame_identity(frame),
            'hash_format': 'pandas_split_json_iso_ns_15digits_v1',
            'session_policy': contract['session_policy'],
            'finality_grace_seconds': grace_seconds, 'timeframe': '5Min',
            'source_evidence': json.loads(json.dumps(evidence, default=str)),
            'effective_adjustment_verified': attrs.get('effective_adjustment') is not None,
            'qualification_authority': False}
