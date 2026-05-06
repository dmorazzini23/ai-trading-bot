# Memory Telemetry And Hotspot Audit

The trading service records lightweight memory snapshots during `HEALTH_TICK`.
Samples are observational only; they do not change trading decisions.

## Runtime Samples

Default sample path:

```bash
/var/lib/ai-trading-bot/runtime/memory_samples.jsonl
```

Useful checks:

```bash
tail -n 20 /var/lib/ai-trading-bot/runtime/memory_samples.jsonl
jq -s '.[-5:]' /var/lib/ai-trading-bot/runtime/memory_samples.jsonl
```

Config knobs:

```bash
AI_TRADING_MEMORY_TELEMETRY_ENABLED=1
AI_TRADING_MEMORY_SAMPLE_PATH=runtime/memory_samples.jsonl
AI_TRADING_MEMORY_SAMPLE_MAX_BYTES=5000000
AI_TRADING_MEMORY_WARN_MB=1200
AI_TRADING_MEMORY_CRITICAL_MB=1600
```

## Hotspot Audit

Run the audit manually:

```bash
./venv/bin/python -m ai_trading.tools.memory_hotspot_audit \
  --output-json /var/lib/ai-trading-bot/runtime/memory_hotspot_audit_latest.json
```

Review the result:

```bash
jq . /var/lib/ai-trading-bot/runtime/memory_hotspot_audit_latest.json
```

The audit reports service memory from systemd, recent memory sample trends,
largest runtime artifacts, and code paths that likely read large files into
memory. Treat findings as investigation leads, not automatic blockers.

## Runtime Artifact Retention

Plan compaction without changing files:

```bash
./venv/bin/python -m ai_trading.tools.runtime_artifact_retention \
  --output-json /var/lib/ai-trading-bot/runtime/runtime_artifact_retention_latest.json
```

Apply compaction after reviewing the plan:

```bash
./venv/bin/python -m ai_trading.tools.runtime_artifact_retention \
  --apply \
  --output-json /var/lib/ai-trading-bot/runtime/runtime_artifact_retention_latest.json
```

Compaction keeps the newest records, writes a timestamped compressed backup,
and never runs in apply mode from the daily research automation. The daily
research bundle includes the memory audit and retention plan so operators can
decide whether cleanup is needed.
