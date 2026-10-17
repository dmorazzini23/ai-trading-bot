# Scheduled market preflight

The market preflight runs at 09:20 America/New_York on weekdays via the
`ai-trading-slack-report-preflight.timer` user unit, which formats this report and
posts it to Slack (see docs/OPERATIONS.md "Scheduled Slack Reports"). The
read-only health command, from /home/aiuser/ai-trading-bot, is:

```sh
/home/aiuser/ai-trading-bot/venv/bin/python -m ai_trading.tools.market_preflight
```

The command reads the canonical local :9001/healthz endpoint, including the
JSON body of HTTP 503 responses. It does not reimplement health or qualification
gates. A valid JSON report exits zero even when readiness_verdict is blocked;
that means collection succeeded, not that trading is authorized. Invalid JSON,
missing boolean ok, unexpected HTTP status or transport failure emits a failed
report with readiness unknown and exits one. Requests have a 15-second timeout
and responses are bounded to one MiB.

The job separately reads systemctl service status and the last 120 journal lines.
It reports service liveness independently from stale-model/replay readiness
blockers and consumes the actual replay_live_parity_gate/readiness_gates fields.
It must not improvise curl/jq pipelines, guess missing fields or relax gates.

The September 17 alert was an invalid jq object-value fallback expression;
the downstream parse failure caused curl error 23. The agent subsequently
corrected its expression and produced a report, but the failed tool call caused
an alarming job notification. The fixed Python command removes this quoting
and jq syntax failure mode. Regression coverage includes blocked HTTP 200/503,
true readiness, malformed bodies, wrong ok types, transport failure and HTTP 500.

(History: until 2026-10-17 this job ran in the OpenClaw gateway, which was retired.)

Validation on September 17: 10 focused regression tests and 233 selected tests
in the required market-hours validator passed, with lint/type/compile checks.
The live command successfully reported blocked HTTP503 with existing stale-model
and replay-parity flags. Non-sending incident snapshot passed. The scheduled job
message was updated using the matching CLI and read back; the gateway trimmed
only its trailing newline. Schedule, delivery and enabled state were preserved.
No manual cron run or extra notification was sent; natural delivery remains to
be observed at the next scheduled run.
