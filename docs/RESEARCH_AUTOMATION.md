# Trading Research Automation

The research automation system turns the bot's existing evidence tools into a
repeatable after-hours operating loop. It creates artifacts for Slack/OpenClaw
and operators to read; it does not promote production models or enable live
capital automatically.

## Cadence

Daily after market close:

- refresh runtime performance reports
- build the live cost model
- build the minute-frame ML shadow report
- refresh replay governance
- refresh the symbol universe scorecard
- refresh runtime decay controls
- capture runtime go/no-go status
- generate the trading-day attribution report
- generate the daily research/readiness report
- generate the live-capital readiness gate artifact
- optionally run the training accelerator when a research data directory is
  configured. The accelerator caches replay-aligned feature frames and trains
  lightweight 1-bar and 15-bar risk-adjusted candidates with no promotion
  authority.
- summarize Hugging Face research artifacts. By default this is metadata-only
  and research-only; it cannot promote models or alter runtime authority.

Weekly:

- pin the current live cost model
- train/evaluate 1/3/5/15-bar candidates across net, spread-adjusted,
  risk-adjusted, and MAE/MFE objectives
- use the same cached training accelerator for the broader weekly candidate
  refresh
- optionally join shadow quote telemetry to replay candidates with the
  microstructure bridge
- optionally discover Hugging Face models/datasets for offline research ideas
  when `AI_TRADING_HF_RESEARCH_ENABLED=1`

Monthly:

- refresh replay governance
- run broader logistic and histogram-gradient architecture checks when research
  data is configured
- run a paper-mode live-cutover drill artifact for capital-profile review
- refresh Hugging Face candidate intake for architecture review

Manual only:

- model promotion report
- live-money cutover drill
- incident replay review
- major strategy-change research

## Artifacts

Each run writes a dated bundle under:

```bash
/var/lib/ai-trading-bot/runtime/research_reports/<cadence>/<run-id>/
```

Stable latest pointers are written under:

```bash
/var/lib/ai-trading-bot/runtime/research_reports/latest/
```

Important files:

- `research_automation_report.json`
- `operator_summary.json`
- `daily_research_report.json`
- `trading_day_report.json`
- `live_capital_readiness.json`
- `training_accelerator/training_accelerator_report.json`
- `evidence_manifest` inside `research_automation_report.json`
- `<cadence>_research_latest.json`
- `<cadence>_operator_summary.json`
- `latest/daily_readiness_latest.json`
- `latest/trading_day_latest.json`
- `latest/trading_day_latest.md`
- `latest/hf_discovery_latest.json`
- `latest/hf_candidate_intake_latest.json`
- `latest/hf_cache_materialization_latest.json`

Slack/OpenClaw should read these artifacts and summarize them. Heavy training
should stay in systemd/repo scripts, not in Slack app handlers.

When `AI_TRADING_RESEARCH_NOTIFY_SLACK=1`, the runner posts a completion summary
after each daily, weekly, monthly, or manual research job. The message is sent
through `AI_TRADING_RESEARCH_SLACK_WEBHOOK_URL`, falling back to
`AI_TRADING_SLACK_WEBHOOK_URL` or `SLACK_WEBHOOK_URL`. The packaged systemd
units set the target label to `#all-beatwallstreet`; make sure the configured
webhook URL is the incoming webhook for that channel.

## Operator Commands

Plan a run without executing steps:

```bash
./venv/bin/python -m ai_trading.tools.research_automation daily --plan-only
./venv/bin/python -m ai_trading.tools.research_automation weekly --plan-only
./venv/bin/python -m ai_trading.tools.research_automation monthly --plan-only
./venv/bin/python -m ai_trading.tools.research_automation weekend-saturday --plan-only
./venv/bin/python -m ai_trading.tools.research_automation weekend-sunday --plan-only
```

Run the canonical script:

```bash
scripts/run_research_automation.sh daily
scripts/run_research_automation.sh weekly
scripts/run_research_automation.sh monthly
scripts/run_research_automation.sh weekend-saturday
scripts/run_research_automation.sh weekend-sunday
```

## Weekend Research Schedule

Weekend automation uses extra CPU headroom without changing runtime authority.
Saturday runs broad research/training and Sunday runs validation plus Monday
operator synthesis. Both cadences write under:

```bash
/var/lib/ai-trading-bot/runtime/research_reports/weekend/<run-id>/
```

Stable latest pointers are:

```bash
/var/lib/ai-trading-bot/runtime/research_reports/latest/weekend_research_latest.json
/var/lib/ai-trading-bot/runtime/research_reports/latest/weekend_operator_summary.json
```

Weekend research is evidence only. It must not promote models, change launch
profiles, change executable symbols, increase size, restart services, or enable
live trading. Any authority-increasing recommendation still requires manual
operator approval plus the existing replay, shadow, live-cost, promotion, and
live-capital readiness gates.

Safe caps are controlled through:

```bash
AI_TRADING_WEEKEND_RESEARCH_ENABLED=1
AI_TRADING_WEEKEND_RESEARCH_MAX_RUNTIME_MINUTES=180
AI_TRADING_WEEKEND_RESEARCH_MAX_SYMBOLS=25
AI_TRADING_WEEKEND_RESEARCH_MAX_CANDIDATES=100
AI_TRADING_WEEKEND_RESEARCH_MAX_REPLAY_CANDIDATES=15
AI_TRADING_WEEKEND_RESEARCH_MAX_PARALLEL_WORKERS=2
AI_TRADING_WEEKEND_RESEARCH_CACHE_ENABLED=1
AI_TRADING_WEEKEND_VALIDATION_MAX_RUNTIME_MINUTES=120
AI_TRADING_WEEKEND_VALIDATION_MAX_REPLAY_CANDIDATES=20
```

Saturday initially targets a 3-hour cap. Sunday targets a 2-hour cap and should
produce the Monday preparation answer: whether the system may trade next
session, what mode is recommended, which blockers remain, and what manual
operator actions are required.

Manual workflows:

```bash
./venv/bin/python -m ai_trading.tools.research_automation manual \
  --workflow promotion \
  --model-path /path/to/candidate.joblib \
  --manifest-path /path/to/candidate.joblib.manifest.json

./venv/bin/python -m ai_trading.tools.research_automation manual \
  --workflow live-cutover

./venv/bin/python -m ai_trading.tools.research_automation manual \
  --workflow incident-replay

./venv/bin/python -m ai_trading.tools.research_automation manual \
  --workflow strategy-change
```

These workflows produce artifacts only. The operator must still review gates and
explicitly perform any runtime model cutover, live-money enablement, incident
response action, or strategy change.

## Live-Capital Readiness

The daily bundle now answers the operator question:

```text
Can this system trade tomorrow, with what limits, and why?
```

The answer is split across two artifacts:

- `daily_research_report.json` summarizes runtime health, provider authority,
  live costs, shadow evidence, replay governance, promotion status, symbol
  actions, and the recommended next-session mode.
- `live_capital_readiness.json` is the hard live-money gate. It requires a green
  full-validation artifact, healthy runtime, broker/database/OMS/replay health,
  acceptable promotion and live-cost evidence, explicit live-account
  confirmation, a launch profile, and a canary plan before allowing live capital.

The live-capital artifact can report `blocked`, `paper_only`,
`live_canary_allowed`, or `live_allowed`. Only the operator may act on an
allowed status; automation never flips execution mode or account authority.

## Systemd

Install the automation units:

```bash
sudo cp packaging/systemd/ai-trading-research-*.service /etc/systemd/system/
sudo cp packaging/systemd/ai-trading-research-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-research-daily.timer
sudo systemctl enable --now ai-trading-research-weekly.timer
sudo systemctl enable --now ai-trading-research-monthly.timer
sudo systemctl enable --now ai-trading-research-weekend-saturday.timer
sudo systemctl enable --now ai-trading-research-weekend-sunday.timer
```

Check timers and recent runs:

```bash
systemctl list-timers 'ai-trading-research-*' --no-pager
journalctl -u ai-trading-research-daily.service -n 100 --no-pager
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_operator_summary.json
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/weekend_operator_summary.json
```

Smoke-test the completion notification without sending:

```bash
./venv/bin/python -m ai_trading.tools.research_completion_notify \
  --cadence daily \
  --channel '#all-beatwallstreet' \
  --dry-run
```

Run the training accelerator directly in plan mode:

```bash
./venv/bin/python -m ai_trading.tools.training_accelerator \
  --cadence daily \
  --data-dir /path/to/research/bars \
  --symbols AAPL,AMZN \
  --plan-only
```

Run Hugging Face research discovery in metadata-only mode:

```bash
./venv/bin/python -m ai_trading.tools.huggingface_research_discovery \
  --report-date "$(date -u +%F)" \
  --offline-results-json /path/to/hf_fixture.json
```

Enable weekly/monthly Hugging Face API discovery only when you intentionally
want external metadata scans:

```bash
AI_TRADING_HF_RESEARCH_ENABLED=1 \
./venv/bin/python -m ai_trading.tools.huggingface_research_discovery \
  --report-date "$(date -u +%F)" \
  --enabled \
  --use-hf-api
```

Convert selected candidates into manual offline research hypotheses:

```bash
./venv/bin/python -m ai_trading.tools.huggingface_candidate_intake \
  --discovery-json /var/lib/ai-trading-bot/runtime/research_reports/latest/hf_discovery_latest.json \
  --candidate-id org/model-name
```

Plan, but do not download, optional cache materialization:

```bash
./venv/bin/python -m ai_trading.tools.huggingface_cache_materializer \
  --intake-json /var/lib/ai-trading-bot/runtime/research_reports/latest/hf_candidate_intake_latest.json \
  --dry-run
```

Actual downloads require `AI_TRADING_HF_ALLOW_DOWNLOADS=1` or
`--allow-downloads`, and cached files stay under the configured Hugging Face
research cache. They must never overwrite production model artifacts.

Safe Hugging Face knobs:

```bash
AI_TRADING_HF_RESEARCH_ENABLED=0
AI_TRADING_HF_ALLOW_DOWNLOADS=0
AI_TRADING_HF_MAX_RESULTS=25
AI_TRADING_HF_CACHE_DIR=/var/lib/ai-trading-bot/runtime/research_reports/huggingface/cache
AI_TRADING_HF_TOKEN_SECRET_NAME=AI_TRADING_HF_TOKEN
```

The default posture is disabled or metadata-only. Discovery may find ideas;
candidate intake turns selected ideas into experiment-ledger entries; cache
materialization is optional offline research storage. Promotion remains a
separate local evidence process.

Non-plan accelerator runs use cached feature frames and a bounded
successive-halving style replay pass. Daily runs replay only the strongest small
candidate set by default; weekly/monthly runs search more broadly while still
keeping expensive replay bounded. Override with `--max-replay-candidates` when
running an explicit research experiment.

Run a manual workflow through systemd:

```bash
sudo systemctl start ai-trading-research-manual@incident-replay.service
```

For promotion workflows that need explicit model paths, prefer the direct CLI so
the candidate path is visible in shell history and the generated report.

## Safety Rules

- Automated cadences never mutate production model paths.
- Automated cadences never enable live money.
- Weekly/monthly research can train candidates, but those candidates stay
  research artifacts until a manual promotion report passes and an operator
  performs the cutover.
- Hugging Face artifacts are external research context only. They must retain
  `runtime_authority=false`, `promotion_authority=false`, and
  `live_money_authority=false`.
- Hugging Face tokens, if needed for gated/private metadata, should be provided
  through managed secrets. Do not place tokens in Slack messages, logs, health
  payloads, or committed files.
- Live-money cutover remains a separate manual decision and should start with
  canary limits, no shorts, strict quote authority, strict spread/quote-age
  caps, and a tiny capital cap.
