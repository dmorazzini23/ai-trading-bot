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
- optionally train lightweight 1-bar and 15-bar risk-adjusted candidates when a
  research data directory is configured

Weekly:

- pin the current live cost model
- train/evaluate 1/3/5/15-bar candidates across net, spread-adjusted,
  risk-adjusted, and MAE/MFE objectives
- optionally join shadow quote telemetry to replay candidates with the
  microstructure bridge

Monthly:

- refresh replay governance
- run broader logistic and histogram-gradient architecture checks when research
  data is configured
- run a paper-mode live-cutover drill artifact for capital-profile review

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
- `<cadence>_research_latest.json`
- `<cadence>_operator_summary.json`
- `latest/daily_readiness_latest.json`
- `latest/trading_day_latest.json`
- `latest/trading_day_latest.md`

Slack/OpenClaw should read these artifacts and summarize them. Heavy training
should stay in systemd/repo scripts, not in Slack app handlers.

## Operator Commands

Plan a run without executing steps:

```bash
./venv/bin/python -m ai_trading.tools.research_automation daily --plan-only
./venv/bin/python -m ai_trading.tools.research_automation weekly --plan-only
./venv/bin/python -m ai_trading.tools.research_automation monthly --plan-only
```

Run the canonical script:

```bash
scripts/run_research_automation.sh daily
scripts/run_research_automation.sh weekly
scripts/run_research_automation.sh monthly
```

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
```

Check timers and recent runs:

```bash
systemctl list-timers 'ai-trading-research-*' --no-pager
journalctl -u ai-trading-research-daily.service -n 100 --no-pager
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_operator_summary.json
```

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
- Live-money cutover remains a separate manual decision and should start with
  canary limits, no shorts, strict quote authority, strict spread/quote-age
  caps, and a tiny capital cap.
