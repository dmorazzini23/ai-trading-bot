SLOs and Alerts for Trading Bot Cycle

This document defines recommended SLOs and example Prometheus alert rules for the trading cycle. Metrics are emitted by the application and exposed under `/metrics` when Prometheus client is available.

Key Metrics
- `cycle_stage_seconds{stage=fetch|compute|execute}`: Histogram of per-stage durations in seconds.
- `cycle_budget_over_total{stage=fetch|compute|execute}`: Counter of budget overruns per stage.
- `alpaca_call_latency_seconds`: Histogram of Alpaca call latency.
- `alpaca_calls_total`, `alpaca_errors_total`: Counters for Alpaca calls and errors.
- `orders_submitted_total`, `orders_rejected_total`, `orders_duplicate_total`: Execution flow counters.

Suggested SLO Targets
- Compute stage p95 < 5s; p99 < 10s
- Execute stage p95 < 1s; p99 < 2s
- Budget overruns for any stage < 1 per 10 minutes under normal operations
- Alpaca call latency p95 < 500ms; error rate < 1%

PromQL Examples
```
# p95 per stage over 5m
histogram_quantile(
  0.95,
  sum(rate(cycle_stage_seconds_bucket{stage="compute"}[5m])) by (le)
)

# Budget overruns over 10m window
increase(cycle_budget_over_total{stage="compute"}[10m])

# Alpaca call error rate over 5m
sum(increase(alpaca_errors_total[5m])) / sum(increase(alpaca_calls_total[5m]))
```

Alert Rules (Prometheus)
```
groups:
- name: ai-trading-slos
  rules:
  - alert: CycleComputeLatencyHighP95
    expr: |
      histogram_quantile(0.95, sum(rate(cycle_stage_seconds_bucket{stage="compute"}[5m])) by (le)) > 5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Compute stage p95 latency high"
      description: "p95 > 5s for 10m. Investigate compute bottlenecks."

  - alert: CycleExecuteLatencyHighP95
    expr: |
      histogram_quantile(0.95, sum(rate(cycle_stage_seconds_bucket{stage="execute"}[5m])) by (le)) > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Execute stage p95 latency high"
      description: "p95 > 1s for 10m. Broker/API latency likely."

  - alert: CycleBudgetOverruns
    expr: |
      increase(cycle_budget_over_total{stage=~"compute|execute|fetch"}[10m]) > 5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Frequent cycle budget overruns"
      description: ">5 budget overruns in 10m across stages."

  - alert: AlpacaErrorRateHigh
    expr: |
      sum(increase(alpaca_errors_total[5m])) / sum(increase(alpaca_calls_total[5m])) > 0.05
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Alpaca error rate > 5%"
      description: "Alpaca call error rate sustained above 5% for 10m."
```

Rollout
- Load rules into Prometheus and alertmanager.
- Confirm metrics scrape from `/metrics`.
- Tune thresholds as you profile in your environment.

