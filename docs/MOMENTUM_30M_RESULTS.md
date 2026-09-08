# Fixed 30-minute momentum study: retired

Executed September 8, 2026 using the pre-recorded
`SEPTEMBER_8_RESEARCH_CHECKPOINT.md` protocol. No model was fitted, no holdout
outcomes were evaluated, and no trading settings changed.

The frozen governed dataset passed coverage validation for AAPL, AMZN and MSFT:
126 sessions each, with missing-bar ratios below 0.21%. Evaluation used only the
five saved development windows: 2,353 common opportunities across 75 sessions,
including 1,235 momentum selections. Signals use completed bars, entries use
the next bar open, and exits occur 30 minutes later. Positions do not overlap
within a symbol. These are equal-notional markouts, not portfolio returns.

| Control | Net bps per common opportunity at 6 bps cost |
| --- | ---: |
| Cash | 0.0000 |
| Always-long | -4.9302 |
| Momentum | -2.1250 |

Momentum's 95% session-block bootstrap interval is [-3.1931, -1.0176] bps,
using 10,000 replicates and the pre-recorded seed. Only one of five folds was
profitable; all three symbols had negative net edge. Every fold exceeded the
minimum 30 selections. Momentum beat always-long by 2.8052 bps per opportunity
but failed the positive-edge, uncertainty, fold-stability and symbol criteria.
The protocol is retired; no threshold or horizon tuning is justified on these
same development windows under this protocol.

Gross break-even cost was 1.9512 bps per momentum selection. Net edge per common
opportunity at round-trip costs of 0, 3, 6 and 10 bps was respectively 1.0241,
-0.5504, -2.1250 and -4.2245 bps. Passing the zero-cost sensitivity is not passing
the fixed acceptance criterion. Bar prices do not establish executable costs.

## Reproducibility and exclusions

Canonical implementation: `ai_trading.tools.momentum_horizon_study`.
Artifacts: `artifacts/momentum_30m_study_frozen/` contains the protocol snapshot,
source hashes, quality report, every accepted opportunity and final report.
The original checkpoint document remains the pre-evaluation protocol.

The initial attempt at `artifacts/momentum_30m_study/` stopped before evaluation:
the scheduled acquisition had changed the latest dataset manifest. Its metadata
is preserved. The successful run used the archived September 4 acquisition
report matching the frozen benchmark hash; no hypothesis or code correction
was needed to continue. Reproduction command (use a new output directory to
preserve evidence, and do not treat reproduction as a new experiment):

```bash
./venv/bin/python -m ai_trading.tools.momentum_horizon_study \
  --benchmark artifacts/baseline_benchmark/baseline_benchmark.json \
  --acquisition-manifest /var/lib/ai-trading-bot/runtime/research_reports/daily/20260904T203559Z_daily/historical_training_backfill.json \
  --protocol docs/SEPTEMBER_8_RESEARCH_CHECKPOINT.md \
  --output-dir artifacts/momentum_30m_study_reproduction
```

Per symbol, 67 grid points lay outside fold signal windows, 74 exits crossed
the regular-session boundary, and 5 crossed fold boundaries. Missing contiguous
bars excluded 24 AAPL, 15 AMZN and 14 MSFT opportunities. Exclusions are mutually
exclusive in that order; loader cleanup diagnostics are separately preserved.

## Validation and limitations

`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed lint,
mypy, compilation and all three focused regression tests. Tests verify next-bar
entry, exact exit timing, non-overlap, missing-bar and fold-boundary rejection,
causal feature invariance, cost arithmetic, whole-session bootstrap and stopping
decisions. The actual historical study completed successfully. Runtime smoke
checks are inapplicable to this standalone offline study; no service was changed.

The dataset and development windows informed the hypothesis, so this is an
exploratory result, not untouched validation. Execution fee evidence remains
incomplete independently of this research outcome. Rollback removes the new
module and tests while preserving the protocol and result artifacts.
