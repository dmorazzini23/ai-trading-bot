# GitHub workflow repair — September 17, 2026

## Worklog and root causes

CI push run 35178704342, scheduled run 35232739148 and current push run
35234635440 fail on the same asynchronous replay regression: a buy limit below
all subsequent prices was expected to fill, with its fill timestamp before the
next observation. That contradicts the enforced simulator timing/limit contract.
The regression now supplies later equal prices and verifies fills occur only
at actual observation timestamps and respect the limit. Production fill rules
are unchanged. The earlier offline replay fixture repairs are already in HEAD.

Dependency Audit run 34865562264 reports vulnerable versions of aiohttp, click,
cryptography, idna, msgpack, pyarrow, pygments, soupsieve, torch, transformers and
urllib3. Update their direct pins/floors and affected transitive constraints.
Keep Alpaca 0.42.1 and Python 3.12. Regenerate the lock with --allow-unsafe and
--strip-extras; plain pip constraints cannot contain the CUDA-toolkit extras
introduced by the new Torch metadata. The selected Transformers 5.10.4 avoids
the yanked 5.10.0 release. PyPI metadata and the resolver confirm availability.

Workflow Lint, CodeQL, SBOM and the latest Scorecard had successful runs in the
inspected history. The latest research-backtest gate passed. These findings do
not claim every historical run or the unpublished repair branch is green.

## Patchset

All source edits use apply_patch: asynchronous replay test, runtime and optional
ML requirements, pyproject metadata, regenerated constraints, dependency-audit
workflow and CI workflow. Audit now triggers on constraints.txt changes and
removes the old advisory exclusions. CI retains the 80% coverage gate, collects
all failures instead of stopping at the first one and uploads JUnit artifacts.
Packaging regression validates constraint syntax and version compatibility with
both runtime dependency declarations.
An optional run_nightly_full_deps workflow-dispatch input allows the scheduled
full-dependency job to be validated on the repair branch before merge. That
manual mode runs only the nightly job; normal push/PR and scheduled selection
is unchanged. It does not disable any test or coverage threshold.

## Validation and runtime check

- 64 targeted replay/determinism tests passed using the existing environment.
- Packaging/constraint contract tests pass; workflow actionlint passes.
- Required --market-hours changed-file validation passed its selected regression,
  lint, type and compile checks. No broad local suite ran during market hours.
- Isolated Python 3.12 dry run resolves requirements-dev.txt and
  requirements-test.txt against the new constraints.
- All 125 packages in the resolved runtime lock audited clean, with no ignored
  vulnerabilities. Evidence: /tmp/workflow-patched-audit.json.
- Production installed packages, service and model gates were not changed.
  The first complete PR run passed 6,773 tests and exposed nine failures;
  the full-dependency run passed 6,777 and exposed five of those same failures.
  Coverage was 78.76% and 78.80%, respectively, below the unchanged 80% gate.
- Corrected stale one-minute feature fixtures to valid five-minute sessions,
  replaced the constructor regex with AST inspection, isolated health response
  adapters from collection-time Flask stubs, and refreshed SDK/config class
  fixtures after module reloads. Deferred SDK exception imports in retry and
  reconciliation preserve lazy imports without changing exception handling.
  Added import regressions for each affected module; 89 focused tests pass.
- Dependency Audit on the PR installed 140 packages and found no vulnerabilities.
  Replay, all determinism seeds, research, actionlint, CodeQL and SBOM passed.
  CI now retains coverage XML alongside JUnit for exact gap analysis. The full
  suite must be rerun after these repairs; coverage remains an unresolved gate.
- Follow-up changed-file validation passed lint, types (11 files), compilation
  and its related regression suite. A separate two-worker isolation run passed
  seven tests. Live health returned required_model_stale (existing degraded
  readiness); the non-sending incident snapshot passed. No branch deployment.

## Publishing and next action

### September 18 follow-up

Commit cf7a973e8 retains the replacement returned by immutable TradingConfig.update
in four test modules (11 tests pass). Artifact uploads now explicitly include
hidden files so `.ci/*.xml` is actually retained. The prior full-dependency run
passed 6,785 tests but failed the unchanged 80% gate at 78.82% coverage.

Additional boundary regressions found two runtime defects: startup broker-order
reconciliation failure could fall through to the trading worker, and an invalid
sliced-order type silently became a market order. Both now fail closed. Added
startup, child-order conservation/retry/type and trade-history boundary tests.
Related main tests: 38 passed; execution plus offline replay: 63 passed;
trade-history tests: 15 passed. These patches remain undeployed.
Changed-file validator passed lint, types (five files), compilation and 43
regressions. The final startup read-timeout/retry regression also passes (nine
startup tests). No runtime model or qualification gate was relaxed.

Operationally, market-preflight now uses OpenClaw's direct command payload for
the canonical Python module, preserving its schedule, enabled state and delivery.
Direct non-sending execution completed with blocked readiness; natural scheduled
delivery remains to be observed. Service remains active with zero restarts.

A fresh two-day broker accounting fetch at 2026-09-18T01:35:38Z returned 16 fills
but no execution-linked fee amounts. Existing comparison has one paired order
and zero net-cost pairs, versus minimum 30 orders over five sessions. Historical
benchmark mismatches and missing simulated orders remain explicit exclusions;
do not invent fees, normalize historical benchmarks or relax qualification.

Residual investigation: a legacy audit-to-meta converter test exposed two
opposite-reward rows for one buy/sell round trip. This is not repaired here;
training remains paused under the research reset. Full CI coverage is still an
open gate; added focused tests do not establish 80% overall coverage.

Prepared isolated worktree /tmp/ai-trading-workflow-fix on branch
codex/fix-actions-20260917. Automatic approval review rejected the bundled
commit/push/draft-PR command, stating that publishing repository contents to the
remote and opening a PR require explicit authorization. No commit, push or PR
creation took place at that point. The user subsequently explicitly authorized
commit, push and draft PR creation. Run remote CI and resolve any further
failures before claiming all workflows fixed. Merge/deployment remain separate.

PR description prepared in /tmp/workflow-pr-body.md. Logs and artifacts:
/tmp/workflow-{replay-targeted,packaging-final,local-validation,lock-final,
install-plan,patched-audit}.log; original failure logs /tmp/ci-*.log.

## Risk and rollback

The security fixes upgrade Torch to 2.13.0 (CUDA 13 dependency family),
Transformers to 5.10.4, Arrow to 23.0.1 and cryptography to 50.x. Resolver/audit
success alone does not prove model or runtime compatibility. Keep production
unchanged pending CI and deployment review. Roll back requirements and lockfile
together; test and reporting changes can be reverted independently. No data
migration or training campaign was performed.

References: [failed CI](https://github.com/dmorazzini23/ai-trading-bot/actions/runs/35178704342),
[failed audit](https://github.com/dmorazzini23/ai-trading-bot/actions/runs/34865562264),
[Torch release metadata](https://pypi.org/project/torch/2.13.0/).
