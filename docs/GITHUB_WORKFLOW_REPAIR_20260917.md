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
  Full execution of the suite with upgraded dependencies still requires GitHub
  CI before merge or deployment. The initial coverage failures followed early
  test termination; full-suite coverage has not yet been established.

## Publishing and next action

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
