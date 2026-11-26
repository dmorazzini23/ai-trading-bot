# Current handoff

## November 26: backstop first-cycle fix and paper-only parity flag scoping deployed (`892b254f4`) (start here)

- **Running bot:** `892b254f4` since 04:46:07 UTC November 26 (market closed,
  flat at deploy), no restarts, health `healthy`. Config hash unchanged
  (`043df350…`); only the commit changed in the release spec. Full suite on
  this commit: 6,912 passed, 2 skipped.
- **Deployed:** the position backstop takes a new position's entry basis from
  the broker rows the current cycle already read (at most 600 s old), so it
  arms on the first cycle that sees the position; the fill-to-next-cycle gap
  (up to about 96 s) remains. In paper only, the attention flag
  `replay_live_parity_gate_failed` is withheld when the economics check
  (`replay_counterfactual`) is the sole, fully evidenced failure; the gate,
  readiness and the real-money entry block are unchanged (see
  `docs/OPERATIONS.md`).
- **Expected:** the parity flag shows after any restart until the first
  open-market cycle publishes its record, then clears. `/control-plane` and
  the standalone health process still show it.
- **Rollback:** stop, `git checkout --detach 11fe01837`, restore
  `runtime/release_spec.11fe01837.bak.json`, start.
- **Not yet traded:** none of the code deployed since the November 25 close
  has traded a session; first is Friday November 27 (half day, closes
  18:00 UTC). Check then: `POSITION_BACKSTOP_ARMED` on a new position's first
  cycle, the parity flag clearing, no `ALERT_DECISION_LOOP_STALLED`.
- **Owner still to do:** install the packaged unit (commands in the next
  section).
- **In progress (not merged):** `claude/earnings-trial-runner` (offline runner
  and draft contract; nothing registered; owner decision sheet being
  prepared).

## November 25 (late): leftover model code removal and liveness alert fix deployed (`11fe01837`)

- **Running bot:** `11fe01837` since 23:27:28 UTC November 25, no restarts,
  health `healthy`. Config hash
  `043df3509073df05038c8731f2eb48363e004687934c0c85a941ba921260a738`
  (owner approved). Full suite on this commit: 6,838 passed, 2 skipped.
- **Deployed:** `signal_ml`, `ai_trading/model_loader.py`, their helpers and
  the `strict_model_loading` ConfigSpec removed; `AI_TRADING_MODEL_MODULE`
  removed from the packaged unit and `start.sh`; RL liveness alert at error
  level; Slack "Signal age" line without ML.
- **Rollback:** stop, `git checkout --detach 92dddb269`, restore
  `runtime/release_spec.92dddb269.bak.json`, start.
- **`.env`:** six unread names removed (1,617 lines). Release check passes.
- **Owner still to do:** install the packaged unit once
  (`sudo install -m 0644 /home/aiuser/ai-trading-bot/packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service`,
  `sudo systemctl daemon-reload`, `sudo systemctl restart ai-trading.service`).
  After that, a rollback to code older than step 2 needs the old unit back.
- **Not yet traded:** first session on this code is Friday November 27
  (half day, closes 18:00 UTC).
- **In progress (owner approved, not merged):** `claude/parity-flag-scope`
  (paper-only display scoping of `replay_live_parity_gate_failed`),
  `claude/backstop-basis` (arm the backstop on a new position's first cycle),
  `claude/earnings-trial-runner` (offline runner; nothing registered).

## November 25 (evening): day-sleeve step 2, slippage-spec removal and loop-stall alert deployed (`92dddb269`)

- **Running bot:** `92dddb269` since 22:33:59 UTC November 25, no restarts.
  Config hash `c635a50662cdfbb4a7a0ad6b2af414a11eaca6697edc3267ae63975aaf232fc7`
  (was `3084db64…`; owner approved). Health `healthy`; only flag
  `replay_live_parity_gate_failed`. `decision_loop.monitored: true`,
  threshold 900 s. Paper burn-in counters reset on a config hash change.
- **Deployed:** legacy startup model load and ML liveness leg removed; release
  check no longer pins a model (`--models-root` ignored, logged as
  `RELEASE_IDENTITY_MODELS_ROOT_IGNORED`); `marketable_limit_slippage_bps` and
  `warn_if_model_missing` ConfigSpecs removed; alert-only decision-loop
  watchdog (`DECISION_LOOP_STALLED`, `docs/OPERATIONS.md`).
- **Validation:** full suite on `92dddb269`: 6,874 passed, 2 skipped. Release
  check passed both phases. Bot was flat at the stop.
- **Rollback:** stop, `git checkout --detach 7ade302ee`, restore
  `runtime/release_spec.7ade302ee.bak.json`, start.
- **Installed systemd unit is still the old one** (passes `--models-root`).
  New code runs under it. Owner installs the packaged unit once, after the
  model-leftovers deploy (that branch edits the unit again). The new unit does
  not start code older than step 2.
- **`.env`:** `MARKETABLE_LIMIT_SLIPPAGE_BPS` and
  `AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS` removed (1,623 lines).
  `AI_TRADING_ML_SHADOW_ENABLED` must stay on: it gates the pre-rank quote
  refresh (feeds the de-risk gate) and the per-cycle pre-rank snapshot row.
  A rename was considered and dropped.
- **Session November 25 on `7ade302ee`:** clean. `rule_paper_sampling` labels on
  all orders; `POSITION_BACKSTOP_ARMED` seen for the first time (AMZN, MSFT).
  A new position is unarmed (`no_basis`) for one cycle until the broker-sync
  snapshot includes it.
- **Merged to main, test only:** SLO monitor isolation fix
  (`tests/monitoring/test_slo_institutional.py`).
- **Built, not merged, not deployed:**
  - `claude/liveness-alerts` (`e1531bf32`): RL liveness alert at error level;
    Slack "Signal age" line without ML. No config change.
  - `claude/model-leftovers` (`fc28964c5`): removes `signal_ml`,
    `model_loader`, helpers, `strict_model_loading` ConfigSpec and
    `AI_TRADING_MODEL_MODULE` from the unit and `start.sh`. Reviewed; changes
    the config hash; full suite not yet run. After its deploy these `.env`
    names can go: `AI_TRADING_REQUIRE_ML_MODEL`,
    `AI_TRADING_ML_USE_REGIME_THRESHOLDS`, `AI_TRADING_MODEL_MANIFEST_PATH`,
    `AI_TRADING_ML_SHADOW_MODEL_MODULE`, `AI_TRADING_ML_SHADOW_MODEL_PATH`,
    `AI_TRADING_MODEL_MODULE`.
- **Open owner decision:** `replay_live_parity_gate_failed` is permanently on
  because its economics check (250 samples, positive net edge) cannot pass
  with only the rule sampler; all parity checks pass. Proposal: in paper, raise
  the flag only for parity/freshness/contract failures and report economics
  separately; gate semantics for real money unchanged.
- **Research:** earnings draft `d60cc5a60` on `claude/earnings-trial-runner`
  (unregistered). Owner delegated the "distinct" ruling: stands on condition
  that out-of-basket entries pass on their own. Desk comparison ranked
  earnings first (about 8–12% pass chance on full-history data), dividend
  ex-date run-up second. Nothing registered; waits on Sharadar data
  (reminder December 3).

## November 25: exit-order fixes, day-sleeve step 1 and readiness block deployed (`7ade302ee`)

- **Running service:** `7ade302ee` since Wed Nov 25 00:30 UTC (closed, flat,
  no open orders), owner-approved. Config hash unchanged (`3084db64...`);
  `release_identity` pre-check passed both phases on this commit.
- **Carries:** `claude/exit-order-gaps`, `claude/day-sleeve-retire` (step 1)
  and `claude/readiness-no-strategy`, described in the next section, which
  were merged after it was written.
- **Readiness block:** `paper_sampling.qualified_strategy_available()`
  returns False in code (not an env var or ConfigSpec).
  `live_capital_readiness` always adds the blocking reason
  `no_qualified_strategy`; the `paper_only` status no longer appears.
  `daily_research_pipeline._trade_allowed` does not read it (owner decision:
  left alone).
- **Pre-deploy:** full suite on this commit: 6,852 passed, 2 skipped, 8
  failed. The 8 (`tests/scripts/test_research_automation_script.py`,
  `tests/tools/test_runtime_backup_sync.py`) run `./venv/bin/python` and
  failed only because the worktree had no `venv` link; with the link, all
  17 tests in those files pass. Independent pre-deploy review: no blocking
  finding. A new worktree needs `ln -s /home/aiuser/ai-trading-bot/venv venv`.
- **Startup:** all four markers, no tracebacks, `NRestarts=0`. `/healthz`
  is now `healthy`; `operating_mode.mode` is `paper_rule_sampling`,
  `strategy_state` `no_qualified_strategy`. `required_model_stale` and the
  `day_sleeve_model` key are gone; `replay_live_parity_gate_failed` stays.
- **Rollback:** stop, `git checkout --detach 1553f2b60`, restore
  `runtime/release_spec.1553f2b60.bak.json`, start.
- **First session (Nov 25) to check:** orders carry the
  `rule_paper_sampling` label; `POSITION_BACKSTOP_ARMED` per position;
  any `PENDING_CANCEL_HOLD_LONG`; closing-order fills.
- **`.env` names now unread (not yet removed, needs the owner's say):**
  `AI_TRADING_DAY_SLEEVE_ML_ENABLED`, `AI_TRADING_DAY_SLEEVE_ML_BLEND_WEIGHT`,
  `AI_TRADING_HEALTH_REQUIRE_DAY_SLEEVE_MODEL`,
  `AI_TRADING_PAPER_ALLOW_SHADOW_MODEL`,
  `AI_TRADING_PAPER_SAMPLING_STALE_MODEL_DIAGNOSTICS_ENABLED`, and the
  pending-new replace/ladder/hard-timeout names listed in
  `docs/OPERATIONS.md`.
- **Next:** day-sleeve step 2 (legacy `trained_model.pkl` load, release-check
  model pin, unit `--models-root`) needs a unit install by the owner.

## November 24: main deployed (`1553f2b60`)

- **Running service:** `1553f2b60` since Tue Nov 24 21:07 UTC (closed, flat,
  no open orders), owner-approved. Adds the unread-settings cleanup
  (`745096a46`), the docs accuracy pass and the `POSITION_BACKSTOP_ARMED`
  log line. Config hash is now `3084db64...` (owner-approved Nov 23;
  `release_identity` pre-check passed both phases on this commit).
- **Deploy order is stop-first:** back up the spec, stop the service, check
  out the commit, edit SHA and hash in the spec, start.
- **Pre-deploy:** full suite on this tree: 6,833 passed, 2 skipped. Session
  on `03344e00b` was clean: 0 restarts, no tracebacks, one ERROR line
  (`MODEL_REGISTRY_STALE`, expected). Six orders: MSFT buy unfilled; AAPL
  buy and sell filled; AMZN two buys part-filled (1 share each), one sell
  of 2 filled. Both closing orders filled in 11-14 s. No
  `POSITION_BACKSTOP` line all session.
- **Startup:** `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`,
  `BROKER_SYNC`, `DrawdownCircuitBreaker initialized`; no errors or
  tracebacks in the first four minutes; `NRestarts=0`; `/healthz` degraded
  only for `required_model_stale` (unchanged).
- **Rollback:** stop, `git checkout --detach 03344e00b`, restore
  `runtime/release_spec.03344e00b.bak.json` (old SHA and hash
  `49f5e469...`), start. `API_HOST` is gone from `.env`; `03344e00b` still
  has that spec, so re-run the `release_identity` pre-check before relying
  on the old hash.
- **Wednesday Nov 25:** first session that can show
  `POSITION_BACKSTOP_ARMED`; check that the stop arms for each position.
- **Operator `.env`:** 77 more unread names deleted on Nov 24 and
  `API_HOST` after this restart (1,635 lines). The two `.env.bak-*`
  backups were deleted at the owner's request, so the pre-Nov 23 hashes
  can no longer be reproduced from disk.
- **GitHub:** 103 stale remote branches deleted on Nov 23-24 (owner's
  request, no backups kept). Remaining: `main`, `gh-pages`,
  `claude/research-candidates`.
- **Built, NOT merged, NOT deployed:**
  - `claude/exit-order-gaps` (from `1553f2b60`): holds a symbol until the
    broker stops listing a cancelled order; part-filled closing orders
    reach the stale sweep; the real-money cancel suppression and the
    marketable-replace helper (incl. the 8 s order-TTL replacement) are
    removed; every pending-new policy cancels at the normal timeout.
  - `claude/day-sleeve-retire` (from `1553f2b60`): step 1 of retiring the
    day-sleeve model. Model serving removed; the rule scorer trades only
    under paper sampling, labelled `rule_paper_sampling`; otherwise
    positions are held (`NO_QUALIFIED_STRATEGY`) and only the backstop
    exits. Health reports the strategy state and drops
    `required_model_stale`. Step 2 (legacy `trained_model.pkl` load, the
    release-check model pin, the unit's `--models-root`) is a later deploy
    that needs a unit install.
  - Both edit `docs/OPERATIONS.md`; expect a merge conflict. Neither
    changes a ConfigSpec. Each needs the owner's yes to merge and deploy.
- **Shelved:** `claude/portfolio-manager-mode` is merged up with main but
  shelved by the owner (Nov 24). Do not resume it unless told. It has an
  unfixed snapshot-freshness skip under `live_portfolio`.
- **Real-money preconditions still open:** no qualified strategy; the
  readiness tool has no explicit "no qualified strategy" reason (owner
  decision pending); broker behaviour for `position_intent` and long
  `pending_cancel` unverified; the durable intent store's handling of a
  re-sent close with a reused client id unverified.
- **Research:** a pre-earnings hypothesis is drafted, not registered. It
  waits on survivorship-free daily data and the owner's word.

## November 23: five-change deploy done (`03344e00b`)

- **Running service:** `03344e00b` since Mon Nov 23 21:47 UTC (closed,
  flat), owner-approved. Carries the full-close fix, exit pricing, breaker
  persistence, Phase 4 cleanup and the breaker health section. Config hash
  is now `49f5e469...` (owner-approved; derived by the `release_identity`
  pre-check, both phases, before the spec was edited). Paper burn-in
  counter reset.
- **Pre-deploy:** full suite on `03344e00b`, finder stripped: 6,836 passed,
  2 skipped, 0 failed. Session on `d22babac4` was clean: 0 restarts, no
  tracebacks, three acknowledged orders (AMZN buy cancelled unfilled; MSFT
  buy and sell).
- **Startup:** `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`,
  `BROKER_SYNC`, `DrawdownCircuitBreaker initialized`; no errors or
  warnings in the first four minutes; `NRestarts=0`. `/healthz` shows
  `drawdown_breaker` with `state: unknown`, `state_load: pending` (the saved
  file loads on the first equity reading of a market cycle; expect one
  `DRAWDOWN_BREAKER_STATE_LOAD_FAILED` `missing` warning then).
- **Rollback:** `git checkout --detach d22babac4`, restore
  `runtime/release_spec.d22babac4.bak.json` (old SHA and old hash
  `b46e8e82...`), restart.
- **Backstop not yet exercised:** on Nov 23 it saw the one position (MSFT)
  and logged `POSITION_BACKSTOP_UNARMED` (`no_basis`) once, ~90 s after the
  buy, before the broker snapshot carried an entry price. There is no
  marker when a basis later arrives, so the journal cannot show the stop
  was armed. The position closed by a normal exit 27 minutes later.
  Consider logging the first arming per position.
- **Tuesday Nov 24:** first session for exit pricing (compare closing-order
  fills with November's) and for the breaker state in `/healthz`.
- **Operator `.env`:** the 30 retired keys that were still set were deleted
  on Nov 23 at the owner's request (backup
  `.env.bak-20261123-pre-retired-key-removal`); `release_identity` still
  passes on `03344e00b` with the edited file. Takes effect at the next
  restart.
- **Merged, NOT deployed (`745096a46`, `claude/env-cleanup`):** 13 unread
  ConfigSpecs (incl. `api_host`), 19 Settings fields and 12 unused config
  exports removed. `main` is therefore ahead of the running service and its
  config hash is `3084db64...` (owner-approved Nov 23; derived by the
  pre-check on the merged tree). Deploy it with the next change that needs
  a restart: set `tested_commit_sha` and that hash together. Full suite on
  the merged tree: 6,832 passed, 2 skipped; decision baseline unchanged on
  the branch; independent review, no must-fix. After that deploy `API_HOST`
  can also be deleted from `.env`. Out-of-range values for the removed
  specs no longer stop startup.

## November 23: breaker health section merged; five-change deploy approved (`90df63509`)

- **Running service (at the time):** still `d22babac4`. Nothing was restarted.
- **Approved deploy (owner, Nov 23):** the restart after the close on Mon
  Nov 23 now carries five changes: the four in the next section plus the
  breaker health section. Same condition (clean session first), same
  approved config hash (`49f5e469...`; this change adds no settings).
- **Change (`claude/breaker-health`, 2 commits):** `/healthz` and `/health`
  gain an informational `drawdown_breaker` section (state, peak, drawdown,
  thresholds, `last_reading_at`, `state_load`, `last_write_failed`). It
  never changes `ok`, `status` or the HTTP code and omits the account id.
  `state` is `unknown` until the first equity reading of a market cycle
  loads the saved file; the section is `not_initialized` until the bot
  context is built, and always in the standalone health app. See
  `API_DOCUMENTATION.md`.
- **Validation:** full suite 6836 passed, 2 skipped; decision baseline
  unchanged (`02fa7cd4...`); independent review (no must-fix; follow-ups
  done); trip-and-restart probe on the final code. The follow-up commit had
  no second independent review.
- **Correction:** the exit-pricing section below says an unfilled closing
  order is never cancelled or repriced. That is wrong. A trace on Nov 22
  (no code written; branch `claude/exit-reprice` is empty) found: the
  pending-new timeout policy replaces a stale order 6 bps wider at 20 s and
  cancels by 90 s on paper (`live_trading.py`, `replace_widen` in `.env`);
  the stale sweep cancels any open order after 75 s in both modes
  (`pending_order_runtime.py`), and the next cycle resubmits at the current
  quote; a symbol with a resting order is dropped from the netting cycle.
  Neither mechanism fired between Nov 2 and Nov 22 (journal).
- **Real-money checklist (owner deferred these on Nov 23; resolve or accept
  before any real-money switch):** (1) the 75 s sweep does not confirm the
  cancel before the symbol is released, so after a failed sync a
  resubmitted sell is clipped only by a stale open-order index, and a short
  cover is not clipped against open buys at all; (2) the timeout policy
  refuses to cancel verified closing orders in real-money mode
  (`LIVE_CLOSING_ORDER_CANCEL_SUPPRESSED`) but the sweep cancels them;
  (3) a partly filled exit rests up to 15 minutes before the sweep takes
  it; (4) the paper replace helper prices off the old limit, skips pretrade
  and never reads the position. See also `docs/SUPPORTED_TRADING_PATH.md`
  on held replacements.

## November 22: breaker persistence and Phase 4 merged; four-change deploy approved (`7dffb1379`)

- **Running service:** still `d22babac4` (position backstop). Nothing was
  restarted.
- **Approved deploy (owner, Nov 22):** one restart after the close on Mon
  Nov 23, provided the backstop's first market session looks clean, carrying
  all four changes on main: full-close fix (`db356eb49`), exit pricing
  (`da2570eb4`), drawdown-breaker persistence and Phase 4 cleanup.
- **Config hash:** Phase 4 removes 12 ConfigSpecs, so `resolved_config_hash`
  goes from `b46e8e828ec2...` to
  `49f5e4697a4f298a1b12c21eac0be155211ba7d60e85ddb474310f402c3939a3`. The
  owner approved this change for this deploy. Re-derive the hash with the
  `release_identity` pre-check before editing `release_spec.json`; do not
  type it from this note. The paper burn-in counter resets on the first
  paper cycle after the restart (it gates real-money mode only).
- **Breaker persistence (`claude/breaker-peak`):** peak equity and tripped
  state are saved to `runtime/drawdown_breaker_state.json` and restored on
  start. First start logs one `DRAWDOWN_BREAKER_STATE_LOAD_FAILED`
  (`missing`) warning and creates the file. A breaker restored open logs
  critical `DRAWDOWN_BREAKER_RESTORED_OPEN` (a trading-stop marker for
  journal alerts). Once tripped (8%) and flat, entries stay blocked across
  restarts until the operator stops the service, deletes the file and
  restarts; the owner accepted this. See `docs/OPERATIONS.md`.
- **Phase 4 (`claude/phase4-cleanup`):** dead settings, state fields,
  globals and unreachable branches removed (44 files); the retired keys are
  listed in `docs/OPERATIONS.md` ("Retired Settings Keys"). The operator
  `.env` still sets 29 of them; they are ignored and can be deleted after
  the deploy without changing the hash.
- **Validation:** full suite on the combined tree 6815 passed, 2 skipped;
  decision baseline unchanged
  (`COMBINED_DECISIONS 02fa7cd460ac3c191209e944b762b63b480773dd1b48112f46378a5a72dee83a`);
  each branch independently reviewed (no must-fix), should-fixes done and
  the reviewers' probes re-run on the final code.
- **Not covered:** the breaker follow-up commits had no second independent
  review; breaker fields are not in the health payload yet; ~60 unread env
  keys and ~13 Settings fields were not proven dead and remain.
- **Monday checks before deploying:** order count back to 2-8, one
  `POSITION_BACKSTOP_STATE_LOAD_FAILED` (`missing`) warning, backstop
  markers, no errors.
- **Rollback after deploy:** `git checkout --detach d22babac4`, restore the
  spec backup (it holds the old SHA and the old config hash),
  `sudo systemctl restart ai-trading.service`. The breaker state file can
  stay; `d22babac4` does not read it.

## November 22: exit pricing merged, not deployed (`da2570eb4`)

- **Running service:** still `d22babac4` (position backstop). Nothing was
  restarted. The owner approved deploying this together with the full-close
  fix (`db356eb49`) as one restart after the close on Mon Nov 23, provided
  the backstop's first market session looks clean.
- **Change (`claude/exit-pricing`, 5 commits):** a protective exit or a full
  close ("urgent exit", decided once in `netting_symbol_cycle.py`) goes out
  as a DAY limit priced to fill: sell at bid - 10 bps, cover at ask + 10 bps;
  with no usable quote a protective exit uses the bar close moved 50 bps
  (`ai_trading/core/netting_exit_pricing.py`, marker
  `URGENT_EXIT_LIMIT_PRICE`). A full close is blocked (`PRICE_COLLAR_BLOCK`)
  when the touch is beyond the collar from the bar close; a protective exit
  follows the quote up to 10%, then falls back. Protective exits only: stale
  quote, missing quote and `SLIPPAGE_CEILING_BLOCK` waived. Urgent exits:
  execution-quality gate waived. Every waiver requires a reducing order
  (`exit_waivers` in `oms/pretrade.py`). `compute_current_positions` parses
  fractional quantities (`POSITION_FRACTIONAL_QTY_TRUNCATED`). No settings
  or ConfigSpec changes.
- **Evidence:** two read-only traces (limit price; order/position read
  timing) found no defect in the full-close fix. November TCA records: 28
  sells submitted, 28 filled (paper).
- **Validation:** full suite on merged main 6777 passed, 2 skipped;
  `agent_validate_changed.sh` on the branch; independent review (no
  must-fix; three should-fix, all done); reviewer's 144-case prelude and
  submission comparison re-run on the final code: trims, adds and entries
  identical to the pre-change main. Decision baseline reference is now
  `COMBINED_DECISIONS 02fa7cd460ac3c191209e944b762b63b480773dd1b48112f46378a5a72dee83a`
  (`exit_pricing2`); `dccac997...` no longer applies.
- **Not covered:** the last two commits had no second independent review;
  no run against the broker, so better fills are unproven until paper runs.
- **Left as is:** a crossed quote blocks every exit (`QUOTE_SANITY_BLOCK`);
  a remainder under one share is never exited; `audit_positions` and
  `portfolio/core.py` still use `int(qty)`; sub-dollar names use a 0.01
  tick; quote age is measured from cycle start; real money: an unfilled
  closing order is never cancelled or repriced, and a failed post-submit
  sync leaves a stale open-order index for closing sells.
- **Crash alert:** installed Nov 22 (`OnFailure=` set, units match
  packaging); delivery not test-fired.
- **In progress (branches, not merged):** `claude/breaker-peak` (persist the
  drawdown breaker's peak and tripped state) and `claude/phase4-cleanup`
  (dead settings and code; one config-hash commit needing explicit owner
  approval at deploy).
- **Rollback after deploy:** `git checkout --detach d22babac4`, restore the
  spec backup, `sudo systemctl restart ai-trading.service`.

## November 21: full-close exit fix merged, not deployed (`db356eb49`)

- **Running service:** still `d22babac4` (position backstop). Nothing was
  restarted. Deploy of `db356eb49` is planned for after the close on Mon
  Nov 23, once the backstop's first market session looks clean (order count
  back to 2-8, backstop state loads); it needs the owner's go-ahead on the
  day.
- **Change (`claude/exit-gate-fix`, 8 commits):** an order that takes a
  position exactly to flat passes the entry-quality gates the way protective
  exits do (marker `CLOSING_EXIT_OVERRIDE`); partial trims, adds and entries
  are gated as before. A flip is cut to its exit leg before the gates run and
  the new side is judged from flat on a later bar. For protective exits and
  full closes, a soft throttle that rounds the order to zero no longer blocks
  it and the paper sampling floor no longer shrinks it. No settings changed;
  config hash untouched. Behaviour is in `docs/OPERATIONS.md`.
- **Evidence:** Nov 9-20 records had 8 blocked and 1 shrunk reductions, all
  full closes or flip exit legs. A 4,320-case differential against
  `00c2fa65c` (script in the session scratchpad, not kept) gave trims 864/864,
  adds, entries, holds and flats identical; protective exits differ only by
  the two fixes above.
- **Validation:** full suite on `db356eb49` 6,687 passed, 2 skipped. Decision
  baseline reference is now
  `dccac9978868dd3a7de64503e0e7a8c400eb2e97b5dddac833fe5ee80120975b`
  (`runs/exit_gate_fix_f6891059a`); the old `5bd1a138...` no longer applies.
- **Left as is:** `SPREAD_TOO_WIDE` (a bar-range measure) is bypassed for full
  closes; the thin cost multiplier still slices larger exits; hard risk blocks
  can still hold a full close; engine limit-price construction for exits and
  the timing of open-order versus position reads were not traced.
- **Research:** weak-close trial scored once and retired (Nov 21); its record
  and the opening-range trial are on main (`230906cb2`). Next direction is
  many individual stocks, waiting on a delisting-inclusive data source
  (Sharadar SEP preferred; Norgate needs Windows); the owner decides the
  subscription.
- **Still open:** the systemd `OnFailure` crash alert is not installed (two
  `sudo` commands in `docs/OPERATIONS.md`, owner to run).

## November 20: position backstop deployed (`d22babac4`)

- **Running service:** `d22babac4` since Fri Nov 20 21:24 UTC (market closed),
  owner-approved. It is `b0b2f8b59` (phase 3) plus the
  `claude/position-backstop` merge. Design and behaviour are in the position
  backstop section below and in `docs/OPERATIONS.md`.
- **As deployed:** 5% stop from the broker's average entry price, 5-session
  max hold, position cap from `max_portfolio_positions` (10 in the env), all
  as constants. Protective exits bypass the entry-quality gates, go out
  capped under the thin-order cap and on zero rolling volume, keep full size
  under soft throttles, and still respect halts, the kill switch, hard risk
  blocks, missing price and pretrade checks.
- **Phase 3's first session (Nov 20, on `b0b2f8b59`):** no errors beyond the
  standing `MODEL_REGISTRY_STALE`, no restarts, 89 decision records, 0 orders
  (every earlier session Nov 9-19 had 2-8). Only 8 long proposals that day
  against 13-28 on Nov 17-19, with the same gate set, so it reads as
  market-driven; not proven. Watch the order count on Nov 23.
- **Validation:** full suite on `d22babac4` after the close with the editable
  finder stripped: 6,489 passed, 2 skipped, 0 failed. `release_identity`
  passed both phases (config hash unchanged). Spec backup
  `runtime/release_spec.b0b2f8b59.bak.json` (SHA only).
- **Startup:** `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`,
  `DrawdownCircuitBreaker initialized`, `BROKER_SYNC` x3, 0 errors,
  NRestarts=0 after four minutes. `/healthz` 503 with only the preserved
  `required_model_stale` and `replay_live_parity_gate_failed`.
- **Not yet observed:** any backstop activity. The backstop loads on the
  first market-hours cycle (Mon Nov 23 14:30 UTC); expect one
  `POSITION_BACKSTOP_STATE_LOAD_FAILED` warning with reason `missing` then,
  and `runtime/position_backstop_state.json` once a position is held.
- **Not tested:** live-engine pricing and cancel/replace of an exit order;
  how often the stop would have fired historically.
- **Rollback:** `git checkout --detach b0b2f8b59`, restore the spec backup,
  `sudo systemctl restart ai-trading.service`.
- **Next:** stop entry-quality gates from blocking normal exits (owner
  approved as a separate change); phase 4 settings cleanup (config-hash
  change, needs explicit approval).

## November 19: legacy-path removal phase 3 deployed (`b0b2f8b59`)

- **Running service:** `b0b2f8b59` since Thu Nov 19 21:23 UTC (market closed),
  owner-approved. It is `d4803827b` (drawdown breaker) plus the
  `claude/legacy-path-phase3` merge. What phase 3 removed is in the phase 3
  section below.
- **Breaker's first session (Nov 19, on `d4803827b`):** normal. One error
  (the standing `MODEL_REGISTRY_STALE`), orders submitted, no restarts, no
  breaker trip.
- **Validation:** full suite on `b0b2f8b59` after the close with the editable
  finder stripped: 6,393 passed, 2 skipped, 0 failed. `release_identity`
  passed both phases against a scratch spec (config hash unchanged). Spec
  backup `runtime/release_spec.d4803827b.bak.json` (SHA only).
- **Startup:** `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`,
  `DrawdownCircuitBreaker initialized`, `BROKER_SYNC` x3, 0 errors, no import
  errors, NRestarts=0 after four minutes. `/healthz` 503 with only the
  preserved `required_model_stale` and `replay_live_parity_gate_failed`.
- **Not yet observed:** a market-hours cycle on this commit (first one is
  Fri Nov 20 14:30 UTC).
- **Rollback:** `git checkout --detach d4803827b`, restore the spec backup,
  `sudo systemctl restart ai-trading.service`.
- **Next:** position backstop (`claude/position-backstop`, built on phase 3,
  reviewed, full suite clean apart from a test this merge already deleted);
  deploy needs the owner's go-ahead.

## November 18: drawdown breaker deployed (`d4803827b`)

- **Running service:** `d4803827b` since Wed Nov 18 21:26 UTC (market closed),
  owner-approved. It is `8fca5a4b6` (phase 2 + Slack journal alerts) plus the
  `claude/netting-drawdown-breaker` merge and the Slack alert field fix
  (`current_drawdown` / `peak_equity`).
- **Breaker as deployed:** trips at drawdown >= 8% from the highest equity this
  process has seen, resets at <= 6.4% (`DRAWDOWN_RESET_FRACTION = 0.8`). While
  open, orders that add exposure are blocked (`DRAWDOWN_BREAKER_BLOCK`) and
  exits still go out. The sections below predate the reset margin; where they
  say "no recovery margin", this section wins.
- **Peak is in memory only.** It restarted from current equity at this
  restart. Persisting it across restarts is deferred by the owner ("later").
- **Validation:** full suite on `d4803827b` after the close: 6,660 passed,
  2 skipped, 0 failed. `release_identity` passed both phases against a scratch
  spec (config hash unchanged). Spec backup
  `runtime/release_spec.8fca5a4b6.bak.json` (SHA only).
- **Startup:** `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`,
  `DrawdownCircuitBreaker initialized`, `BROKER_SYNC` x4, 0 errors,
  NRestarts=0 after four minutes. `/healthz` 503 with only the preserved
  `required_model_stale` and `replay_live_parity_gate_failed`.
- **Not yet observed:** a market-hours cycle on this commit (first one is
  Thu Nov 19 14:30 UTC), and a real trip or reset.
- **Rollback:** `git checkout --detach 8fca5a4b6`, restore the spec backup,
  `sudo systemctl restart ai-trading.service`.
- **Next, each needing the owner's go-ahead to deploy:** phase 3
  (`claude/legacy-path-phase3`, full suite not yet run), then the position
  backstop (`claude/position-backstop`, in build).

## November 18: position backstop built, not deployed (`8bb0ec7ba`)

Branch `claude/position-backstop` from `f76f3c397` (phase 3 tip). Not pushed,
not merged, not deployed; no service restart.
- **What:** owner-approved safety backstops in the netting cycle
  (`ai_trading/core/netting_position_backstop.py`, constants only, no
  ConfigSpec/Settings/env change): 5% stop loss against broker
  `avg_entry_price`, maximum hold of 5 NYSE sessions, position cap on new
  openings at `cfg.max_portfolio_positions`. A fired backstop forces a flat
  target through the normal netting submission and locks the symbol until the
  next session open (`runtime/position_backstop_state.json`). Operator notes in
  `docs/OPERATIONS.md` ("Position Backstop").
- **Protective exit passes:** policy ablation, quarantine block mode, event
  blackout (also not re-raised to pretrade), adaptive sizing, approval soft
  blocks (net edge, spread, calibration, pacing, stale order), capacity
  throttle, portfolio optimizer; recorded as the original gate plus
  `PROTECTIVE_EXIT_OVERRIDE`. **Still respects:** halts (daily/weekly loss
  included), missing price, kill switch, bar dedup, participation cap,
  `LIQ_THIN_MAX_ORDER_BLOCK`, sell-to-available, duplicate intent, broker
  breakers, pretrade. Normal exits unchanged.
- **Not wired on purpose:** `state.position_entry_times` / `_position_opened_at`
  (would switch on the alpha time stop), `STOP_LOCK`, re-entry cooldown, the
  dead `AI_TRADING_LIVE_STOP_LOSS_BPS` / `AI_TRADING_MAX_POSITION_HOLD_SECONDS`.
- **Log markers:** per-symbol markers log as `MARKER | symbol=SYM`; the
  runtime log throttle drops identical messages within 5 s, so a bare marker
  lost every symbol after the first in a cycle. Journal alerts match the
  part before ` | `.
- **Decision parity:** phase-2 harness `backstop_412ffd8d0` and
  `backstop_8bb0ec7ba` give
  `COMBINED_DECISIONS 5bd1a138...dbdf0b192`; all six files identical to
  `post_phase2_4e9d951a1` (nothing arms in the harness).
- **Scenarios** (`/home/aiuser/ai-trading-baselines/backstop_scenarios`, a copy
  of the harness; runs 1, 2 and a third on `8bb0ec7ba` identical): stop fires on the AAPL dip, the
  exit sells 42 to 0 reduce-only one bar later (the arming bar was held by
  `LIQ_THIN_MAX_ORDER_BLOCK`), re-entry blocked the rest of the day, AAPL
  bought again next session; stop + kill switch: no orders, lock kept, one
  `EXIT_STALLED`; hold (constant 1): every held symbol exits on day 2's first
  bar, with `PROTECTIVE_EXIT_OVERRIDE` over `SPREAD_TOO_WIDE_BLOCK`,
  `NET_EDGE_FLOOR_BLOCK` and the capacity throttle; cap 2: never more than 2
  open, `POSITION_CAP_BLOCK` only on openings.
- **Validation:** 78 tests in `tests/test_netting_position_backstop.py`, 3 new
  in `tests/tools/test_journal_alerts.py`; `agent_validate_changed.sh
  --market-hours` (463 tests, live `/healthz`, incident snapshot); ruff, mypy,
  `typecheck_strict.sh`, vulture clean. Full suite after the close: 6465
  passed, 10 failed; 9 also fail on `f76f3c397` (research automation script,
  runtime backup sync, `test_issue_1_meta_learning_trigger_exists`), the tenth
  (an order-dependent short test of this branch) is fixed in `a82adfb07`.
- **Review follow-up (`1ea5dd677`, `80d1c98bf`, `bde6f5ec8`):** a protective
  exit is clipped to the thin-order cap instead of blocked (at least one
  share) and keeps its full size under approval soft throttles; a malformed
  store row drops only itself; a failed calendar lookup is not cached; a
  restart with a flipped side starts a `first_seen` episode; Slack sends one
  message per marker type listing symbols. Phase-2 baseline still
  `5bd1a138...`; scenario (a) now exits 34 of 42 on the arming bar and the
  last 8 one bar later (the thin-block delay is gone).
- **Zero-volume participation block (owner-approved, November 19):** a
  protective exit blocked only because rolling volume is zero or missing now
  goes out clipped to the thin-order dollar cap (at least one share), with
  `LIQ_PARTICIPATION_BLOCK` plus `PROTECTIVE_EXIT_OVERRIDE`; the thin rules do
  not clip it again. Positive-volume participation and normal orders are
  unchanged.
- **Still holds a protective exit:** positive-volume participation cap, hard
  risk blocks, a missing price and a missing quote (pretrade
  `DERISK_DATA_QUALITY_BLOCK`).
- **Next:** owner review, then a deploy decision. Check the release config
  hash before restarting (no ConfigSpec change was made).

## November 18: legacy-path removal phase 3 built, not deployed (`28292960e`)

Branch `claude/legacy-path-phase3` from `d4803827b` (main with the drawdown
breaker). Not pushed, not merged, not deployed.
- **Commits:** `e4910cda9` removes `ai_trading/position/` (no importer
  outside its tests); `877253b78` removes `core/trade_cycle.py` (reached only
  via `bot_engine.trade_logic`, which had no non-test caller), the
  `services.execution` trade-cycle and signal-order entry points, the
  `bot_engine` helpers only that cycle reached, their settings getters,
  `utils.base.validate_ohlcv`, `meta_learning.trigger_meta_learning_conversion`,
  `data/sanitize.py`, `data/corp_actions.py` and `portfolio/sizing.py`;
  `28292960e` removes the guards stale-symbol counter, `begin_cycle`,
  `end_cycle` and the worker's `guard_end_cycle` hook (`CYCLE_GATES` now logs
  only `shadow`).
- **Kept (live):** `submit_runtime.submit_order_runtime` and
  `services.execution.submit_order` (the netting submit path goes
  `bot_engine.submit_order` -> service -> runtime), `decision_journal`
  (used by `run_all_trades_execution`), `control_plane` (only its services
  label changed), `guards.shadow_active` (read by `live_trading`).
- **Size:** source 11 files deleted, 11 edited, 11,767 lines removed; tests
  31 files deleted, 34 edited, 8,553 lines removed (about 250 test functions).
- **Decision parity:** `phase3_28292960e` gives `COMBINED_DECISIONS
  5bd1a138...dbdf0b192` and `COMBINED_FULL 2ecc5ccd...32e877bb`; all six files
  identical to `post_phase2_4e9d951a1`. Coverage of the harness on
  `d4803827b` executed none of the removed functions or modules.
- **Validation (market open, so no full suite):** 462 targeted tests with the
  editable finder stripped; `agent_validate_changed.sh --market-hours`
  (295 tests, ruff, mypy, py_compile, live `/healthz`, incident snapshot);
  `typecheck_strict.sh`; `vulture --min-confidence 80`; import of all 446
  `ai_trading` modules from the worktree. No `CONFIG_SPECS`, Settings field or
  BotState field changed.
- **Phase 4 candidates (not touched):** BotState/BotContext fields and
  Settings fields that lost their only reader; live-env keys with no reader
  (`AI_TRADING_PROFITABILITY_GOVERNOR_*`, `AI_TRADING_FEED_RELIABILITY_*`,
  `AI_TRADING_PAPER_SAMPLING_*` trade limits); `circuit_breakers.manual_halt_trading`
  (no caller; breaker file is off limits); `guards.shadow_active` (no writer,
  always False); about 135 top-level functions already unreferenced before
  this phase; the `netting_enabled` ConfigSpec.
- **Note:** `ExecutionService.submit_order` raises
  `NonNettingLiveExecutionBlockedError` when the execution mode is `live`,
  and the netting pipeline submits through it. Paper is unaffected; review
  before any real-money switch.
- **Next:** owner review, full `pytest` after the close, then a deploy
  decision.

## November 17: drawdown breaker wired into the netting cycle, not deployed (`0e0abae3b`)

Branch `claude/netting-drawdown-breaker` from the phase 2 tip `8a3a07b18`.
Not pushed, not merged, not deployed; the running service is still `98f7b9d3b`.
- **Gap closed:** only the deleted legacy tail scored the drawdown breaker, so
  on the netting path it never saw equity and could not trip.
- **Change:** `_run_netting_cycle` calls `_score_drawdown_breaker` after the
  halt, market-closed, reconciliation and daily/weekly-loss returns. It feeds
  `DrawdownCircuitBreaker.score_account_snapshot` the execution engine's
  per-cycle account snapshot (`_get_account_snapshot`, already fetched by
  `start_cycle`; no extra broker call). `process_netting_symbol` applies the
  result after the stop lock, quarantine and kill switch: an order that
  increases exposure is recorded with gate `DRAWDOWN_BREAKER_BLOCK` and not
  submitted; a reducing order goes out; a reversal is cut to an exit and
  tagged `DRAWDOWN_BREAKER_REDUCE_ONLY`. It does not set `state.halt_trading`
  (that latches and stops exits). The gate is in `_NON_DISABLEABLE_GATE_NAMES`.
- **Logs:** `DRAWDOWN_BREAKER_TRIPPED` (critical), `DRAWDOWN_BREAKER_RESET`
  (warning), `DRAWDOWN_BREAKER_FEED_SKIPPED` (warning; equity missing,
  non-numeric, non-finite, zero or negative: state unchanged),
  `DRAWDOWN_BREAKER_ACCOUNT_SNAPSHOT_FAILED`, and once per day
  `DRAWDOWN_BREAKER_UNAVAILABLE` when the runtime has no breaker.
- **Thresholds with today's config:** `MAX_DRAWDOWN_THRESHOLD=0.08`. Trips at
  drawdown >= 8% from the highest equity this process has seen; resets as soon
  as drawdown is < 8% (the hard-coded `recovery_threshold=0.8` resolves to the
  same 8%, so there is no recovery margin). With the Nov 16 close equity of
  $81,655.06 as the peak, that is equity <= $75,122.65.
- **Restart:** the peak is in memory only and starts at the first valid
  reading, so a restart cannot trip it; it also closes an open breaker.
  `peak_equity.txt` (10000.0, Oct 18) is not read by the breaker.
- **Decision parity:** `breaker_0e0abae3b` gives `COMBINED_DECISIONS
  5bd1a138...dbdf0b192` and `COMBINED_FULL 2ecc5ccd...32e877bb`; all six files
  identical to `baseline_98f7b9d3b_run1`. The harness runtime has no breaker
  and no account snapshot, so this shows the cycle is unchanged when the
  breaker is absent; fed-and-closed and tripped cycles are covered by
  `tests/test_netting_drawdown_breaker.py` (44 tests).
- **Validation (market open, so no full suite):** 964 targeted tests passed
  with the editable finder stripped; `ruff check`, `mypy` on changed files,
  `scripts/typecheck_strict.sh`, `vulture --min-confidence 80` and
  `scripts/agent_validate_changed.sh --market-hours` passed. No `CONFIG_SPECS`
  change (`ai_trading/config/runtime.py` untouched).
- **Not covered:** full `pytest`; no service restart; no run against a real
  broker account object.
- **Open:** no recovery margin (flaps at the 8% line); state lost on restart;
  a cash withdrawal reads as drawdown. Sibling gaps not fixed: the
  acceptance-rate governor's symbol cap has no netting consumer, and
  `guards.begin_cycle` has no caller.
- **Next:** owner review with phase 2, full suite after the close, then a
  deploy decision.

## November 17: legacy-path removal phase 2 built, not deployed (`4e9d951a1`)
## November 17: legacy-path removal phase 2 deployed (`8a3a07b18`)

- **Deployed** Tue 21:29 UTC (closed, flat), owner-approved. Full `pytest` on
  `8a3a07b18`: 6,579 passed, 2 skipped, 0 failed. `release_identity` passed
  both phases with the unchanged config hash. Spec backup
  `runtime/release_spec.98f7b9d3b.bak.json`. First start:
  `IMPORT_PREFLIGHT_OK`, `ExecutionEngine initialized`, `BROKER_SYNC`, 0
  errors, NRestarts=0. Rollback: checkout `98f7b9d3b`, restore that backup,
  restart. Not yet observed: a market-hours session on this commit.
- The build notes below were written before the deploy.

Branch `claude/legacy-path-phase2` from main `783ec7d6f`. Not pushed, not
merged, not deployed; the running service is still `98f7b9d3b`.
- **Change:** `execute_run_all_trades_cycle` always runs the netting pipeline
  and no longer reads `netting_enabled`; the 509-line legacy tail is gone.
  Deleted with it: `core/strategy_cycle.py`, `core/prepare_run_runtime.py`,
  `core/position_risk_runtime.py`, `data/dynamic_universe.py`,
  `data/alpaca_screener.py`, and the legacy-only helpers in `bot_engine`
  (screening, legacy pricing, intent de-duplication, `_prepare_run`,
  `_process_symbols`, `run_multi_strategy`, `get_strategies`), `data/fetch`,
  `signals`, `utils/base`, `rl_trading/features`, `market/calendars`.
  Source: 17 files, 6,482 lines removed; tests: 56 files, 4,679 lines removed
  (25 test files deleted, 136 fewer collected tests in the changed files).
- **`run_trade()`** (`ai_trading/__main__.py`) now applies
  `_require_netting_pipeline` before its first cycle. It has a non-test
  caller, `scripts/runtime_artifacts_reset.sh`, so it was kept.
- **`STRATEGIES`** is no longer read and no strategy classes are loaded;
  `.env.example` and `docs/OPERATIONS.md` say so. Fifteen `.env.example` keys
  whose only reader was deleted were removed (screener, prepare/warm-up
  symbol limits, data-retry flatten, RL feature window, governor symbol cap).
- **Decision parity:** harness in `/home/aiuser/ai-trading-baselines/phase2_baseline`.
  `pre_phase2_783ec7d6f` and `post_phase2_4e9d951a1` both give
  `COMBINED_DECISIONS 5bd1a138...dbdf0b192` and `COMBINED_FULL 2ecc5ccd...32e877bb`,
  identical to `baseline_98f7b9d3b_run1` (372 cycles, 1,884 decision records,
  413 stub orders). The harness drives `_run_netting_cycle`; the cycle wrapper
  is covered by `tests/test_run_all_trades_execution.py`, including a new test
  that the cycle runs netting when the config says `netting_enabled=False`.
- **Validation (market open, so no full suite):** 1,161 targeted tests in 180
  files passed with the editable finder stripped; `ruff`, `mypy` on changed
  files, `scripts/typecheck_strict.sh`, `vulture --min-confidence 80`,
  `py_compile`, import of all 454 `ai_trading` modules from the worktree, and
  `scripts/agent_validate_changed.sh --market-hours` passed. No `CONFIG_SPECS`
  change; the resolved config hash is the same on both commits.
- **Kept, now without a runtime caller:** `ai_trading/position/` (2,589 lines,
  its only importer was a deleted legacy helper), `bot_engine.trade_logic`
  (bridge to the phase 3 chain), the allocator wiring, `guards.begin_cycle`,
  and the methods only the legacy cycle called (`DrawdownCircuitBreaker.update_equity`
  among them: the netting path never scored the drawdown breaker).
- **Not covered:** full `pytest` (run after the close); no service restart.
- **Next:** owner review and the full suite, then a deploy decision; decide
  on `ai_trading/position/`; phase 3 (`trade_cycle`, `services/execution`);
  phase 4 (the `netting_enabled` ConfigSpec, which changes the config hash).

## November 16: legacy-path removal phase 1 deployed (`98f7b9d3b`)

Branch `claude/legacy-path-phase1` (on top of `claude/unreachable-module-cleanup`).
- **Change:** `_fail_fast_env()` in `ai_trading/main.py` exits with
  `NETTING_DISABLED_UNSUPPORTED` when `netting_enabled` is false outside test
  mode; `_netting_pipeline_enabled()` and the root `conftest.py` no longer
  route tests to the legacy path by default. Seven legacy-path tests opt in
  with `AI_TRADING_NETTING_ENABLED=0`. No `CONFIG_SPECS` change; no legacy
  code deleted.
- **Monday session check of `fb2df3aee` (14:25-15:17 UTC):** normal; no new
  errors against Nov 12.
- **Validation:** full `pytest` 6,626 passed, 2 skipped, 1 failed
  (`tests/test_idempotency_cache_fallback.py::test_order_idempotency_cache_fallback`,
  the known frozen-clock failure, identical on `fb2df3aee`).
  `scripts/agent_validate_changed.sh --market-hours` passed.
- **Deployed** Mon 21:27 UTC (closed, flat), owner-approved. `release_identity`
  passed both phases with the unchanged config hash. Spec backup
  `runtime/release_spec.fb2df3aee.bak.json`. First start: `IMPORT_PREFLIGHT_OK`,
  `ExecutionEngine initialized`, `BROKER_SYNC`, 0 errors, NRestarts=0.
- **Rollback:** checkout `fb2df3aee`, restore that backup, restart.
- **Not covered:** `ai_trading/__main__.py::run_trade()` does not call the
  startup check (no non-test caller); handle in phase 2.
- **Next:** phase 2 (delete the legacy cycle) needs the owner's go-ahead and
  an offline replay before and after.

## November 15: two registered strategy trials run once each; both retired

Branch `claude/research-candidates`. Each trial was registered with the
owner's approval of plain-language rules, reviewed independently before its
single fit, and claimed once. Holdout unread; no orders; service untouched.
- `swing_reversal_trial_20261115`: retired. Long-only +12.0 bps per period
  over matched SPY at 5 bps, bootstrap lower bound -6.8, ahead in 5 of 10
  years. Record: `docs/RESEARCH_RESET_AMENDMENT_20261115_SWING_REVERSAL.md`.
- `intraday_momentum_trial_20261115` (SPY last half hour): retired. Long-only
  -1.55 bps per eligible day at 1 bp per side, negative before costs,
  positive in 1 of 10 years. Record:
  `docs/RESEARCH_RESET_AMENDMENT_20261115_INTRADAY_MOMENTUM.md`.
- Six strategies tested since the reset; none has a confident after-cost edge.

Older sections (through November 14, 2026): docs/archive/CODEX_HANDOFF_ARCHIVE_THROUGH_20261114.md
