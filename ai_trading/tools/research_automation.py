"""Orchestrate recurring trading research automation runs.

The automation layer intentionally composes existing tools instead of adding
new model-promotion authority. Daily, weekly, and monthly runs create evidence
bundles; production model promotion and live-capital cutover remain manual and
gated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

_CADENCES = {"daily", "weekly", "monthly", "manual"}
_MANUAL_WORKFLOWS = {"promotion", "live-cutover", "incident-replay", "strategy-change"}


@dataclass(frozen=True)
class ResearchStep:
    """A single automation action."""

    name: str
    command: tuple[str, ...]
    purpose: str
    required: bool = False
    output_path: Path | None = None
    stdout_path: Path | None = None
    skip_if_missing: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    blocked_returncodes: tuple[int, ...] = ()

    def to_plan(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "required": self.required,
            "command": list(self.command),
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "stdout_path": str(self.stdout_path) if self.stdout_path is not None else None,
            "skip_if_missing": [str(path) for path in self.skip_if_missing],
            "metadata": dict(self.metadata),
            "blocked_returncodes": list(self.blocked_returncodes),
        }


@dataclass(frozen=True)
class ResearchConfig:
    cadence: str
    workflow: str
    report_root: Path
    run_dir: Path
    run_id: str
    symbols: str
    data_dir: Path | None
    shadow_jsonl: Path
    accepted_candidates_jsonl: Path | None
    model_path: Path | None
    manifest_path: Path | None
    current_champion_path: str
    report_date: str
    plan_only: bool
    dry_run: bool


def _now() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _now().isoformat().replace("+00:00", "Z")


def _default_market_report_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _env_text(name: str, default: str) -> str:
    return str(get_env(name, default, cast=str, resolve_aliases=False) or default).strip()


def _default_report_root() -> Path:
    configured = _env_text(
        "AI_TRADING_RESEARCH_REPORT_ROOT",
        "runtime/research_reports",
    )
    return resolve_runtime_artifact_path(
        configured,
        default_relative="runtime/research_reports",
        for_write=True,
    )


def _default_shadow_jsonl() -> Path:
    configured = _env_text(
        "AI_TRADING_ML_SHADOW_LOG_PATH",
        "runtime/ml_shadow_predictions.jsonl",
    )
    return resolve_runtime_artifact_path(
        configured,
        default_relative="runtime/ml_shadow_predictions.jsonl",
    )


def _default_symbols() -> str:
    canary = _env_text("AI_TRADING_CANARY_SYMBOLS", "")
    if canary:
        return canary
    universe = _env_text("AI_TRADING_SYMBOLS", "")
    if universe:
        return universe
    return "AAPL,AMZN"


def _maybe_data_dir(raw: str) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        value = _env_text("AI_TRADING_RESEARCH_DATA_DIR", "")
    if not value:
        return None
    return Path(value).expanduser()


def _run_id(cadence: str, configured: str) -> str:
    if configured.strip():
        return configured.strip()
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{cadence}"


def _python_module(module: str, *args: str | Path) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *(str(arg) for arg in args))


def _script(path: str | Path, *args: str | Path) -> tuple[str, ...]:
    return ("bash", str(path), *(str(arg) for arg in args))


def _runtime_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative, for_write=True)


def _runtime_input_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative, for_write=False)


def _daily_steps(config: ResearchConfig) -> list[ResearchStep]:
    live_cost = config.run_dir / "live_cost_model.json"
    shadow_report = config.run_dir / "ml_shadow_report.json"
    scorecard = config.run_dir / "symbol_universe_scorecard.json"
    decay = config.run_dir / "runtime_decay_controls.json"
    gonogo = config.run_dir / "runtime_gonogo_status.json"
    replay = config.run_dir / "replay_governance_summary.json"
    trading_day = config.run_dir / "trading_day_report.json"
    daily_research = config.run_dir / "daily_research_report.json"
    daily_research_md = config.run_dir / "daily_research_report.md"
    live_readiness = config.run_dir / "live_capital_readiness.json"
    memory_audit = config.run_dir / "memory_hotspot_audit.json"
    artifact_retention = config.run_dir / "runtime_artifact_retention.json"
    multi_horizon_dir = config.run_dir / "multi_horizon_lightweight"
    steps = [
        ResearchStep(
            name="refresh_runtime_reports",
            command=_script("scripts/refresh_runtime_reports.sh"),
            purpose="Refresh end-of-day runtime performance and gate reports.",
            required=False,
        ),
        ResearchStep(
            name="memory_hotspot_audit",
            command=_python_module(
                "ai_trading.tools.memory_hotspot_audit",
                "--output-json",
                memory_audit,
            ),
            purpose="Audit service memory, runtime artifact sizes, and likely whole-file reader hotspots.",
            output_path=memory_audit,
        ),
        ResearchStep(
            name="runtime_artifact_retention_plan",
            command=_python_module(
                "ai_trading.tools.runtime_artifact_retention",
                "--output-json",
                artifact_retention,
            ),
            purpose="Plan safe runtime JSONL retention without compacting evidence automatically.",
            output_path=artifact_retention,
            metadata={"mutates_runtime_artifacts": False, "apply_requires_operator_command": True},
        ),
        ResearchStep(
            name="live_cost_model",
            command=_python_module(
                "ai_trading.tools.live_cost_model",
                "--output-json",
                live_cost,
                "--window-minutes",
                "780",
                "--min-samples",
                "5",
            ),
            purpose="Aggregate current live execution and quote-cost evidence.",
            required=True,
            output_path=live_cost,
        ),
        ResearchStep(
            name="ml_shadow_report",
            command=_python_module(
                "ai_trading.tools.ml_shadow_report",
                "--input-jsonl",
                config.shadow_jsonl,
                "--output-json",
                shadow_report,
                "--frame-filter",
                "minute",
                "--provider-filter",
                "healthy-primary",
            ),
            purpose="Summarize current shadow-model evidence from live runtime telemetry.",
            output_path=shadow_report,
            skip_if_missing=(config.shadow_jsonl,),
        ),
        ResearchStep(
            name="replay_governance_refresh",
            command=_python_module(
                "ai_trading.tools.replay_governance",
                "--force",
                "--replay-output-dir",
                _runtime_path("runtime/replay_outputs"),
                "--summary-json",
                replay,
            ),
            purpose="Refresh replay governance evidence used by health and promotion gates.",
            output_path=replay,
            blocked_returncodes=(1, 2),
        ),
        ResearchStep(
            name="symbol_universe_scorecard",
            command=_python_module(
                "ai_trading.tools.symbol_universe_scorecard",
                "--live-cost-model-json",
                live_cost,
                "--shadow-report-json",
                shadow_report,
                "--executable-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--output-json",
                scorecard,
            ),
            purpose="Refresh symbol allow/shadow/disable scorecards.",
            output_path=scorecard,
        ),
        ResearchStep(
            name="runtime_decay_controls",
            command=_python_module(
                "ai_trading.tools.runtime_decay_controls",
                "--live-cost-model-json",
                live_cost,
                "--symbol-universe-scorecard-json",
                scorecard,
                "--output-json",
                decay,
            ),
            purpose="Refresh reversible throttle recommendations from the evidence bundle.",
            output_path=decay,
        ),
        ResearchStep(
            name="runtime_gonogo_status",
            command=_python_module("ai_trading.tools.runtime_gonogo_status", "--json"),
            purpose="Capture the daily go/no-go status as an artifact.",
            stdout_path=gonogo,
            blocked_returncodes=(2,),
        ),
        ResearchStep(
            name="trading_day_report",
            command=_python_module(
                "ai_trading.tools.trading_day_report",
                "--report-date",
                config.report_date,
                "--order-intents-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--shadow-jsonl",
                config.shadow_jsonl,
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--live-cost-model-json",
                live_cost,
                "--symbol-scorecard-json",
                scorecard,
                "--output-json",
                trading_day,
                "--latest-json",
                config.report_root / "latest" / "trading_day_latest.json",
                "--latest-md",
                config.report_root / "latest" / "trading_day_latest.md",
            ),
            purpose="Summarize desired/submitted/rejected trades and daily attribution.",
            output_path=trading_day,
        ),
        ResearchStep(
            name="daily_research_pipeline",
            command=_python_module(
                "ai_trading.tools.daily_research_pipeline",
                "--report-date",
                config.report_date,
                "--health-url",
                "http://127.0.0.1:9001/healthz",
                "--live-cost-model-json",
                live_cost,
                "--shadow-report-json",
                shadow_report,
                "--replay-governance-json",
                replay,
                "--symbol-scorecard-json",
                scorecard,
                "--runtime-gonogo-json",
                gonogo,
                "--memory-audit-json",
                memory_audit,
                "--artifact-retention-json",
                artifact_retention,
                "--output-json",
                daily_research,
                "--latest-json",
                config.report_root / "latest" / "daily_readiness_latest.json",
                "--output-md",
                daily_research_md,
            ),
            purpose="Produce the daily operator answer: can the system trade tomorrow, with what limits, and why.",
            output_path=daily_research,
        ),
        ResearchStep(
            name="live_capital_readiness",
            command=_python_module(
                "ai_trading.tools.live_capital_readiness",
                "--health-url",
                "http://127.0.0.1:9001/healthz",
                "--live-cost-model-json",
                live_cost,
                "--promotion-report-json",
                config.report_root / "latest" / "promotion_report_latest.json",
                "--canary-plan-json",
                daily_research,
                "--output-json",
                live_readiness,
                "--success-on-blocked",
            ),
            purpose="Create the live-capital cutover gate artifact without enabling live money.",
            output_path=live_readiness,
            metadata={"live_money_authority": False, "manual_approval_required": True},
        ),
    ]
    if config.data_dir is not None:
        steps.append(
            ResearchStep(
                name="multi_horizon_lightweight",
                command=_python_module(
                    "ai_trading.tools.multi_horizon_research_pipeline",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    config.symbols,
                    "--output-dir",
                    multi_horizon_dir,
                    "--horizons",
                    "1,15",
                    "--label-objectives",
                    "risk_adjusted",
                    "--lead-horizon-bars",
                    "15",
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                ),
                purpose="Train lightweight daily candidates without promotion authority.",
                output_path=multi_horizon_dir / "multi_horizon_research_report.json",
                skip_if_missing=(config.data_dir,),
                metadata={"promotion_authority": False},
            )
        )
    return steps


def _weekly_steps(config: ResearchConfig) -> list[ResearchStep]:
    live_cost = config.run_dir / "live_cost_model.json"
    multi_horizon_dir = config.run_dir / "multi_horizon_weekly"
    bridge = config.run_dir / "microstructure_replay_bridge.json"
    steps = [
        ResearchStep(
            name="live_cost_model",
            command=_python_module("ai_trading.tools.live_cost_model", "--output-json", live_cost),
            purpose="Pin the current observed cost model for weekly research.",
            output_path=live_cost,
        ),
    ]
    if config.data_dir is not None:
        steps.append(
            ResearchStep(
                name="multi_horizon_objective_search",
                command=_python_module(
                    "ai_trading.tools.multi_horizon_research_pipeline",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    config.symbols,
                    "--output-dir",
                    multi_horizon_dir,
                    "--horizons",
                    "1,3,5,15",
                    "--label-objectives",
                    "net_markout,spread_adjusted,risk_adjusted,mae_mfe",
                    "--lead-horizon-bars",
                    "15",
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                ),
                purpose="Search horizons/objectives for better signal, exit, and sizing evidence.",
                output_path=multi_horizon_dir / "multi_horizon_research_report.json",
                skip_if_missing=(config.data_dir,),
                metadata={"promotion_authority": False},
            )
        )
    if config.accepted_candidates_jsonl is not None:
        steps.append(
            ResearchStep(
                name="microstructure_replay_bridge",
                command=_python_module(
                    "ai_trading.tools.microstructure_replay_bridge",
                    "--shadow-jsonl",
                    config.shadow_jsonl,
                    "--accepted-candidates-jsonl",
                    config.accepted_candidates_jsonl,
                    "--output-json",
                    bridge,
                    "--match-time-of-day",
                    "--reject-missing",
                ),
                purpose="Join shadow quote telemetry to replay candidates before enforcing vetoes.",
                output_path=bridge,
                skip_if_missing=(config.shadow_jsonl, config.accepted_candidates_jsonl),
                metadata={"enforcement_authority": False},
            )
        )
    return steps


def _monthly_steps(config: ResearchConfig) -> list[ResearchStep]:
    steps = [
        ResearchStep(
            name="replay_governance_refresh",
            command=_python_module(
                "ai_trading.tools.replay_governance",
                "--force",
                "--replay-output-dir",
                _runtime_path("runtime/replay_outputs"),
                "--summary-json",
                config.run_dir / "replay_governance_summary.json",
            ),
            purpose="Refresh replay governance before monthly architecture review.",
            output_path=config.run_dir / "replay_governance_summary.json",
            blocked_returncodes=(1, 2),
        ),
    ]
    if config.data_dir is not None:
        for model_type in ("logistic", "hist_gradient"):
            out_dir = config.run_dir / f"walk_forward_{model_type}"
            steps.append(
                ResearchStep(
                    name=f"walk_forward_research_{model_type}",
                    command=_python_module(
                        "ai_trading.tools.multi_horizon_research_pipeline",
                        "--data-dir",
                        config.data_dir,
                        "--symbols",
                        config.symbols,
                        "--output-dir",
                        out_dir,
                        "--horizons",
                        "1,3,5,15",
                        "--label-objectives",
                        "net_markout,spread_adjusted,risk_adjusted,mae_mfe",
                        "--lead-horizon-bars",
                        "15",
                        "--model-type",
                        model_type,
                    ),
                    purpose="Challenge the signal architecture under a broader monthly research run.",
                    output_path=out_dir / "multi_horizon_research_report.json",
                    skip_if_missing=(config.data_dir,),
                    metadata={"promotion_authority": False, "model_type": model_type},
                )
            )
    steps.append(
        ResearchStep(
            name="live_capital_cutover_plan",
            command=_python_module(
                "ai_trading.tools.live_cutover_drill",
                "--execution-mode",
                "paper",
                "--output-json",
                config.run_dir / "live_cutover_paper_drill.json",
            ),
            purpose="Run a non-live cutover drill and produce a capital-profile readiness artifact.",
            output_path=config.run_dir / "live_cutover_paper_drill.json",
            metadata={"live_money_authority": False},
        )
    )
    return steps


def _manual_steps(config: ResearchConfig) -> tuple[list[ResearchStep], list[str]]:
    blocked: list[str] = []
    workflow = config.workflow
    if workflow == "promotion":
        if config.model_path is None:
            blocked.append("manual_promotion_requires_model_path")
            return [], blocked
        output = config.run_dir / "manual_promotion_report.json"
        command: list[str | Path] = [
            "--model-path",
            config.model_path,
            "--output-json",
            output,
            "--full-replay-json",
            _runtime_input_path("runtime/replay_governance_refresh_latest.json"),
            "--tail-replay-json",
            _runtime_input_path("runtime/replay_governance_refresh_latest.json"),
            "--recent-replay-json",
            _runtime_input_path("runtime/replay_governance_refresh_latest.json"),
            "--shadow-report-json",
            _runtime_input_path("runtime/ml_shadow_report_latest.json"),
            "--live-cost-model-json",
            _runtime_input_path("runtime/live_cost_model_latest.json"),
            "--runtime-decay-controls-json",
            _runtime_input_path("runtime/runtime_decay_controls_latest.json"),
        ]
        if config.manifest_path is not None:
            command.extend(["--manifest-path", config.manifest_path])
        if config.current_champion_path:
            command.extend(["--current-champion-path", config.current_champion_path])
        return [
            ResearchStep(
                name="manual_promotion_report",
                command=_python_module("ai_trading.tools.promotion_pipeline", *command),
                purpose="Generate a gated promotion report without mutating runtime model paths.",
                output_path=output,
                metadata={"manual_approval_required": True, "promotion_authority": False},
            )
        ], blocked
    if workflow == "live-cutover":
        output = config.run_dir / "manual_live_cutover_drill.json"
        return [
            ResearchStep(
                name="manual_live_cutover_drill",
                command=_python_module(
                    "ai_trading.tools.live_cutover_drill",
                    "--execution-mode",
                    "live",
                    "--output-json",
                    output,
                ),
                purpose="Generate a live-capital cutover readiness drill artifact.",
                output_path=output,
                metadata={"manual_approval_required": True, "live_money_authority": False},
            )
        ], blocked
    if workflow == "incident-replay":
        output = config.run_dir / "incident_replay_governance.json"
        gonogo = config.run_dir / "incident_runtime_gonogo.json"
        return [
            ResearchStep(
                name="incident_replay_governance",
                command=_python_module(
                    "ai_trading.tools.replay_governance",
                    "--force",
                    "--summary-json",
                    output,
                ),
                purpose="Recreate replay governance evidence for incident review.",
                output_path=output,
                blocked_returncodes=(1, 2),
            ),
            ResearchStep(
                name="incident_runtime_gonogo",
                command=_python_module("ai_trading.tools.runtime_gonogo_status", "--json"),
                purpose="Capture runtime go/no-go state for incident review.",
                stdout_path=gonogo,
                blocked_returncodes=(2,),
            ),
        ], blocked
    if workflow == "strategy-change":
        return _weekly_steps(config), blocked
    blocked.append(f"unsupported_manual_workflow:{workflow}")
    return [], blocked


def build_research_steps(config: ResearchConfig) -> tuple[list[ResearchStep], list[str]]:
    if config.cadence == "daily":
        return _daily_steps(config), []
    if config.cadence == "weekly":
        return _weekly_steps(config), []
    if config.cadence == "monthly":
        return _monthly_steps(config), []
    if config.cadence == "manual":
        return _manual_steps(config)
    return [], [f"unsupported_cadence:{config.cadence}"]


def _missing_inputs(step: ResearchStep) -> list[str]:
    return [str(path) for path in step.skip_if_missing if not path.exists()]


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _json_payload_from_stdout(text: str) -> dict[str, Any] | None:
    payload: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
    return payload


def _write_stdout_artifact(path: Path, stdout: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = _json_payload_from_stdout(stdout)
        if payload is not None:
            _write_json(path, payload)
            return
    path.write_text(stdout, encoding="utf-8")


def _run_step(step: ResearchStep) -> dict[str, Any]:
    missing = _missing_inputs(step)
    if missing:
        return {
            "name": step.name,
            "status": "skipped",
            "required": step.required,
            "reason": "missing_inputs",
            "missing_inputs": missing,
        }
    started = _iso_now()
    completed = started
    try:
        proc = subprocess.run(
            list(step.command),
            capture_output=True,
            text=True,
            check=False,
        )
        completed = _iso_now()
    except OSError as exc:
        return {
            "name": step.name,
            "status": "failed",
            "required": step.required,
            "started_at": started,
            "completed_at": _iso_now(),
            "returncode": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    if step.stdout_path is not None:
        _write_stdout_artifact(step.stdout_path, proc.stdout)
    if proc.returncode == 0:
        status = "passed"
    elif int(proc.returncode) in set(step.blocked_returncodes):
        status = "blocked"
    else:
        status = "failed"
    return {
        "name": step.name,
        "status": status,
        "required": step.required,
        "started_at": started,
        "completed_at": completed,
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "output_path": str(step.output_path) if step.output_path is not None else None,
        "stdout_path": str(step.stdout_path) if step.stdout_path is not None else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _copy_latest(report: Mapping[str, Any], config: ResearchConfig) -> Path:
    latest_dir = config.report_root / "latest"
    latest_path = latest_dir / f"{config.cadence}_research_latest.json"
    _write_json(latest_path, report)
    return latest_path


def _operator_summary(
    *,
    config: ResearchConfig,
    status: str,
    blocked_reasons: Iterable[str],
    step_results: Sequence[Mapping[str, Any]],
    latest_path: Path | None,
) -> dict[str, Any]:
    failed = [row["name"] for row in step_results if row.get("status") == "failed"]
    blocked = [row["name"] for row in step_results if row.get("status") == "blocked"]
    skipped = [row["name"] for row in step_results if row.get("status") == "skipped"]
    return {
        "artifact_type": "research_operator_summary",
        "generated_at": _iso_now(),
        "cadence": config.cadence,
        "workflow": config.workflow,
        "status": status,
        "blocked_reasons": list(blocked_reasons),
        "failed_steps": failed,
        "blocked_steps": blocked,
        "skipped_steps": skipped,
        "latest_report": str(latest_path) if latest_path is not None else None,
        "operator_action": _operator_action(status, config.cadence, config.workflow),
        "manual_gates": {
            "production_model_promotion": "manual_only",
            "live_money_cutover": "manual_only",
            "incident_response": "manual_review_required",
            "major_strategy_change": "manual_review_required",
        },
    }


def _operator_action(status: str, cadence: str, workflow: str) -> str:
    if status in {"planned", "dry_run"}:
        return "review_plan_then_run_when_ready"
    if status == "complete":
        if cadence == "manual" and workflow == "promotion":
            return "review_promotion_report_before_any_runtime_cutover"
        if cadence == "manual" and workflow == "live-cutover":
            return "review_cutover_drill_before_enabling_live_money"
        return "review_summary_and_generated_artifacts"
    if status == "blocked":
        return "resolve_blocked_reasons_then_rerun"
    return "inspect_failed_steps_before_restarting_automation"


def run_research_automation(config: ResearchConfig) -> dict[str, Any]:
    steps, blocked_reasons = build_research_steps(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.plan_only:
        status = "planned"
        step_results: list[dict[str, Any]] = []
    elif config.dry_run:
        status = "dry_run"
        step_results = []
    elif blocked_reasons:
        status = "blocked"
        step_results = []
    else:
        step_results = [_run_step(step) for step in steps]
        failed_steps = [row for row in step_results if row.get("status") == "failed"]
        blocked_steps = [row for row in step_results if row.get("status") == "blocked"]
        required_skipped = [
            row
            for row in step_results
            if row.get("required") and row.get("status") == "skipped"
        ]
        if failed_steps:
            status = "failed"
        elif blocked_steps or required_skipped:
            status = "blocked"
        else:
            status = "complete"

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "research_automation_report",
        "generated_at": _iso_now(),
        "cadence": config.cadence,
        "workflow": config.workflow,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "config": {
            "run_id": config.run_id,
            "run_dir": str(config.run_dir),
            "symbols": config.symbols,
            "data_dir": str(config.data_dir) if config.data_dir is not None else None,
            "shadow_jsonl": str(config.shadow_jsonl),
            "manual_model_path": str(config.model_path) if config.model_path is not None else None,
            "report_date": config.report_date,
        },
        "safety": {
            "production_model_promotion": "manual_only",
            "live_money_cutover": "manual_only",
            "automated_runtime_mutations": False,
            "slack_openclaw_source": "generated_artifacts",
        },
        "steps": [step.to_plan() for step in steps],
        "step_results": step_results,
    }
    report_path = config.run_dir / "research_automation_report.json"
    _write_json(report_path, report)
    latest_path = _copy_latest(report, config)
    summary = _operator_summary(
        config=config,
        status=status,
        blocked_reasons=blocked_reasons,
        step_results=step_results,
        latest_path=latest_path,
    )
    summary_path = config.run_dir / "operator_summary.json"
    _write_json(summary_path, summary)
    _write_json(config.report_root / "latest" / f"{config.cadence}_operator_summary.json", summary)
    report["paths"] = {
        "report": str(report_path),
        "operator_summary": str(summary_path),
        "latest_report": str(latest_path),
    }
    _write_json(report_path, report)
    _copy_latest(report, config)
    logger.info(
        "RESEARCH_AUTOMATION_COMPLETE",
        extra={
            "cadence": config.cadence,
            "workflow": config.workflow,
            "status": status,
            "report": str(report_path),
        },
    )
    return report


def _build_config(args: argparse.Namespace) -> ResearchConfig:
    cadence = str(args.cadence).strip().lower()
    workflow = str(args.workflow or "").strip().lower()
    if cadence != "manual":
        workflow = cadence
    elif not workflow:
        workflow = "promotion"
    report_root = Path(args.report_root).expanduser() if args.report_root else _default_report_root()
    run_id = _run_id(cadence, str(args.run_id or ""))
    run_dir = report_root / cadence / run_id
    data_dir = _maybe_data_dir(str(args.data_dir or ""))
    shadow_jsonl = Path(args.shadow_jsonl).expanduser() if args.shadow_jsonl else _default_shadow_jsonl()
    accepted = (
        Path(args.accepted_candidates_jsonl).expanduser()
        if args.accepted_candidates_jsonl
        else None
    )
    model_path = Path(args.model_path).expanduser() if args.model_path else None
    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else None
    return ResearchConfig(
        cadence=cadence,
        workflow=workflow,
        report_root=report_root,
        run_dir=run_dir,
        run_id=run_id,
        symbols=str(args.symbols or "").strip() or _default_symbols(),
        data_dir=data_dir,
        shadow_jsonl=shadow_jsonl,
        accepted_candidates_jsonl=accepted,
        model_path=model_path,
        manifest_path=manifest_path,
        current_champion_path=str(args.current_champion_path or "").strip(),
        report_date=str(args.report_date or "").strip() or _default_market_report_date(),
        plan_only=bool(args.plan_only),
        dry_run=bool(args.dry_run),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cadence", choices=sorted(_CADENCES))
    parser.add_argument("--workflow", choices=sorted(_MANUAL_WORKFLOWS), default="")
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--shadow-jsonl", type=Path, default=None)
    parser.add_argument("--accepted-candidates-jsonl", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--current-champion-path", type=str, default="")
    parser.add_argument("--report-date", type=str, default="")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)
    if config.cadence not in _CADENCES:
        parser.error(f"unsupported cadence: {config.cadence}")
    if config.cadence == "manual" and config.workflow not in _MANUAL_WORKFLOWS:
        parser.error(f"unsupported manual workflow: {config.workflow}")
    report = run_research_automation(config)
    sys.stdout.write(
        json.dumps(
            {
                "status": report["status"],
                "cadence": report["cadence"],
                "workflow": report["workflow"],
                "paths": report.get("paths", {}),
            },
            sort_keys=True,
        )
        + "\n"
    )
    if report["status"] in {"complete", "planned", "dry_run"}:
        return 0
    if report["status"] == "blocked":
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
