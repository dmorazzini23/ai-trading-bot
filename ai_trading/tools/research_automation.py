"""Orchestrate recurring trading research automation runs.

The automation layer intentionally composes existing tools instead of adding
new model-promotion authority. Daily, weekly, monthly, and weekend runs create
evidence bundles; production model promotion and live-capital cutover remain
manual and gated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

_WEEKEND_CADENCES = {"weekend-saturday", "weekend-sunday"}
_CADENCES = {"daily", "weekly", "monthly", "manual", *_WEEKEND_CADENCES}
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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _env_text(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return max(minimum, int(default))
    return max(minimum, value)


def _is_weekend_cadence(cadence: str) -> bool:
    return str(cadence or "").strip().lower() in _WEEKEND_CADENCES


def _weekend_enabled() -> bool:
    return _truthy(_env_text("AI_TRADING_WEEKEND_RESEARCH_ENABLED", "1"))


def _capped_symbols(symbols: str, max_symbols: int) -> str:
    seen: set[str] = set()
    capped: list[str] = []
    for raw in str(symbols or "").split(","):
        symbol = raw.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        capped.append(symbol)
        if len(capped) >= max_symbols:
            break
    return ",".join(capped)


def _weekend_runtime_limit_minutes(cadence: str) -> int:
    if cadence == "weekend-sunday":
        return _env_int("AI_TRADING_WEEKEND_VALIDATION_MAX_RUNTIME_MINUTES", 120, minimum=15)
    return _env_int("AI_TRADING_WEEKEND_RESEARCH_MAX_RUNTIME_MINUTES", 180, minimum=15)


def _weekend_cap_summary(config: ResearchConfig) -> dict[str, Any]:
    max_symbols = _env_int("AI_TRADING_WEEKEND_RESEARCH_MAX_SYMBOLS", 25, minimum=1)
    if config.cadence == "weekend-sunday":
        max_replay = _env_int("AI_TRADING_WEEKEND_VALIDATION_MAX_REPLAY_CANDIDATES", 20, minimum=1)
    else:
        max_replay = _env_int("AI_TRADING_WEEKEND_RESEARCH_MAX_REPLAY_CANDIDATES", 15, minimum=1)
    return {
        "enabled": _weekend_enabled(),
        "max_runtime_minutes": _weekend_runtime_limit_minutes(config.cadence),
        "max_symbols": max_symbols,
        "max_candidates": _env_int("AI_TRADING_WEEKEND_RESEARCH_MAX_CANDIDATES", 100, minimum=1),
        "max_replay_candidates": max_replay,
        "max_parallel_workers": _env_int("AI_TRADING_WEEKEND_RESEARCH_MAX_PARALLEL_WORKERS", 2, minimum=1),
        "cache_enabled": _truthy(_env_text("AI_TRADING_WEEKEND_RESEARCH_CACHE_ENABLED", "1")),
        "effective_symbols": _capped_symbols(config.symbols, max_symbols),
    }


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


def _truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_research_enabled() -> bool:
    return _truthy(_env_text("AI_TRADING_HF_RESEARCH_ENABLED", "0"))


def _hf_research_use_api_enabled() -> bool:
    return _truthy(_env_text("AI_TRADING_HF_RESEARCH_USE_API", "0"))


def _hf_research_command_flags() -> tuple[str, ...]:
    if not _hf_research_enabled():
        return ()
    if _hf_research_use_api_enabled():
        return ("--enabled", "--use-hf-api")
    return ("--enabled",)


def _maybe_data_dir(raw: str) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        value = _env_text("AI_TRADING_RESEARCH_DATA_DIR", "")
    if not value:
        return None
    target = Path(value).expanduser()
    if target.is_absolute():
        return target
    return resolve_runtime_artifact_path(
        target,
        default_relative=value,
        for_write=False,
    )


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


def _training_cache_args(config: ResearchConfig, cadence: str) -> tuple[str, ...]:
    if _is_weekend_cadence(config.cadence) and not _truthy(
        _env_text("AI_TRADING_WEEKEND_RESEARCH_CACHE_ENABLED", "1")
    ):
        return ()
    return (
        "--training-cache-dir",
        str(config.report_root / "latest" / "training_cache" / cadence),
    )


def _daily_steps(config: ResearchConfig) -> list[ResearchStep]:
    live_cost = config.run_dir / "live_cost_model.json"
    shadow_report = config.run_dir / "ml_shadow_report.json"
    scorecard = config.run_dir / "symbol_universe_scorecard.json"
    decay = config.run_dir / "runtime_decay_controls.json"
    gonogo = config.run_dir / "runtime_gonogo_status.json"
    replay = config.run_dir / "replay_governance_summary.json"
    replay_alignment = config.run_dir / "replay_live_cost_alignment.json"
    regime_throttle = config.run_dir / "regime_entry_throttle.json"
    expected_edge_calibration = config.run_dir / "expected_edge_calibration.json"
    execution_capture = config.run_dir / "execution_capture.json"
    evidence_starvation = config.run_dir / "evidence_starvation.json"
    trading_day = config.run_dir / "trading_day_report.json"
    symbol_promotion = config.run_dir / "symbol_promotion_comparison.json"
    symbol_lifecycle = config.run_dir / "symbol_lifecycle.json"
    counterfactual_execution = config.run_dir / "counterfactual_execution.json"
    portfolio_edge = config.run_dir / "portfolio_edge_control.json"
    decision_receipts = config.run_dir / "decision_receipts.json"
    model_registry = config.run_dir / "model_registry.json"
    experiment_ledger = config.run_dir / "experiment_ledger.json"
    pretrade_risk = config.run_dir / "pretrade_risk_control_verification.json"
    post_trade_surveillance = config.run_dir / "post_trade_surveillance.json"
    walk_forward_capital = config.run_dir / "walk_forward_capital_simulation.json"
    order_type_optimizer = config.run_dir / "order_type_optimizer.json"
    regime_champions = config.run_dir / "regime_champion_models.json"
    adversarial_failure = config.run_dir / "adversarial_failure_simulation.json"
    drift_monitor = config.run_dir / "model_data_drift_monitor.json"
    operator_control = config.run_dir / "operator_control_plane.json"
    hf_discovery = config.run_dir / "hf_discovery.json"
    hf_intake = config.run_dir / "hf_candidate_intake.json"
    hf_cache = config.run_dir / "hf_cache_materialization.json"
    daily_research = config.run_dir / "daily_research_report.json"
    daily_research_md = config.run_dir / "daily_research_report.md"
    live_readiness = config.run_dir / "live_capital_readiness.json"
    memory_audit = config.run_dir / "memory_hotspot_audit.json"
    artifact_retention = config.run_dir / "runtime_artifact_retention.json"
    multi_horizon_dir = config.run_dir / "multi_horizon_lightweight"
    training_accelerator = config.run_dir / "training_accelerator" / "training_accelerator_report.json"
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
            name="regime_entry_throttle_report",
            command=_python_module(
                "ai_trading.tools.regime_entry_throttle_report",
                "--report-date",
                config.report_date,
                "--live-cost-model-json",
                live_cost,
                "--output-json",
                regime_throttle,
            ),
            purpose="Summarize conservative session/regime entry throttle evidence.",
            output_path=regime_throttle,
            metadata={"enforcement_authority": False},
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
            name="replay_live_cost_alignment",
            command=_python_module(
                "ai_trading.tools.replay_live_cost_alignment_report",
                "--live-cost-model-json",
                live_cost,
                "--replay-report-json",
                replay,
                "--output-json",
                replay_alignment,
                "--min-samples",
                "5",
            ),
            purpose="Compare replay cost assumptions with observed live cost buckets.",
            output_path=replay_alignment,
            metadata={"promotion_authority": False, "live_money_authority": False},
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
            name="expected_edge_calibration_report",
            command=_python_module(
                "ai_trading.tools.expected_edge_calibration_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXPECTED_EDGE_CALIBRATION_MIN_SAMPLES", "25"),
                "--output-json",
                expected_edge_calibration,
                "--latest-json",
                config.report_root / "latest" / "expected_edge_calibration_latest.json",
            ),
            purpose="Diagnose whether expected edge is calibrated to realized fills and markouts.",
            output_path=expected_edge_calibration,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="execution_capture_report",
            command=_python_module(
                "ai_trading.tools.execution_capture_classification_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXECUTION_CAPTURE_MIN_SAMPLES", "10"),
                "--output-json",
                execution_capture,
                "--latest-json",
                config.report_root / "latest" / "execution_capture_latest.json",
            ),
            purpose="Classify whether execution preserved expected edge by symbol/session/regime.",
            output_path=execution_capture,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="evidence_starvation_report",
            command=_python_module(
                "ai_trading.tools.evidence_starvation_report",
                "--report-date",
                config.report_date,
                "--executable-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--order-intents-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--runtime-gonogo-json",
                gonogo,
                "--sample-target",
                _env_text("AI_TRADING_DIAGNOSTIC_SAMPLE_TARGET", "150"),
                "--output-json",
                evidence_starvation,
                "--latest-json",
                config.report_root / "latest" / "evidence_starvation_latest.json",
            ),
            purpose="Warn when throttling prevents enough paper evidence collection.",
            output_path=evidence_starvation,
            metadata={"promotion_authority": False, "live_money_authority": False},
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
                "--regime-entry-throttle-json",
                regime_throttle,
                "--expected-edge-calibration-json",
                expected_edge_calibration,
                "--execution-capture-json",
                execution_capture,
                "--counterfactual-execution-json",
                counterfactual_execution,
                "--portfolio-edge-json",
                portfolio_edge,
                "--decision-receipts-json",
                decision_receipts,
                "--model-registry-json",
                model_registry,
                "--pretrade-risk-json",
                pretrade_risk,
                "--post-trade-surveillance-json",
                post_trade_surveillance,
                "--experiment-ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--walk-forward-capital-json",
                walk_forward_capital,
                "--order-type-optimizer-json",
                order_type_optimizer,
                "--regime-champions-json",
                regime_champions,
                "--adversarial-failure-json",
                adversarial_failure,
                "--drift-monitor-json",
                drift_monitor,
                "--operator-control-plane-json",
                operator_control,
                "--huggingface-discovery-json",
                hf_discovery,
                "--huggingface-candidate-intake-json",
                hf_intake,
                "--huggingface-cache-json",
                hf_cache,
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
            name="counterfactual_execution_replay",
            command=_python_module(
                "ai_trading.tools.counterfactual_execution_replay_report",
                "--report-date",
                config.report_date,
                "--decisions-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--output-json",
                counterfactual_execution,
                "--latest-json",
                config.report_root / "latest" / "counterfactual_execution_latest.json",
            ),
            purpose="Estimate whether accepted and rejected decisions helped or hurt under historical evidence.",
            output_path=counterfactual_execution,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="symbol_promotion_comparison",
            command=_python_module(
                "ai_trading.tools.symbol_promotion_comparison",
                "--report-date",
                config.report_date,
                "--symbols",
                "AAPL,AMZN,MSFT",
                "--live-cost-model-json",
                live_cost,
                "--replay-report-json",
                replay,
                "--shadow-report-json",
                shadow_report,
                "--trading-day-json",
                trading_day,
                "--symbol-scorecard-json",
                scorecard,
                "--canary-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--output-json",
                symbol_promotion,
                "--latest-json",
                config.report_root / "latest" / "symbol_promotion_latest.json",
            ),
            purpose="Compare executable/canary/shadow symbols with manual-only promotion recommendations.",
            output_path=symbol_promotion,
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
        ResearchStep(
            name="symbol_lifecycle_report",
            command=_python_module(
                "ai_trading.tools.symbol_lifecycle_report",
                "--report-date",
                config.report_date,
                "--symbols",
                "AAPL,AMZN,MSFT",
                "--live-cost-model-json",
                live_cost,
                "--replay-report-json",
                replay,
                "--shadow-report-json",
                shadow_report,
                "--trading-day-json",
                trading_day,
                "--symbol-scorecard-json",
                scorecard,
                "--canary-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--output-json",
                symbol_lifecycle,
                "--latest-json",
                config.report_root / "latest" / "symbol_lifecycle_latest.json",
            ),
            purpose="Summarize manual-only symbol lifecycle recommendations.",
            output_path=symbol_lifecycle,
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
        ResearchStep(
            name="portfolio_edge_control",
            command=_python_module(
                "ai_trading.tools.portfolio_edge_control_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--expected-edge-calibration-json",
                expected_edge_calibration,
                "--output-json",
                portfolio_edge,
                "--latest-json",
                config.report_root / "latest" / "portfolio_edge_control_latest.json",
            ),
            purpose="Summarize portfolio-level edge controls and concentration risks.",
            output_path=portfolio_edge,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="decision_receipts_report",
            command=_python_module(
                "ai_trading.tools.decision_receipts_report",
                "--report-date",
                config.report_date,
                "--decisions-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--order-intents-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--output-json",
                decision_receipts,
                "--latest-json",
                config.report_root / "latest" / "decision_receipts_latest.json",
            ),
            purpose="Create operator-readable why-did-we-trade decision receipts from runtime logs.",
            output_path=decision_receipts,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="pretrade_risk_control_verifier",
            command=_python_module(
                "ai_trading.tools.pretrade_risk_control_verifier",
                "--report-date",
                config.report_date,
                "--intents-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--output-json",
                pretrade_risk,
                "--latest-json",
                config.report_root / "latest" / "pretrade_risk_control_verification_latest.json",
            ),
            purpose="Verify pre-trade risk controls fail closed before any live-capital authority.",
            output_path=pretrade_risk,
            blocked_returncodes=(2,),
            metadata={"live_money_authority": False, "fail_closed": True},
        ),
        ResearchStep(
            name="post_trade_surveillance",
            command=_python_module(
                "ai_trading.tools.post_trade_surveillance_report",
                "--report-date",
                config.report_date,
                "--decisions-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--orders-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--oms-jsonl",
                _runtime_input_path("runtime/oms_events.jsonl"),
                "--positions-json",
                _runtime_input_path("runtime/open_position_reconciliation_latest.json"),
                "--output-json",
                post_trade_surveillance,
                "--latest-json",
                config.report_root / "latest" / "post_trade_surveillance_latest.json",
            ),
            purpose="Detect harmful post-trade behavior without mutating runtime state.",
            output_path=post_trade_surveillance,
            metadata={"live_money_authority": False},
        ),
        ResearchStep(
            name="adversarial_failure_simulation",
            command=_python_module(
                "ai_trading.tools.adversarial_failure_simulation",
                "--report-date",
                config.report_date,
                "--pretrade-json",
                pretrade_risk,
                "--surveillance-json",
                post_trade_surveillance,
                "--output-json",
                adversarial_failure,
                "--latest-json",
                config.report_root / "latest" / "adversarial_failure_simulation_latest.json",
            ),
            purpose="Run non-live fail-closed simulations against generated safety artifacts.",
            output_path=adversarial_failure,
            metadata={"live_money_authority": False, "non_live": True},
        ),
        ResearchStep(
            name="walk_forward_capital_simulation",
            command=_python_module(
                "ai_trading.tools.walk_forward_capital_simulation",
                "--events-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--output-json",
                walk_forward_capital,
            ),
            purpose="Estimate paper/canary capital path and drawdown without enabling live capital.",
            output_path=walk_forward_capital,
            metadata={"live_money_authority": False, "live_enabled": False},
        ),
        ResearchStep(
            name="order_type_optimizer",
            command=_python_module(
                "ai_trading.tools.order_type_optimizer",
                "--candidates-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--live-cost-model-json",
                live_cost,
                "--output-json",
                order_type_optimizer,
            ),
            purpose="Produce shadow-only order-type recommendations from historical evidence.",
            output_path=order_type_optimizer,
            metadata={"live_money_authority": False, "enforcement_authority": False},
        ),
        ResearchStep(
            name="model_data_drift_monitor",
            command=_python_module(
                "ai_trading.tools.model_data_drift_monitor",
                "--baseline-json",
                config.report_root / "latest" / "model_data_drift_baseline.json",
                "--current-json",
                expected_edge_calibration,
                "--output-json",
                drift_monitor,
            ),
            purpose="Check model/data drift with stale-baseline fail-closed semantics.",
            output_path=drift_monitor,
            metadata={"live_money_authority": False},
        ),
        ResearchStep(
            name="model_registry_evaluation",
            command=_python_module(
                "ai_trading.tools.model_registry",
                "evaluate",
                "--registry-json",
                config.report_root / "latest" / "model_registry_latest.json",
                "--output-json",
                model_registry,
                "--latest-json",
                config.report_root / "latest" / "model_registry_evaluation_latest.json",
            ),
            purpose="Evaluate registry champion/challenger evidence without deploying models.",
            output_path=model_registry,
            skip_if_missing=(config.report_root / "latest" / "model_registry_latest.json",),
            blocked_returncodes=(2,),
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
        ResearchStep(
            name="huggingface_research_discovery",
            command=_python_module(
                "ai_trading.tools.huggingface_research_discovery",
                "--report-date",
                config.report_date,
                "--output-json",
                hf_discovery,
                "--latest-json",
                config.report_root / "latest" / "hf_discovery_latest.json",
                *_hf_research_command_flags(),
            ),
            purpose="Summarize research-only Hugging Face model/dataset candidates.",
            output_path=hf_discovery,
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "metadata_only": True,
                "non_authoritative": True,
                "requires_explicit_api_opt_in": True,
            },
        ),
        ResearchStep(
            name="huggingface_candidate_intake",
            command=_python_module(
                "ai_trading.tools.huggingface_candidate_intake",
                "--report-date",
                config.report_date,
                "--discovery-json",
                hf_discovery,
                "--ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--ledger-latest-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--output-json",
                hf_intake,
                "--latest-json",
                config.report_root / "latest" / "hf_candidate_intake_latest.json",
            ),
            purpose="Convert HF discoveries into manual offline research hypotheses.",
            output_path=hf_intake,
            skip_if_missing=(hf_discovery,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "manual_review_required": True,
            },
        ),
        ResearchStep(
            name="huggingface_cache_materialization_plan",
            command=_python_module(
                "ai_trading.tools.huggingface_cache_materializer",
                "--report-date",
                config.report_date,
                "--intake-json",
                hf_intake,
                "--dry-run",
                "--output-json",
                hf_cache,
                "--latest-json",
                config.report_root / "latest" / "hf_cache_materialization_latest.json",
            ),
            purpose="Plan optional HF offline cache materialization without downloading by default.",
            output_path=hf_cache,
            skip_if_missing=(hf_intake,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "downloads_enabled_by_default": False,
            },
        ),
        ResearchStep(
            name="operator_control_plane",
            command=_python_module(
                "ai_trading.tools.operator_control_plane",
                "--health-url",
                "http://127.0.0.1:9001/healthz",
                "--readiness-json",
                _runtime_input_path("runtime/live_capital_readiness_latest.json"),
                "--runtime-gonogo-json",
                gonogo,
                "--runtime-performance-json",
                _runtime_input_path("runtime/runtime_performance_report_latest.json"),
                "--oms-json",
                _runtime_input_path("runtime/oms_lifecycle_parity_latest.json"),
                "--model-registry-json",
                model_registry,
                "--latest-research-json",
                config.report_root / "latest" / "daily_readiness_latest.json",
                "--weekend-research-json",
                config.report_root / "latest" / "weekend_research_latest.json",
                "--drift-json",
                drift_monitor,
                "--surveillance-json",
                post_trade_surveillance,
                "--risk-verifier-json",
                pretrade_risk,
                "--paper-sampling-json",
                _runtime_input_path("runtime/paper_sampling_state_latest.json"),
                "--huggingface-research-json",
                hf_discovery,
                "--output-json",
                operator_control,
            ),
            purpose="Aggregate a read-only operator control-plane snapshot from artifacts.",
            output_path=operator_control,
            metadata={"read_only": True, "mutates_runtime": False},
        ),
        ResearchStep(
            name="trading_day_report_enriched",
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
                "--regime-entry-throttle-json",
                regime_throttle,
                "--expected-edge-calibration-json",
                expected_edge_calibration,
                "--execution-capture-json",
                execution_capture,
                "--counterfactual-execution-json",
                counterfactual_execution,
                "--portfolio-edge-json",
                portfolio_edge,
                "--decision-receipts-json",
                decision_receipts,
                "--model-registry-json",
                model_registry,
                "--pretrade-risk-json",
                pretrade_risk,
                "--post-trade-surveillance-json",
                post_trade_surveillance,
                "--experiment-ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--walk-forward-capital-json",
                walk_forward_capital,
                "--order-type-optimizer-json",
                order_type_optimizer,
                "--regime-champions-json",
                regime_champions,
                "--adversarial-failure-json",
                adversarial_failure,
                "--drift-monitor-json",
                drift_monitor,
                "--operator-control-plane-json",
                operator_control,
                "--weekend-research-json",
                config.report_root / "latest" / "weekend_research_latest.json",
                "--huggingface-discovery-json",
                hf_discovery,
                "--huggingface-candidate-intake-json",
                hf_intake,
                "--huggingface-cache-json",
                hf_cache,
                "--output-json",
                trading_day,
                "--latest-json",
                config.report_root / "latest" / "trading_day_latest.json",
                "--latest-md",
                config.report_root / "latest" / "trading_day_latest.md",
            ),
            purpose="Rewrite the trading-day report with all high-end evidence artifacts attached.",
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
                "--symbol-promotion-json",
                symbol_promotion,
                "--symbol-lifecycle-json",
                symbol_lifecycle,
                "--replay-live-cost-alignment-json",
                replay_alignment,
                "--regime-entry-throttle-json",
                regime_throttle,
                "--execution-capture-json",
                execution_capture,
                "--counterfactual-execution-json",
                counterfactual_execution,
                "--portfolio-edge-json",
                portfolio_edge,
                "--decision-receipts-json",
                decision_receipts,
                "--model-registry-json",
                model_registry,
                "--pretrade-risk-json",
                pretrade_risk,
                "--post-trade-surveillance-json",
                post_trade_surveillance,
                "--experiment-ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--walk-forward-capital-json",
                walk_forward_capital,
                "--order-type-optimizer-json",
                order_type_optimizer,
                "--regime-champions-json",
                regime_champions,
                "--adversarial-failure-json",
                adversarial_failure,
                "--drift-monitor-json",
                drift_monitor,
                "--operator-control-plane-json",
                operator_control,
                "--huggingface-discovery-json",
                hf_discovery,
                "--huggingface-candidate-intake-json",
                hf_intake,
                "--huggingface-cache-json",
                hf_cache,
                "--training-accelerator-json",
                training_accelerator,
                "--expected-edge-calibration-json",
                expected_edge_calibration,
                "--evidence-starvation-json",
                evidence_starvation,
                "--paper-sampling-state-json",
                _runtime_input_path("runtime/paper_sampling_state_latest.json"),
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
                "--edge-calibration-json",
                expected_edge_calibration,
                "--execution-capture-json",
                execution_capture,
                "--portfolio-edge-json",
                portfolio_edge,
                "--model-registry-json",
                model_registry,
                "--pretrade-risk-json",
                pretrade_risk,
                "--post-trade-surveillance-json",
                post_trade_surveillance,
                "--walk-forward-capital-json",
                walk_forward_capital,
                "--order-type-optimizer-json",
                order_type_optimizer,
                "--regime-champions-json",
                regime_champions,
                "--adversarial-failure-json",
                adversarial_failure,
                "--drift-monitor-json",
                drift_monitor,
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
        daily_research_index = next(
            (
                index
                for index, step in enumerate(steps)
                if step.name == "daily_research_pipeline"
            ),
            len(steps),
        )
        steps.insert(
            daily_research_index,
            ResearchStep(
                name="training_accelerator_daily",
                command=_python_module(
                    "ai_trading.tools.training_accelerator",
                    "--cadence",
                    "daily",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    config.symbols,
                    "--output-dir",
                    config.run_dir / "training_accelerator",
                    "--training-cache-dir",
                    config.report_root / "latest" / "training_cache" / "daily",
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                ),
                purpose="Refresh cached lightweight replay-aligned training candidates.",
                output_path=training_accelerator,
                skip_if_missing=(config.data_dir,),
                metadata={
                    "promotion_authority": False,
                    "uses_cached_training_features": True,
                    "stable_cache_root": str(config.report_root / "latest" / "training_cache" / "daily"),
                },
            ),
        )
        steps.insert(
            daily_research_index + 1,
            ResearchStep(
                name="regime_champion_models",
                command=_python_module(
                    "ai_trading.tools.regime_champion_models",
                    "--candidates-json",
                    training_accelerator,
                    "--current-registry-json",
                    config.report_root / "latest" / "model_registry_latest.json",
                    "--output-json",
                    regime_champions,
                    "--success-on-blocked",
                ),
                purpose="Evaluate regime-specific champion candidates with conservative fallback.",
                output_path=regime_champions,
                skip_if_missing=(training_accelerator,),
                metadata={
                    "promotion_authority": False,
                    "manual_approval_required": True,
                    "conservative_fallback": True,
                },
            ),
        )
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
    training_accelerator = config.run_dir / "training_accelerator" / "training_accelerator_report.json"
    bridge = config.run_dir / "microstructure_replay_bridge.json"
    hf_discovery = config.run_dir / "hf_discovery.json"
    hf_intake = config.run_dir / "hf_candidate_intake.json"
    hf_cache = config.run_dir / "hf_cache_materialization.json"
    steps = [
        ResearchStep(
            name="live_cost_model",
            command=_python_module("ai_trading.tools.live_cost_model", "--output-json", live_cost),
            purpose="Pin the current observed cost model for weekly research.",
            output_path=live_cost,
        ),
        ResearchStep(
            name="huggingface_research_discovery",
            command=_python_module(
                "ai_trading.tools.huggingface_research_discovery",
                "--report-date",
                config.report_date,
                "--output-json",
                hf_discovery,
                "--latest-json",
                config.report_root / "latest" / "hf_discovery_latest.json",
                *_hf_research_command_flags(),
            ),
            purpose="Discover research-only Hugging Face candidates for weekly experiments.",
            output_path=hf_discovery,
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "metadata_only": True,
                "non_authoritative": True,
                "requires_explicit_api_opt_in": True,
            },
        ),
        ResearchStep(
            name="huggingface_candidate_intake",
            command=_python_module(
                "ai_trading.tools.huggingface_candidate_intake",
                "--report-date",
                config.report_date,
                "--discovery-json",
                hf_discovery,
                "--ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--ledger-latest-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--output-json",
                hf_intake,
                "--latest-json",
                config.report_root / "latest" / "hf_candidate_intake_latest.json",
            ),
            purpose="Record HF candidates as manual offline experiment hypotheses.",
            output_path=hf_intake,
            skip_if_missing=(hf_discovery,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "manual_review_required": True,
            },
        ),
        ResearchStep(
            name="huggingface_cache_materialization_plan",
            command=_python_module(
                "ai_trading.tools.huggingface_cache_materializer",
                "--report-date",
                config.report_date,
                "--intake-json",
                hf_intake,
                "--dry-run",
                "--output-json",
                hf_cache,
                "--latest-json",
                config.report_root / "latest" / "hf_cache_materialization_latest.json",
            ),
            purpose="Plan optional HF cache downloads without downloading by default.",
            output_path=hf_cache,
            skip_if_missing=(hf_intake,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "downloads_enabled_by_default": False,
            },
        ),
    ]
    if config.data_dir is not None:
        steps.append(
            ResearchStep(
                name="training_accelerator_weekly",
                command=_python_module(
                    "ai_trading.tools.training_accelerator",
                    "--cadence",
                    "weekly",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    config.symbols,
                    "--output-dir",
                    config.run_dir / "training_accelerator",
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                ),
                purpose="Run the broader cached weekly horizon/objective candidate refresh.",
                output_path=training_accelerator,
                skip_if_missing=(config.data_dir,),
                metadata={"promotion_authority": False, "uses_cached_training_features": True},
            )
        )
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
    hf_discovery = config.run_dir / "hf_discovery.json"
    hf_intake = config.run_dir / "hf_candidate_intake.json"
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
        ResearchStep(
            name="huggingface_research_discovery",
            command=_python_module(
                "ai_trading.tools.huggingface_research_discovery",
                "--report-date",
                config.report_date,
                "--output-json",
                hf_discovery,
                "--latest-json",
                config.report_root / "latest" / "hf_discovery_latest.json",
                *_hf_research_command_flags(),
            ),
            purpose="Discover research-only HF candidates for monthly architecture review.",
            output_path=hf_discovery,
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "metadata_only": True,
                "non_authoritative": True,
                "requires_explicit_api_opt_in": True,
            },
        ),
        ResearchStep(
            name="huggingface_candidate_intake",
            command=_python_module(
                "ai_trading.tools.huggingface_candidate_intake",
                "--report-date",
                config.report_date,
                "--discovery-json",
                hf_discovery,
                "--ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--ledger-latest-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--output-json",
                hf_intake,
                "--latest-json",
                config.report_root / "latest" / "hf_candidate_intake_latest.json",
            ),
            purpose="Record monthly HF candidates as manual offline research hypotheses.",
            output_path=hf_intake,
            skip_if_missing=(hf_discovery,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "manual_review_required": True,
            },
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


def _weekend_saturday_steps(config: ResearchConfig) -> tuple[list[ResearchStep], list[str]]:
    if not _weekend_enabled():
        return [], ["weekend_research_disabled"]
    caps = _weekend_cap_summary(config)
    symbols = str(caps["effective_symbols"])
    live_cost = config.run_dir / "live_cost_model.json"
    shadow_report = config.run_dir / "ml_shadow_report.json"
    scorecard = config.run_dir / "symbol_universe_scorecard.json"
    expected_edge = config.run_dir / "expected_edge_calibration.json"
    execution_capture = config.run_dir / "execution_capture.json"
    symbol_promotion = config.run_dir / "symbol_promotion_comparison.json"
    symbol_lifecycle = config.run_dir / "symbol_lifecycle.json"
    training_accelerator = config.run_dir / "training_accelerator" / "training_accelerator_report.json"
    multi_horizon_dir = config.run_dir / "multi_horizon_weekend"
    hf_discovery = config.run_dir / "hf_discovery.json"
    hf_intake = config.run_dir / "hf_candidate_intake.json"
    hf_cache = config.run_dir / "hf_cache_materialization.json"
    model_registry = config.run_dir / "model_registry.json"
    steps = [
        ResearchStep(
            name="live_cost_model",
            command=_python_module("ai_trading.tools.live_cost_model", "--output-json", live_cost),
            purpose="Pin observed live costs before broad weekend research.",
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
            purpose="Refresh shadow-model evidence before weekend candidate search.",
            output_path=shadow_report,
            skip_if_missing=(config.shadow_jsonl,),
        ),
        ResearchStep(
            name="expected_edge_calibration_report",
            command=_python_module(
                "ai_trading.tools.expected_edge_calibration_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXPECTED_EDGE_CALIBRATION_MIN_SAMPLES", "25"),
                "--output-json",
                expected_edge,
                "--latest-json",
                config.report_root / "latest" / "expected_edge_calibration_latest.json",
            ),
            purpose="Refresh expected-edge calibration for broad weekend search.",
            output_path=expected_edge,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="execution_capture_report",
            command=_python_module(
                "ai_trading.tools.execution_capture_classification_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXECUTION_CAPTURE_MIN_SAMPLES", "10"),
                "--output-json",
                execution_capture,
                "--latest-json",
                config.report_root / "latest" / "execution_capture_latest.json",
            ),
            purpose="Refresh execution-capture evidence before weekend model search.",
            output_path=execution_capture,
            metadata={"promotion_authority": False, "live_money_authority": False},
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
            purpose="Refresh symbol scorecards before weekend expansion research.",
            output_path=scorecard,
        ),
        ResearchStep(
            name="symbol_promotion_comparison",
            command=_python_module(
                "ai_trading.tools.symbol_promotion_comparison",
                "--report-date",
                config.report_date,
                "--symbols",
                symbols,
                "--live-cost-model-json",
                live_cost,
                "--shadow-report-json",
                shadow_report,
                "--symbol-scorecard-json",
                scorecard,
                "--canary-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--output-json",
                symbol_promotion,
                "--latest-json",
                config.report_root / "latest" / "symbol_promotion_latest.json",
            ),
            purpose="Compare weekend symbols with manual-only promotion recommendations.",
            output_path=symbol_promotion,
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
        ResearchStep(
            name="symbol_lifecycle_report",
            command=_python_module(
                "ai_trading.tools.symbol_lifecycle_report",
                "--report-date",
                config.report_date,
                "--symbols",
                symbols,
                "--live-cost-model-json",
                live_cost,
                "--shadow-report-json",
                shadow_report,
                "--symbol-scorecard-json",
                scorecard,
                "--canary-symbols",
                _env_text("AI_TRADING_CANARY_SYMBOLS", ""),
                "--shadow-symbols",
                _env_text("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", ""),
                "--output-json",
                symbol_lifecycle,
                "--latest-json",
                config.report_root / "latest" / "symbol_lifecycle_latest.json",
            ),
            purpose="Summarize manual-only symbol lifecycle changes after broad weekend review.",
            output_path=symbol_lifecycle,
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
        ResearchStep(
            name="huggingface_research_discovery",
            command=_python_module(
                "ai_trading.tools.huggingface_research_discovery",
                "--report-date",
                config.report_date,
                "--output-json",
                hf_discovery,
                "--latest-json",
                config.report_root / "latest" / "hf_discovery_latest.json",
                *_hf_research_command_flags(),
            ),
            purpose="Discover research-only HF candidates during weekend research, if enabled.",
            output_path=hf_discovery,
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "metadata_only": True,
                "non_authoritative": True,
                "requires_explicit_api_opt_in": True,
            },
        ),
        ResearchStep(
            name="huggingface_candidate_intake",
            command=_python_module(
                "ai_trading.tools.huggingface_candidate_intake",
                "--report-date",
                config.report_date,
                "--discovery-json",
                hf_discovery,
                "--ledger-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--ledger-latest-json",
                config.report_root / "latest" / "experiment_ledger_latest.json",
                "--output-json",
                hf_intake,
                "--latest-json",
                config.report_root / "latest" / "hf_candidate_intake_latest.json",
            ),
            purpose="Convert weekend HF discoveries into manual offline research hypotheses.",
            output_path=hf_intake,
            skip_if_missing=(hf_discovery,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "manual_review_required": True,
            },
        ),
        ResearchStep(
            name="huggingface_cache_materialization_plan",
            command=_python_module(
                "ai_trading.tools.huggingface_cache_materializer",
                "--report-date",
                config.report_date,
                "--intake-json",
                hf_intake,
                "--dry-run",
                "--output-json",
                hf_cache,
                "--latest-json",
                config.report_root / "latest" / "hf_cache_materialization_latest.json",
            ),
            purpose="Plan optional HF cache materialization; downloads remain disabled by default.",
            output_path=hf_cache,
            skip_if_missing=(hf_intake,),
            metadata={
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
                "downloads_enabled_by_default": False,
            },
        ),
        ResearchStep(
            name="model_registry_evaluation",
            command=_python_module(
                "ai_trading.tools.model_registry",
                "evaluate",
                "--registry-json",
                config.report_root / "latest" / "model_registry_latest.json",
                "--output-json",
                model_registry,
                "--latest-json",
                config.report_root / "latest" / "model_registry_evaluation_latest.json",
            ),
            purpose="Evaluate registry champion/challenger evidence without deploying models.",
            output_path=model_registry,
            skip_if_missing=(config.report_root / "latest" / "model_registry_latest.json",),
            blocked_returncodes=(2,),
            metadata={"promotion_authority": False, "manual_approval_required": True},
        ),
    ]
    if config.data_dir is not None:
        cache_args = _training_cache_args(config, "weekend")
        steps.append(
            ResearchStep(
                name="training_accelerator_weekend_broad",
                command=_python_module(
                    "ai_trading.tools.training_accelerator",
                    "--cadence",
                    "weekly",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    symbols,
                    "--output-dir",
                    config.run_dir / "training_accelerator",
                    *cache_args,
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                    "--max-replay-candidates",
                    str(caps["max_replay_candidates"]),
                ),
                purpose="Run bounded broad weekend candidate refresh with cached features.",
                output_path=training_accelerator,
                skip_if_missing=(config.data_dir,),
                metadata={
                    "promotion_authority": False,
                    "uses_cached_training_features": bool(caps["cache_enabled"]),
                    "max_candidates": caps["max_candidates"],
                    "max_replay_candidates": caps["max_replay_candidates"],
                    "research_only": True,
                },
            )
        )
        steps.append(
            ResearchStep(
                name="multi_horizon_weekend_broad",
                command=_python_module(
                    "ai_trading.tools.multi_horizon_research_pipeline",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    symbols,
                    "--output-dir",
                    multi_horizon_dir,
                    "--horizons",
                    "1,3,5,15,30,60",
                    "--label-objectives",
                    "net_markout,spread_adjusted,risk_adjusted,mae_mfe",
                    "--lead-horizon-bars",
                    "15",
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                    "--max-replay-candidates",
                    str(caps["max_replay_candidates"]),
                    *(
                        (
                            "--training-cache",
                            "--training-cache-dir",
                            str(config.report_root / "latest" / "training_cache" / "weekend"),
                        )
                        if bool(caps["cache_enabled"])
                        else ("--no-training-cache",)
                    ),
                ),
                purpose="Run bounded multi-horizon weekend search with successive-halving replay.",
                output_path=multi_horizon_dir / "multi_horizon_research_report.json",
                skip_if_missing=(config.data_dir,),
                metadata={
                    "promotion_authority": False,
                    "research_only": True,
                    "max_symbols": caps["max_symbols"],
                    "max_candidates": caps["max_candidates"],
                    "max_replay_candidates": caps["max_replay_candidates"],
                    "manual_approval_required": True,
                },
            )
        )
    return steps, []


def _weekend_sunday_steps(config: ResearchConfig) -> tuple[list[ResearchStep], list[str]]:
    if not _weekend_enabled():
        return [], ["weekend_research_disabled"]
    caps = _weekend_cap_summary(config)
    live_cost = config.run_dir / "live_cost_model.json"
    replay = config.run_dir / "replay_governance_summary.json"
    replay_alignment = config.run_dir / "replay_live_cost_alignment.json"
    expected_edge = config.run_dir / "expected_edge_calibration.json"
    execution_capture = config.run_dir / "execution_capture.json"
    regime_throttle = config.run_dir / "regime_entry_throttle.json"
    counterfactual_execution = config.run_dir / "counterfactual_execution.json"
    walk_forward = config.run_dir / "walk_forward_capital_simulation.json"
    order_optimizer = config.run_dir / "order_type_optimizer.json"
    post_trade_surveillance = config.run_dir / "post_trade_surveillance.json"
    drift_monitor = config.run_dir / "model_data_drift_monitor.json"
    pretrade_risk = config.run_dir / "pretrade_risk_control_verification.json"
    operator_control = config.run_dir / "operator_control_plane.json"
    live_readiness = config.run_dir / "live_capital_readiness.json"
    steps = [
        ResearchStep(
            name="live_cost_model",
            command=_python_module("ai_trading.tools.live_cost_model", "--output-json", live_cost),
            purpose="Refresh observed costs before Sunday replay/readiness synthesis.",
            output_path=live_cost,
        ),
        ResearchStep(
            name="expected_edge_calibration_report",
            command=_python_module(
                "ai_trading.tools.expected_edge_calibration_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--gate-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXPECTED_EDGE_CALIBRATION_MIN_SAMPLES", "25"),
                "--output-json",
                expected_edge,
                "--latest-json",
                config.report_root / "latest" / "expected_edge_calibration_latest.json",
            ),
            purpose="Refresh calibration evidence before Monday preparation.",
            output_path=expected_edge,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="execution_capture_report",
            command=_python_module(
                "ai_trading.tools.execution_capture_classification_report",
                "--report-date",
                config.report_date,
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--min-samples",
                _env_text("AI_TRADING_EXECUTION_CAPTURE_MIN_SAMPLES", "10"),
                "--output-json",
                execution_capture,
                "--latest-json",
                config.report_root / "latest" / "execution_capture_latest.json",
            ),
            purpose="Refresh execution-capture evidence before Monday preparation.",
            output_path=execution_capture,
            metadata={"promotion_authority": False, "live_money_authority": False},
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
            purpose="Refresh replay governance for Monday-readiness synthesis.",
            output_path=replay,
            blocked_returncodes=(1, 2),
        ),
        ResearchStep(
            name="replay_live_cost_alignment",
            command=_python_module(
                "ai_trading.tools.replay_live_cost_alignment_report",
                "--live-cost-model-json",
                live_cost,
                "--replay-report-json",
                replay,
                "--output-json",
                replay_alignment,
                "--min-samples",
                "5",
            ),
            purpose="Compare replay costs with observed live cost buckets before Monday.",
            output_path=replay_alignment,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="regime_entry_throttle_report",
            command=_python_module(
                "ai_trading.tools.regime_entry_throttle_report",
                "--report-date",
                config.report_date,
                "--live-cost-model-json",
                live_cost,
                "--output-json",
                regime_throttle,
            ),
            purpose="Review opening/midday/closing throttle status before Monday.",
            output_path=regime_throttle,
            metadata={"enforcement_authority": False},
        ),
        ResearchStep(
            name="counterfactual_execution_replay",
            command=_python_module(
                "ai_trading.tools.counterfactual_execution_replay_report",
                "--report-date",
                config.report_date,
                "--decisions-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--output-json",
                counterfactual_execution,
                "--latest-json",
                config.report_root / "latest" / "counterfactual_execution_latest.json",
            ),
            purpose="Summarize whether recent gates helped or hurt under counterfactual evidence.",
            output_path=counterfactual_execution,
            metadata={"promotion_authority": False, "live_money_authority": False},
        ),
        ResearchStep(
            name="walk_forward_capital_simulation",
            command=_python_module(
                "ai_trading.tools.walk_forward_capital_simulation",
                "--events-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--output-json",
                walk_forward,
            ),
            purpose="Estimate Monday paper/canary capital path without enabling live capital.",
            output_path=walk_forward,
            metadata={"live_money_authority": False, "live_enabled": False},
        ),
        ResearchStep(
            name="order_type_optimizer",
            command=_python_module(
                "ai_trading.tools.order_type_optimizer",
                "--candidates-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--live-cost-model-json",
                live_cost,
                "--output-json",
                order_optimizer,
            ),
            purpose="Review shadow-only order-type recommendations before Monday.",
            output_path=order_optimizer,
            metadata={"live_money_authority": False, "enforcement_authority": False},
        ),
        ResearchStep(
            name="post_trade_surveillance",
            command=_python_module(
                "ai_trading.tools.post_trade_surveillance_report",
                "--report-date",
                config.report_date,
                "--decisions-jsonl",
                _runtime_input_path("runtime/gate_effectiveness.jsonl"),
                "--orders-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--fills-jsonl",
                _runtime_input_path("runtime/fill_events.jsonl"),
                "--oms-jsonl",
                _runtime_input_path("runtime/oms_events.jsonl"),
                "--positions-json",
                _runtime_input_path("runtime/open_position_reconciliation_latest.json"),
                "--output-json",
                post_trade_surveillance,
                "--latest-json",
                config.report_root / "latest" / "post_trade_surveillance_latest.json",
            ),
            purpose="Detect post-trade issues before Monday open.",
            output_path=post_trade_surveillance,
            metadata={"live_money_authority": False},
        ),
        ResearchStep(
            name="pretrade_risk_control_verifier",
            command=_python_module(
                "ai_trading.tools.pretrade_risk_control_verifier",
                "--report-date",
                config.report_date,
                "--intents-jsonl",
                _runtime_input_path("runtime/order_events.jsonl"),
                "--output-json",
                pretrade_risk,
                "--latest-json",
                config.report_root / "latest" / "pretrade_risk_control_verification_latest.json",
            ),
            purpose="Verify pre-trade controls are active before Monday preparation.",
            output_path=pretrade_risk,
            blocked_returncodes=(2,),
            metadata={"live_money_authority": False, "fail_closed": True},
        ),
        ResearchStep(
            name="model_data_drift_monitor",
            command=_python_module(
                "ai_trading.tools.model_data_drift_monitor",
                "--baseline-json",
                config.report_root / "latest" / "model_data_drift_baseline.json",
                "--current-json",
                expected_edge,
                "--output-json",
                drift_monitor,
            ),
            purpose="Review drift before Monday preparation.",
            output_path=drift_monitor,
            metadata={"live_money_authority": False},
        ),
        ResearchStep(
            name="operator_control_plane",
            command=_python_module(
                "ai_trading.tools.operator_control_plane",
                "--health-url",
                "http://127.0.0.1:9001/healthz",
                "--readiness-json",
                _runtime_input_path("runtime/live_capital_readiness_latest.json"),
                "--runtime-gonogo-json",
                _runtime_input_path("runtime/runtime_gonogo_status_latest.json"),
                "--runtime-performance-json",
                _runtime_input_path("runtime/runtime_performance_report_latest.json"),
                "--oms-json",
                _runtime_input_path("runtime/oms_lifecycle_parity_latest.json"),
                "--latest-research-json",
                config.report_root / "latest" / "daily_readiness_latest.json",
                "--weekend-research-json",
                config.report_root / "latest" / "weekend_research_latest.json",
                "--drift-json",
                drift_monitor,
                "--surveillance-json",
                post_trade_surveillance,
                "--risk-verifier-json",
                pretrade_risk,
                "--paper-sampling-json",
                _runtime_input_path("runtime/paper_sampling_state_latest.json"),
                "--output-json",
                operator_control,
            ),
            purpose="Build a read-only Monday operator control-plane snapshot from artifacts.",
            output_path=operator_control,
            metadata={"read_only": True, "mutates_runtime": False},
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
                config.report_root / "latest" / "daily_readiness_latest.json",
                "--edge-calibration-json",
                expected_edge,
                "--execution-capture-json",
                execution_capture,
                "--pretrade-risk-json",
                pretrade_risk,
                "--post-trade-surveillance-json",
                post_trade_surveillance,
                "--walk-forward-capital-json",
                walk_forward,
                "--order-type-optimizer-json",
                order_optimizer,
                "--drift-monitor-json",
                drift_monitor,
                "--output-json",
                live_readiness,
                "--success-on-blocked",
            ),
            purpose="Create a Sunday reporting-only live-capital readiness artifact.",
            output_path=live_readiness,
            metadata={"live_money_authority": False, "manual_approval_required": True},
        ),
    ]
    if config.data_dir is not None:
        steps.append(
            ResearchStep(
                name="training_accelerator_weekend_validation",
                command=_python_module(
                    "ai_trading.tools.training_accelerator",
                    "--cadence",
                    "weekly",
                    "--data-dir",
                    config.data_dir,
                    "--symbols",
                    str(caps["effective_symbols"]),
                    "--output-dir",
                    config.run_dir / "training_accelerator_validation",
                    *_training_cache_args(config, "weekend"),
                    "--live-cost-model-json",
                    live_cost,
                    "--use-live-cost-model",
                    "--max-replay-candidates",
                    str(caps["max_replay_candidates"]),
                ),
                purpose="Run bounded Sunday validation refresh for top replay candidates.",
                output_path=config.run_dir
                / "training_accelerator_validation"
                / "training_accelerator_report.json",
                skip_if_missing=(config.data_dir,),
                metadata={
                    "promotion_authority": False,
                    "research_only": True,
                    "max_replay_candidates": caps["max_replay_candidates"],
                },
            )
        )
    return steps, []


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
    if config.cadence == "weekend-saturday":
        return _weekend_saturday_steps(config)
    if config.cadence == "weekend-sunday":
        return _weekend_sunday_steps(config)
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


def _run_step(step: ResearchStep, *, timeout_seconds: float | None = None) -> dict[str, Any]:
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
            timeout=timeout_seconds,
        )
        completed = _iso_now()
    except subprocess.TimeoutExpired as exc:
        return {
            "name": step.name,
            "status": "failed",
            "required": step.required,
            "started_at": started,
            "completed_at": _iso_now(),
            "returncode": None,
            "reason": "timeout",
            "timeout_seconds": timeout_seconds,
            "stdout_tail": _tail(str(exc.stdout or "")),
            "stderr_tail": _tail(str(exc.stderr or "")),
        }
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


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _copy_automation_latest(report: Mapping[str, Any], config: ResearchConfig) -> Path:
    latest_dir = config.report_root / "latest"
    latest_path = latest_dir / f"{config.cadence}_research_automation_latest.json"
    _write_json(latest_path, report)
    if _is_weekend_cadence(config.cadence):
        _write_json(latest_dir / "weekend_research_latest.json", report)
    return latest_path


def _blocked_reasons_from_step_results(
    step_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for row in step_results:
        if row.get("status") != "blocked":
            continue
        name = str(row.get("name") or "unknown_step")
        candidates: list[Mapping[str, Any]] = []
        for key in ("output_path", "stdout_path"):
            raw_path = str(row.get(key) or "").strip()
            if not raw_path:
                continue
            try:
                parsed = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(parsed, Mapping):
                candidates.append(parsed)
        stdout_payload = _json_payload_from_stdout(str(row.get("stdout_tail") or ""))
        if stdout_payload is not None:
            candidates.append(stdout_payload)
        reason = None
        for payload in candidates:
            error = payload.get("error")
            error_message = error.get("message") if isinstance(error, Mapping) else None
            reason = (
                payload.get("reason")
                or payload.get("status_reason")
                or error_message
            )
            if reason not in (None, ""):
                break
        if reason in (None, ""):
            reason = str(row.get("reason") or row.get("returncode") or "blocked")
        reasons.append(f"{name}:{reason}")
    return reasons


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
    effective_blocked_reasons = list(blocked_reasons) + _blocked_reasons_from_step_results(
        step_results
    )
    artifact_summary = _next_level_artifact_summary(config)
    return {
        "artifact_type": "research_operator_summary",
        "generated_at": _iso_now(),
        "cadence": config.cadence,
        "workflow": config.workflow,
        "status": status,
        "blocked_reasons": effective_blocked_reasons,
        "failed_steps": failed,
        "blocked_steps": blocked,
        "skipped_steps": skipped,
        "latest_report": str(latest_path) if latest_path is not None else None,
        "operator_action": _operator_action(status, config.cadence, config.workflow),
        "health_report_summary": artifact_summary,
        "slack_openclaw_summary": {
            "service": "ai-trading-research-automation",
            "severity": "info"
            if status in {"complete", "planned", "dry_run"}
            else "warning",
            "summary": (
                f"research_automation cadence={config.cadence} "
                f"workflow={config.workflow} status={status}"
            ),
            "suggested_action": _operator_action(status, config.cadence, config.workflow),
            "blocked_reasons": effective_blocked_reasons,
            "next_level_artifacts": artifact_summary,
        },
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
        if cadence == "weekend-saturday":
            return "review_broad_research_then_wait_for_sunday_validation"
        if cadence == "weekend-sunday":
            return "review_monday_preparation_before_market_open"
        return "review_summary_and_generated_artifacts"
    if status == "blocked":
        return "resolve_blocked_reasons_then_rerun"
    return "inspect_failed_steps_before_restarting_automation"


def _artifact_status(payload: Mapping[str, Any], default: str = "missing") -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or default)
    return str(raw or default)


def _next_level_artifact_summary(config: ResearchConfig) -> dict[str, Any]:
    latest = config.report_root / "latest"
    daily = _read_json(latest / "daily_readiness_latest.json")
    trading_day = _read_json(latest / "trading_day_latest.json")
    live_readiness = _read_json(latest / "live_capital_readiness_latest.json")
    expected_edge = _read_json(latest / "expected_edge_calibration_latest.json")
    execution_capture = _read_json(latest / "execution_capture_latest.json")
    starvation = _read_json(latest / "evidence_starvation_latest.json")
    symbol_promotion = _read_json(latest / "symbol_promotion_latest.json")
    symbol_lifecycle = _read_json(latest / "symbol_lifecycle_latest.json")
    counterfactual_execution = _read_json(latest / "counterfactual_execution_latest.json")
    portfolio_edge = _read_json(latest / "portfolio_edge_control_latest.json")
    decision_receipts = _read_json(latest / "decision_receipts_latest.json")
    model_registry = _read_json(latest / "model_registry_latest.json") or _read_json(
        latest / "model_registry_evaluation_latest.json"
    )
    experiment_ledger = _read_json(latest / "experiment_ledger_latest.json")
    pretrade_risk = _read_json(latest / "pretrade_risk_control_verification_latest.json")
    post_trade_surveillance = _read_json(latest / "post_trade_surveillance_latest.json")
    walk_forward = _read_json(latest / "walk_forward_capital_simulation_latest.json")
    order_optimizer = _read_json(latest / "order_type_optimizer_latest.json")
    regime_champions = _read_json(latest / "regime_champion_models_latest.json")
    adversarial_failure = _read_json(latest / "adversarial_failure_simulation_latest.json")
    drift_monitor = _read_json(latest / "model_data_drift_monitor_latest.json")
    operator_control = _read_json(latest / "operator_control_plane_latest.json")
    hf_discovery = _read_json(latest / "hf_discovery_latest.json")
    hf_intake = _read_json(latest / "hf_candidate_intake_latest.json")
    hf_cache = _read_json(latest / "hf_cache_materialization_latest.json")
    weekend_research = _read_json(latest / "weekend_research_latest.json")
    weekend_summary = _read_json(latest / "weekend_operator_summary.json")
    return {
        "daily_research": {
            "status": _artifact_status(daily),
            "trade_allowed": daily.get("trade_allowed"),
            "recommended_next_session_mode": daily.get("recommended_next_session_mode"),
            "blocked_reasons": list(daily.get("blocked_reasons", []))
            if isinstance(daily.get("blocked_reasons"), list)
            else [],
        },
        "trading_day": {
            "status": _artifact_status(trading_day, "available" if trading_day else "missing"),
            "desired": trading_day.get("desired_trades", {}).get("count")
            if isinstance(trading_day.get("desired_trades"), Mapping)
            else None,
            "submitted": trading_day.get("submitted_trades", {}).get("count")
            if isinstance(trading_day.get("submitted_trades"), Mapping)
            else None,
            "rejected": trading_day.get("rejected_trades", {}).get("count")
            if isinstance(trading_day.get("rejected_trades"), Mapping)
            else None,
            "fills": trading_day.get("realized_fills", {}).get("count")
            if isinstance(trading_day.get("realized_fills"), Mapping)
            else None,
        },
        "live_capital_readiness": {
            "status": _artifact_status(live_readiness),
            "reasons": list(live_readiness.get("reasons", []))
            if isinstance(live_readiness.get("reasons"), list)
            else [],
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "expected_edge_calibration": {
            "status": _artifact_status(expected_edge),
            "recommended_next_action": expected_edge.get("recommended_next_action"),
        },
        "execution_capture": {
            "status": _artifact_status(execution_capture),
            "summary": execution_capture.get("summary"),
        },
        "evidence_starvation": {
            "status": _artifact_status(starvation),
            "recommendation": starvation.get("recommendation"),
        },
        "symbol_promotion": {
            "status": _artifact_status(symbol_promotion),
            "promotion_authority": bool(symbol_promotion.get("promotion_authority", False)),
            "runtime_symbol_gating_changed": bool(
                symbol_promotion.get("runtime_symbol_gating_changed", False)
            ),
        },
        "symbol_lifecycle": {
            "status": _artifact_status(symbol_lifecycle),
            "summary": symbol_lifecycle.get("summary"),
            "manual_approval_required_for_authority_increase": bool(
                symbol_lifecycle.get("manual_approval_required_for_authority_increase", True)
            ),
        },
        "counterfactual_execution": {
            "status": _artifact_status(counterfactual_execution),
            "summary": counterfactual_execution.get("summary"),
        },
        "portfolio_edge_control": {
            "status": _artifact_status(portfolio_edge),
            "output": portfolio_edge.get("output"),
        },
        "decision_receipts": {
            "status": _artifact_status(decision_receipts),
            "summary": decision_receipts.get("summary"),
        },
        "model_registry": {
            "status": _artifact_status(model_registry),
            "promotion_authority": bool(model_registry.get("promotion_authority", False)),
            "manual_approval_required": True,
        },
        "experiment_ledger": {
            "status": _artifact_status(experiment_ledger),
            "latest_run": experiment_ledger.get("latest_run"),
        },
        "pretrade_risk_control_verifier": {
            "status": _artifact_status(pretrade_risk),
            "fail_closed": bool(pretrade_risk.get("fail_closed", True)),
        },
        "post_trade_surveillance": {
            "status": _artifact_status(post_trade_surveillance),
            "summary": post_trade_surveillance.get("summary"),
        },
        "walk_forward_capital_simulation": {
            "status": _artifact_status(walk_forward),
            "summary": walk_forward.get("summary"),
            "live_enabled": bool(walk_forward.get("live_enabled", False)),
        },
        "order_type_optimizer": {
            "status": _artifact_status(order_optimizer),
            "summary": order_optimizer.get("summary"),
            "live_enabled": bool(order_optimizer.get("live_enabled", False)),
        },
        "regime_champion_models": {
            "status": _artifact_status(regime_champions),
            "summary": regime_champions.get("summary"),
            "manual_approval_required": True,
        },
        "adversarial_failure_simulation": {
            "status": _artifact_status(adversarial_failure),
            "summary": adversarial_failure.get("summary"),
            "live_money_authority": bool(adversarial_failure.get("live_money_authority", False)),
        },
        "model_data_drift_monitor": {
            "status": _artifact_status(drift_monitor),
            "summary": drift_monitor.get("summary"),
        },
        "operator_control_plane": {
            "status": _artifact_status(operator_control),
            "summary": operator_control.get("summary"),
            "read_only": bool(operator_control.get("read_only", True)),
        },
        "huggingface_research": {
            "status": _artifact_status(hf_discovery),
            "intake_status": _artifact_status(hf_intake),
            "cache_status": _artifact_status(hf_cache),
            "summary": hf_discovery.get("summary"),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "weekend_research": {
            "status": _artifact_status(weekend_research),
            "cadence": weekend_research.get("cadence"),
            "workflow": weekend_research.get("workflow"),
            "run_id": weekend_research.get("config", {}).get("run_id")
            if isinstance(weekend_research.get("config"), Mapping)
            else None,
            "operator_action": weekend_summary.get("operator_action"),
            "monday_preparation": weekend_research.get("monday_preparation"),
            "research_only": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
    }


def _artifact_generated_at(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    value = parsed.get("generated_at") or parsed.get("timestamp") or parsed.get("as_of")
    return str(value) if value not in (None, "") else None


def _step_result_by_name(step_results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("name") or ""): row for row in step_results}


def _evidence_manifest(
    steps: Sequence[ResearchStep],
    step_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_name = _step_result_by_name(step_results)
    manifest: list[dict[str, Any]] = []
    for step in steps:
        path = step.output_path or step.stdout_path
        result = by_name.get(step.name, {})
        manifest.append(
            {
                "step": step.name,
                "required": bool(step.required),
                "status": result.get("status", "planned"),
                "path": str(path) if path is not None else None,
                "exists": bool(path is not None and path.exists()),
                "generated_at": _artifact_generated_at(path),
            }
        )
    return manifest


def _copy_authority_artifacts(
    *,
    config: ResearchConfig,
    step_results: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    latest_dir = config.report_root / "latest"
    for row in step_results:
        if row.get("status") != "passed":
            continue
        name = str(row.get("name") or "")
        raw_path = str(row.get("output_path") or row.get("stdout_path") or "").strip()
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.exists():
            continue
        targets: list[Path] = []
        if name == "live_cost_model":
            targets.extend(
                [
                    latest_dir / "live_cost_model_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/live_cost_model_latest.json",
                        default_relative="runtime/live_cost_model_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "ml_shadow_report":
            targets.extend(
                [
                    latest_dir / "ml_shadow_report_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/ml_shadow_report_latest.json",
                        default_relative="runtime/ml_shadow_report_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "replay_governance_refresh":
            targets.extend(
                [
                    latest_dir / "replay_governance_refresh_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/replay_governance_refresh_latest.json",
                        default_relative="runtime/replay_governance_refresh_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "replay_live_cost_alignment":
            targets.extend(
                [
                    latest_dir / "replay_live_cost_alignment_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/replay_live_cost_alignment_latest.json",
                        default_relative="runtime/replay_live_cost_alignment_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "symbol_universe_scorecard":
            targets.extend(
                [
                    latest_dir / "symbol_universe_scorecard_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/symbol_universe_scorecard_latest.json",
                        default_relative="runtime/symbol_universe_scorecard_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "regime_entry_throttle_report":
            targets.extend(
                [
                    latest_dir / "regime_entry_throttle_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/regime_entry_throttle_latest.json",
                        default_relative="runtime/regime_entry_throttle_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "runtime_decay_controls":
            targets.extend(
                [
                    latest_dir / "runtime_decay_controls_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/runtime_decay_controls_latest.json",
                        default_relative="runtime/runtime_decay_controls_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "training_accelerator_daily":
            targets.extend(
                [
                    latest_dir / "training_accelerator_daily_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/training_accelerator_daily_latest.json",
                        default_relative="runtime/training_accelerator_daily_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name in {
            "training_accelerator_weekend_broad",
            "training_accelerator_weekend_validation",
        }:
            targets.extend(
                [
                    latest_dir / "training_accelerator_weekend_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/training_accelerator_weekend_latest.json",
                        default_relative="runtime/training_accelerator_weekend_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "expected_edge_calibration_report":
            targets.extend(
                [
                    latest_dir / "expected_edge_calibration_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/expected_edge_calibration_latest.json",
                        default_relative="runtime/reports/expected_edge_calibration_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "execution_capture_report":
            targets.extend(
                [
                    latest_dir / "execution_capture_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/execution_capture_latest.json",
                        default_relative="runtime/reports/execution_capture_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "evidence_starvation_report":
            targets.extend(
                [
                    latest_dir / "evidence_starvation_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/evidence_starvation_latest.json",
                        default_relative="runtime/reports/evidence_starvation_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name in {"trading_day_report", "trading_day_report_enriched"}:
            targets.extend(
                [
                    latest_dir / "trading_day_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/trading_day_latest.json",
                        default_relative="runtime/reports/trading_day_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "symbol_promotion_comparison":
            targets.extend(
                [
                    latest_dir / "symbol_promotion_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/research_reports/latest/symbol_promotion_latest.json",
                        default_relative="runtime/research_reports/latest/symbol_promotion_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "symbol_lifecycle_report":
            targets.extend(
                [
                    latest_dir / "symbol_lifecycle_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/research_reports/latest/symbol_lifecycle_latest.json",
                        default_relative="runtime/research_reports/latest/symbol_lifecycle_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "counterfactual_execution_replay":
            targets.extend(
                [
                    latest_dir / "counterfactual_execution_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/counterfactual_execution_latest.json",
                        default_relative="runtime/reports/counterfactual_execution_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "portfolio_edge_control":
            targets.extend(
                [
                    latest_dir / "portfolio_edge_control_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/portfolio_edge_control_latest.json",
                        default_relative="runtime/reports/portfolio_edge_control_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "decision_receipts_report":
            targets.extend(
                [
                    latest_dir / "decision_receipts_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/decision_receipts_latest.json",
                        default_relative="runtime/reports/decision_receipts_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "pretrade_risk_control_verifier":
            targets.extend(
                [
                    latest_dir / "pretrade_risk_control_verification_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/pretrade_risk_control_verification_latest.json",
                        default_relative="runtime/reports/pretrade_risk_control_verification_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "post_trade_surveillance":
            targets.extend(
                [
                    latest_dir / "post_trade_surveillance_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/reports/post_trade_surveillance_latest.json",
                        default_relative="runtime/reports/post_trade_surveillance_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "adversarial_failure_simulation":
            targets.extend(
                [
                    latest_dir / "adversarial_failure_simulation_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/adversarial_failure_simulation_latest.json",
                        default_relative="runtime/adversarial_failure_simulation_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "walk_forward_capital_simulation":
            targets.extend(
                [
                    latest_dir / "walk_forward_capital_simulation_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/walk_forward_capital_simulation_latest.json",
                        default_relative="runtime/walk_forward_capital_simulation_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "order_type_optimizer":
            targets.extend(
                [
                    latest_dir / "order_type_optimizer_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/order_type_optimizer_latest.json",
                        default_relative="runtime/order_type_optimizer_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "model_data_drift_monitor":
            targets.extend(
                [
                    latest_dir / "model_data_drift_monitor_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/model_data_drift_monitor_latest.json",
                        default_relative="runtime/model_data_drift_monitor_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "model_registry_evaluation":
            targets.extend(
                [
                    latest_dir / "model_registry_latest.json",
                    latest_dir / "model_registry_evaluation_latest.json",
                ]
            )
        elif name == "regime_champion_models":
            targets.extend(
                [
                    latest_dir / "regime_champion_models_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/regime_champion_models_latest.json",
                        default_relative="runtime/regime_champion_models_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "operator_control_plane":
            targets.extend(
                [
                    latest_dir / "operator_control_plane_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/operator_control_plane_latest.json",
                        default_relative="runtime/operator_control_plane_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "huggingface_research_discovery":
            targets.extend(
                [
                    latest_dir / "hf_discovery_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/research_reports/latest/hf_discovery_latest.json",
                        default_relative="runtime/research_reports/latest/hf_discovery_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "huggingface_candidate_intake":
            targets.extend(
                [
                    latest_dir / "hf_candidate_intake_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/research_reports/latest/hf_candidate_intake_latest.json",
                        default_relative="runtime/research_reports/latest/hf_candidate_intake_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "huggingface_cache_materialization_plan":
            targets.extend(
                [
                    latest_dir / "hf_cache_materialization_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/research_reports/latest/hf_cache_materialization_latest.json",
                        default_relative="runtime/research_reports/latest/hf_cache_materialization_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "daily_research_pipeline":
            targets.extend(
                [
                    latest_dir / "daily_research_latest.json",
                    latest_dir / "daily_readiness_latest.json",
                ]
            )
        elif name == "live_capital_readiness":
            targets.extend(
                [
                    latest_dir / "live_capital_readiness_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/live_capital_readiness_latest.json",
                        default_relative="runtime/live_capital_readiness_latest.json",
                        for_write=True,
                    ),
                ]
            )
        elif name == "manual_promotion_report":
            targets.extend(
                [
                    latest_dir / "promotion_report_latest.json",
                    resolve_runtime_artifact_path(
                        "runtime/promotion/promotion_report_latest.json",
                        default_relative="runtime/promotion/promotion_report_latest.json",
                        for_write=True,
                    ),
                ]
            )
        for target in targets:
            try:
                if row.get("status") == "blocked" and target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                logger.warning(
                    "RESEARCH_AUTHORITY_ARTIFACT_COPY_FAILED",
                    extra={"source": str(source), "target": str(target), "step": name},
                )
                continue
            copied[name] = str(target)
    return copied


def _write_experiment_ledger_for_run(
    *,
    config: ResearchConfig,
    status: str,
    report_path: Path,
    blocked_reasons: Sequence[str],
) -> Path:
    from ai_trading.tools.experiment_ledger import build_experiment_ledger

    ledger_status = {
        "complete": "success",
        "planned": "dry-run",
        "dry_run": "dry-run",
        "blocked": "blocked",
        "failed": "failed",
    }.get(status, "blocked")
    latest_path = config.report_root / "latest" / "experiment_ledger_latest.json"
    payload = build_experiment_ledger(
        run_id=config.run_id,
        workflow=f"{config.cadence}:{config.workflow}",
        status=ledger_status,
        conclusion=(
            "research automation completed"
            if status == "complete"
            else f"research automation {status}"
        ),
        input_paths=[report_path],
        config={
            "cadence": config.cadence,
            "workflow": config.workflow,
            "symbols": config.symbols,
            "report_date": config.report_date,
            "blocked_reasons": list(blocked_reasons),
        },
        previous_ledger=_read_json(latest_path),
        reported_complete=status == "complete",
        notes="Generated by research automation after final status calculation.",
    )
    run_path = config.run_dir / "experiment_ledger.json"
    _write_json(run_path, payload)
    _write_json(latest_path, payload)
    runtime_latest = resolve_runtime_artifact_path(
        "runtime/research_reports/latest/experiment_ledger_latest.json",
        default_relative="runtime/research_reports/latest/experiment_ledger_latest.json",
        for_write=True,
    )
    _write_json(runtime_latest, payload)
    return run_path


def _monday_preparation(
    status: str,
    config: ResearchConfig,
    weekend_caps: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not weekend_caps:
        return None
    if config.cadence == "weekend-saturday":
        action = (
            "review_broad_research_then_let_sunday_validation_synthesize_monday_readiness"
            if status in {"complete", "planned", "dry_run"}
            else "resolve_saturday_research_blockers_before_sunday_validation"
        )
    else:
        action = (
            "review_monday_preparation_before_market_open"
            if status in {"complete", "planned", "dry_run"}
            else "resolve_sunday_validation_blockers_before_market_open"
        )
    return {
        "question": "Can the system trade next session, with what limits, and why?",
        "status": status,
        "recommended_operator_action": action,
        "effective_symbols_reviewed": weekend_caps.get("effective_symbols"),
        "max_symbols": weekend_caps.get("max_symbols"),
        "max_candidates": weekend_caps.get("max_candidates"),
        "max_replay_candidates": weekend_caps.get("max_replay_candidates"),
        "manual_approval_required_for_authority_increase": True,
        "research_only": True,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def run_research_automation(config: ResearchConfig) -> dict[str, Any]:
    steps, blocked_reasons = build_research_steps(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    weekend_caps = _weekend_cap_summary(config) if _is_weekend_cadence(config.cadence) else {}
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
        step_results = []
        deadline = (
            monotonic() + float(int(weekend_caps["max_runtime_minutes"]) * 60)
            if weekend_caps
            else None
        )
        for step in steps:
            timeout_seconds: float | None = None
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    step_results.append(
                        {
                            "name": step.name,
                            "status": "failed",
                            "required": step.required,
                            "reason": "weekend_runtime_cap_exhausted",
                            "timeout_seconds": 0,
                            "output_path": str(step.output_path)
                            if step.output_path is not None
                            else None,
                        }
                    )
                    continue
                timeout_seconds = max(1.0, remaining)
            step_results.append(_run_step(step, timeout_seconds=timeout_seconds))
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

    safety: dict[str, Any] = {
        "production_model_promotion": "manual_only",
        "live_money_cutover": "manual_only",
        "automated_runtime_mutations": False,
        "slack_openclaw_source": "generated_artifacts",
    }
    if _is_weekend_cadence(config.cadence):
        safety["weekend_research_authority"] = "research_only"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "research_automation_report",
        "generated_at": _iso_now(),
        "authority": {
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "research_only": True,
            "manual_approval_required_for_authority_increase": True,
        },
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
        "weekend_schedule": weekend_caps if weekend_caps else None,
        "monday_preparation": _monday_preparation(status, config, weekend_caps),
        "safety": safety,
        "steps": [step.to_plan() for step in steps],
        "step_results": step_results,
        "evidence_manifest": _evidence_manifest(steps, step_results),
    }
    report_path = (
        config.run_dir / "weekend_research_report.json"
        if _is_weekend_cadence(config.cadence)
        else config.run_dir / "research_automation_report.json"
    )
    _write_json(report_path, report)
    latest_path = _copy_automation_latest(report, config)
    authority_copies = _copy_authority_artifacts(config=config, step_results=step_results)
    ledger_path = _write_experiment_ledger_for_run(
        config=config,
        status=status,
        report_path=report_path,
        blocked_reasons=blocked_reasons,
    )
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
    if _is_weekend_cadence(config.cadence):
        _write_json(config.report_root / "latest" / "weekend_operator_summary.json", summary)
    report["paths"] = {
        "report": str(report_path),
        "operator_summary": str(summary_path),
        "latest_report": str(latest_path),
        "experiment_ledger": str(ledger_path),
        "authority_copies": authority_copies,
    }
    _write_json(report_path, report)
    _copy_automation_latest(report, config)
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
    if args.report_root:
        raw_report_root = Path(args.report_root).expanduser()
        report_root = (
            raw_report_root
            if raw_report_root.is_absolute()
            else resolve_runtime_artifact_path(
                raw_report_root,
                default_relative=str(raw_report_root),
                for_write=True,
            )
        )
    else:
        report_root = _default_report_root()
    run_id = _run_id(cadence, str(args.run_id or ""))
    run_dir = report_root / ("weekend" if _is_weekend_cadence(cadence) else cadence) / run_id
    data_dir = _maybe_data_dir(str(args.data_dir or ""))
    if args.shadow_jsonl:
        raw_shadow = Path(args.shadow_jsonl).expanduser()
        shadow_jsonl = (
            raw_shadow
            if raw_shadow.is_absolute()
            else resolve_runtime_artifact_path(
                raw_shadow,
                default_relative=str(raw_shadow),
                for_write=False,
            )
        )
    else:
        shadow_jsonl = _default_shadow_jsonl()
    accepted = (
        (
            Path(args.accepted_candidates_jsonl).expanduser()
            if Path(args.accepted_candidates_jsonl).expanduser().is_absolute()
            else resolve_runtime_artifact_path(
                Path(args.accepted_candidates_jsonl).expanduser(),
                default_relative=str(args.accepted_candidates_jsonl),
                for_write=False,
            )
        )
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
