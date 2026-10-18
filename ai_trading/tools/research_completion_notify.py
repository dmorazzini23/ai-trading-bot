"""Post research-automation completion summaries to Slack."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.config.managed_secrets import hydrate_managed_secrets
from ai_trading.monitoring import slack_format as fmt
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.symbol_promotion_comparison import symbol_promotion_digest


def _env_text(name: str, default: str = "") -> str:
    return str(get_env(name, default, cast=str, resolve_aliases=False) or default).strip()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _report_root(raw: str | Path | None = None) -> Path:
    configured = str(raw or _env_text("AI_TRADING_RESEARCH_REPORT_ROOT", "runtime/research_reports"))
    return resolve_runtime_artifact_path(
        configured,
        default_relative="runtime/research_reports",
        for_write=True,
    )


def _latest_path(report_root: Path, cadence: str, filename: str) -> Path:
    return report_root / "latest" / filename.format(cadence=cadence)


def _latest_research_automation_report(report_root: Path, cadence: str) -> dict[str, Any]:
    return _read_json(_latest_path(report_root, cadence, "{cadence}_research_automation_latest.json"))


def _step_output(report: Mapping[str, Any], step_name: str) -> Path | None:
    for row in list(report.get("steps") or []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("name") or "") != step_name:
            continue
        raw_path = str(row.get("output_path") or "").strip()
        if raw_path:
            return Path(raw_path)
    return None


def _step_statuses(report: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    failed: list[str] = []
    skipped: list[str] = []
    blocked: list[str] = []
    for row in list(report.get("step_results") or []):
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or "unknown")
        status = str(row.get("status") or "").lower()
        if status == "failed":
            failed.append(name)
        elif status == "skipped":
            skipped.append(name)
        elif status == "blocked":
            blocked.append(name)
    return failed, skipped, blocked


_PASSING_CHECK_STATUSES = frozenset({"passed", "complete", "ok", "ready", "healthy", "current"})


def _count(report: Mapping[str, Any], key: str) -> Any:
    row = report.get(key)
    return row.get("count", "n/a") if isinstance(row, Mapping) else "n/a"


def _status_next(report: Mapping[str, Any], next_key: str) -> str:
    if not report:
        return "n/a"
    return f"{fmt.humanize(report.get('status'))} → {fmt.humanize(report.get(next_key))}"


def _report_checks_text(health_summary: Mapping[str, Any]) -> str:
    """Summarize report checks as 'X of N OK' plus the ones that need attention."""

    attention: list[str] = []
    passing = 0
    for name, value in sorted(health_summary.items()):
        check_status = value.get("status") if isinstance(value, Mapping) else value
        if str(check_status).strip().lower() in _PASSING_CHECK_STATUSES:
            passing += 1
        elif check_status in (None, ""):
            attention.append(f"{fmt.humanize(name)}: no status")
        else:
            attention.append(f"{fmt.humanize(name)}: {fmt.humanize(check_status)}")
    heading = f"*Report checks* — {passing} of {len(health_summary)} OK"
    if not attention:
        return heading
    return f"{heading}; needs attention:\n{fmt.bullets(attention)}"


def build_research_completion_payload(
    *,
    cadence: str,
    workflow: str,
    exit_code: int,
    report_root: Path,
    channel: str,
    run_status: str = "",
) -> dict[str, Any]:
    cadence = str(cadence or "daily").strip().lower()
    workflow = str(workflow or cadence).strip().lower()
    run_status = str(run_status or "").strip().lower()
    candidate_report = _latest_research_automation_report(report_root, cadence)
    candidate_report_status = str(candidate_report.get("status") or "unknown")
    suppress_stale = (
        run_status in {"locked", "infrastructure_failed"}
        or (exit_code != 0 and candidate_report_status not in {"blocked", "failed"})
    )
    report = {} if suppress_stale else candidate_report
    summary = (
        {}
        if suppress_stale
        else _read_json(_latest_path(report_root, cadence, "{cadence}_operator_summary.json"))
    )
    daily = (
        _read_json(_latest_path(report_root, cadence, "daily_readiness_latest.json"))
        if cadence == "daily" and not suppress_stale
        else {}
    )
    readiness_path = _step_output(report, "live_capital_readiness")
    readiness = _read_json(readiness_path)
    failed, skipped, blocked_steps = _step_statuses(report)
    trading_day = (
        _read_json(_latest_path(report_root, cadence, "trading_day_latest.json"))
        if cadence == "daily" and not suppress_stale
        else {}
    )
    symbol_promotion = (
        _read_json(_latest_path(report_root, cadence, "symbol_promotion_latest.json"))
        if cadence == "daily" and not suppress_stale
        else {}
    )
    edge_calibration = (
        _read_json(_latest_path(report_root, cadence, "expected_edge_calibration_latest.json"))
        if cadence == "daily" and not suppress_stale
        else {}
    )
    evidence_starvation = (
        _read_json(_latest_path(report_root, cadence, "evidence_starvation_latest.json"))
        if cadence == "daily" and not suppress_stale
        else {}
    )
    report_status = str(report.get("status") or summary.get("status") or "unknown")
    report_config = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    run_id = str(report_config.get("run_id") or report.get("run_id") or "n/a")
    status = run_status or report_status
    if exit_code != 0 and report_status not in {"blocked", "failed"}:
        status = run_status or "failed"
    blocked = list(summary.get("blocked_reasons") or report.get("blocked_reasons") or [])
    if cadence == "daily" and isinstance(daily.get("blocked_reasons"), list):
        blocked.extend(str(item) for item in daily.get("blocked_reasons", []))
    readiness_status = str(readiness.get("status") or "n/a")
    recommended_mode = str(daily.get("recommended_next_session_mode") or "n/a")
    trade_allowed = daily.get("trade_allowed")
    health_summary = summary.get("health_report_summary")
    if not isinstance(health_summary, Mapping):
        health_summary = daily.get("health_report_summary") if isinstance(daily.get("health_report_summary"), Mapping) else {}
    operator_summary = summary.get("operator_summary")
    if not isinstance(operator_summary, Mapping):
        operator_summary = daily.get("operator_summary") if isinstance(daily.get("operator_summary"), Mapping) else {}
    weekend_schedule = report.get("weekend_schedule") if isinstance(report.get("weekend_schedule"), Mapping) else {}
    monday_preparation = report.get("monday_preparation") if isinstance(report.get("monday_preparation"), Mapping) else {}

    if status in {"failed", "infrastructure_failed"} or failed:
        severity = fmt.Severity.CRITICAL
    elif status in {"blocked", "locked"} or blocked_steps:
        severity = fmt.Severity.WARNING
    else:
        severity = fmt.Severity.OK
    title = f"Research {cadence} · {fmt.humanize(status)}"
    if recommended_mode != "n/a":
        title += f" · next session: {fmt.humanize(recommended_mode)}"

    blocks: list[dict[str, Any]] = [
        fmt.header_block(severity, title),
        fmt.fields_block(
            [
                ("Recommended mode", fmt.humanize(recommended_mode)),
                ("Trade allowed", fmt.yes_no(trade_allowed)),
                ("Live readiness", fmt.humanize(readiness_status)),
                ("Operator action", fmt.humanize(summary.get("operator_action"))),
            ]
        ),
        fmt.section_block(f"*Blocked reasons*\n{fmt.bullets(fmt.humanize(item) for item in blocked)}"),
    ]
    if trading_day:
        counts = " · ".join(
            f"{label} {_count(trading_day, key)}"
            for label, key in (
                ("desired", "desired_trades"),
                ("submitted", "submitted_trades"),
                ("fills", "realized_fills"),
                ("rejected", "rejected_trades"),
                ("controlled skips", "controlled_skips"),
            )
        )
        blocks.append(fmt.section_block(f"*Trading day*\n{counts}"))
    signal_fields: list[tuple[str, Any]] = []
    if cadence == "daily" and not suppress_stale:
        signal_fields = [
            ("Symbol promotion", symbol_promotion_digest(symbol_promotion).replace(":", " ").replace("_", " ")),
            ("Expected edge", _status_next(edge_calibration, "recommended_next_action")),
            ("Evidence starvation", _status_next(evidence_starvation, "recommendation")),
        ]
    if str(cadence).startswith("weekend-"):
        symbols = [item for item in str(weekend_schedule.get("effective_symbols") or "").split(",") if item.strip()]
        signal_fields.append(
            (
                "Weekend research (research only)",
                (
                    f"{len(symbols)} symbols ({fmt.inline_list(symbols, limit=5)}), "
                    f"≤{weekend_schedule.get('max_candidates', 'n/a')} candidates, "
                    f"≤{weekend_schedule.get('max_replay_candidates', 'n/a')} replays · Monday: "
                    f"{fmt.humanize(monday_preparation.get('recommended_operator_action'))}"
                ),
            )
        )
    if signal_fields:
        blocks.append(fmt.fields_block(signal_fields))
    steps = [
        f"{label} ({len(names)}): {fmt.inline_list((fmt.humanize(name) for name in names), limit=3)}"
        for label, names in (("Failed", failed), ("Blocked", blocked_steps), ("Skipped", skipped))
        if names
    ]
    if steps:
        blocks.append(fmt.section_block(f"*Steps*\n{fmt.bullets(steps)}"))
    if operator_summary and operator_summary.get("summary"):
        blocks.append(fmt.section_block(f"*Operator summary*\n{operator_summary.get('summary')}"))
    if health_summary:
        blocks.append(fmt.section_block(_report_checks_text(health_summary)))
    freshness = "stale latest artifacts suppressed" if suppress_stale else "current artifacts"
    report_paths = report.get("paths") if isinstance(report.get("paths"), Mapping) else {}
    report_path = report_paths.get("report")
    blocks.append(
        fmt.context_block(
            f"Run {run_id} · exit {exit_code} · {freshness}",
            f"Report: `{report_path}`" if report_path else "",
            f"Generated {fmt.now_label()}",
        )
    )
    text = fmt.fallback_text(
        severity,
        title,
        f"trade allowed: {fmt.yes_no(trade_allowed)}",
        f"{len(blocked)} blocked reasons",
    )
    payload = fmt.message(text, blocks)
    if channel:
        payload["channel"] = channel
    return payload


def _webhook_url(raw: str = "") -> str:
    return (
        str(raw or "").strip()
        or _env_text("AI_TRADING_RESEARCH_SLACK_WEBHOOK_URL")
        or _env_text("AI_TRADING_SLACK_WEBHOOK_URL")
        or _env_text("SLACK_WEBHOOK_URL")
    )


def _hydrate_webhook_secret() -> str:
    try:
        hydrate_managed_secrets()
    except RuntimeError as exc:  # pragma: no cover - environment/backend specific
        return f"{type(exc).__name__}: {exc}"
    return ""


def _post_slack_message(webhook_url: str, payload: Mapping[str, Any], timeout_s: float) -> int:
    request = urllib.request.Request(
        url=webhook_url,
        method="POST",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Slack webhook returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack webhook request failed: {exc.reason}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", default="daily")
    parser.add_argument("--workflow", default="")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--webhook-url", default="")
    parser.add_argument("--channel", default="")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-status", default="")
    args = parser.parse_args(argv)
    channel = str(args.channel or _env_text("AI_TRADING_RESEARCH_SLACK_CHANNEL", "#all-beatwallstreet")).strip()
    payload = build_research_completion_payload(
        cadence=str(args.cadence),
        workflow=str(args.workflow or args.cadence),
        exit_code=int(args.exit_code),
        report_root=_report_root(args.report_root),
        channel=channel,
        run_status=str(args.run_status),
    )
    webhook_url = _webhook_url(str(args.webhook_url))
    hydration_error = ""
    if not webhook_url and not str(args.webhook_url or "").strip():
        hydration_error = _hydrate_webhook_secret()
        webhook_url = _webhook_url()
    if args.dry_run:
        sys.stdout.write(json.dumps({"sent": False, "reason": "dry_run", "payload": payload}, sort_keys=True) + "\n")
        return 0
    if not webhook_url:
        reason = "secret_hydration_failed" if hydration_error else "missing_webhook"
        payload = {"sent": False, "reason": reason, "channel": channel}
        if hydration_error:
            payload["error"] = hydration_error
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0
    status_code = _post_slack_message(webhook_url, payload, timeout_s=float(args.timeout_s))
    sys.stdout.write(json.dumps({"sent": True, "status_code": status_code, "channel": channel}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
