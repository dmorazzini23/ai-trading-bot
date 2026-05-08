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


def _field(label: str, value: Any) -> dict[str, Any]:
    text = str(value if value not in (None, "") else "n/a")
    return {"type": "mrkdwn", "text": f"*{label}*\n{text[:1800]}"}


def _field_sections(fields: list[dict[str, Any]], *, chunk_size: int = 10) -> list[dict[str, Any]]:
    return [
        {"type": "section", "fields": fields[index : index + chunk_size]}
        for index in range(0, len(fields), chunk_size)
    ]


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
    status = run_status or report_status
    if exit_code != 0 and report_status not in {"blocked", "failed"}:
        status = run_status or "failed"
    failed_text = ", ".join(failed) if failed else "none"
    skipped_text = ", ".join(skipped) if skipped else "none"
    blocked_steps_text = ", ".join(blocked_steps) if blocked_steps else "none"
    blocked = list(summary.get("blocked_reasons") or report.get("blocked_reasons") or [])
    if cadence == "daily" and isinstance(daily.get("blocked_reasons"), list):
        blocked.extend(str(item) for item in daily.get("blocked_reasons", []))
    blocked_text = ", ".join(str(item) for item in blocked) if blocked else "none"
    readiness_status = str(readiness.get("status") or "n/a")
    recommended_mode = str(daily.get("recommended_next_session_mode") or "n/a")
    trade_allowed = daily.get("trade_allowed")
    health_summary = summary.get("health_report_summary")
    if not isinstance(health_summary, Mapping):
        health_summary = daily.get("health_report_summary") if isinstance(daily.get("health_report_summary"), Mapping) else {}
    openclaw_summary = summary.get("slack_openclaw_summary")
    if not isinstance(openclaw_summary, Mapping):
        openclaw_summary = daily.get("openclaw_summary") if isinstance(daily.get("openclaw_summary"), Mapping) else {}
    trading_counts = (
        f"desired={trading_day.get('desired_trades', {}).get('count', 'n/a')}, "
        f"submitted={trading_day.get('submitted_trades', {}).get('count', 'n/a')}, "
        f"rejected={trading_day.get('rejected_trades', {}).get('count', 'n/a')}, "
        f"fills={trading_day.get('realized_fills', {}).get('count', 'n/a')}"
    )
    symbol_promotion_text = symbol_promotion_digest(symbol_promotion)
    edge_text = (
        f"{edge_calibration.get('status', 'n/a')} / "
        f"{edge_calibration.get('recommended_next_action', 'n/a')}"
    )
    starvation_text = (
        f"{evidence_starvation.get('status', 'n/a')} / "
        f"{evidence_starvation.get('recommendation', 'n/a')}"
    )
    title = f"ai-trading research {cadence} finished: {status}"
    text = (
        f"{title}\n"
        f"Workflow: {workflow}\n"
        f"Exit code: {exit_code}\n"
        f"Live-capital readiness: {readiness_status}\n"
        f"Recommended mode: {recommended_mode}"
    )
    fields = [
        _field("Workflow", workflow),
        _field("Run status", run_status or "n/a"),
        _field("Exit code", exit_code),
        _field("Report status", report_status),
        _field("Artifact freshness", "stale_latest_suppressed" if suppress_stale else "current_latest"),
        _field("Operator action", summary.get("operator_action")),
        _field("Blocked reasons", blocked_text),
        _field("Failed steps", failed_text),
        _field("Blocked steps", blocked_steps_text),
        _field("Skipped steps", skipped_text),
        _field("Recommended mode", recommended_mode),
        _field("Live readiness", readiness_status),
        _field("Trade allowed", str(trade_allowed).lower() if trade_allowed is not None else "n/a"),
        _field("Trading day", trading_counts),
        _field("Symbol promotion", symbol_promotion_text),
        _field("Expected edge", edge_text),
        _field("Evidence starvation", starvation_text),
        _field("Health/report summary", json.dumps(health_summary, sort_keys=True) if health_summary else "n/a"),
        _field("OpenClaw summary", openclaw_summary.get("summary") if openclaw_summary else "n/a"),
        _field("Run report", report.get("paths", {}).get("report") if isinstance(report.get("paths"), Mapping) else None),
    ]
    payload: dict[str, Any] = {
        "text": text,
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": title[:150], "emoji": True}},
            *_field_sections(fields),
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"Channel target: {channel or '#all-beatwallstreet'} | "
                            f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}"
                        ),
                    }
                ],
            },
        ],
    }
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
