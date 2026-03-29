"""Prometheus/Grafana metrics MCP server for execution trend queries."""

from __future__ import annotations

import importlib
import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_unix() -> int:
    return int(time.time())


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout_s: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method="GET")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise RuntimeError("metrics endpoint returned non-object JSON")
            return payload
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"metrics endpoint HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"metrics endpoint request failed: {exc.reason}") from exc


def _resolve_backend(args: dict[str, Any]) -> dict[str, Any]:
    requested = str(args.get("backend") or "auto").strip().lower()
    prom_url = str(args.get("prometheus_url") or os.getenv("AI_TRADING_PROMETHEUS_URL", "")).strip()
    grafana_url = str(args.get("grafana_url") or os.getenv("AI_TRADING_GRAFANA_URL", "")).strip()
    grafana_uid = str(
        args.get("grafana_prometheus_uid") or os.getenv("AI_TRADING_GRAFANA_PROMETHEUS_UID", "")
    ).strip()
    grafana_token = str(
        args.get("grafana_api_token") or os.getenv("AI_TRADING_GRAFANA_API_TOKEN", "")
    ).strip()

    if requested == "prometheus":
        if not prom_url:
            raise RuntimeError("backend=prometheus requires prometheus_url or AI_TRADING_PROMETHEUS_URL")
        return {"backend": "prometheus", "prometheus_url": prom_url.rstrip("/")}

    if requested == "grafana":
        if not grafana_url or not grafana_uid:
            raise RuntimeError(
                "backend=grafana requires grafana_url + grafana_prometheus_uid "
                "(or AI_TRADING_GRAFANA_URL + AI_TRADING_GRAFANA_PROMETHEUS_UID)"
            )
        return {
            "backend": "grafana",
            "grafana_url": grafana_url.rstrip("/"),
            "grafana_prometheus_uid": grafana_uid,
            "grafana_api_token": grafana_token,
        }

    if prom_url:
        return {"backend": "prometheus", "prometheus_url": prom_url.rstrip("/")}
    if grafana_url and grafana_uid:
        return {
            "backend": "grafana",
            "grafana_url": grafana_url.rstrip("/"),
            "grafana_prometheus_uid": grafana_uid,
            "grafana_api_token": grafana_token,
        }

    return {"backend": "none"}


def _query_range_url(backend: dict[str, Any], query: str, start: int, end: int, step_s: int) -> tuple[str, dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "start": str(start),
            "end": str(end),
            "step": str(step_s),
        }
    )
    kind = str(backend.get("backend") or "none")
    if kind == "prometheus":
        base = str(backend["prometheus_url"])
        return (f"{base}/api/v1/query_range?{params}", {})
    if kind == "grafana":
        base = str(backend["grafana_url"])
        uid = str(backend["grafana_prometheus_uid"])
        token = str(backend.get("grafana_api_token") or "")
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return (f"{base}/api/datasources/proxy/uid/{uid}/api/v1/query_range?{params}", headers)
    raise RuntimeError("no metrics backend configured")


def _normalize_series(raw: dict[str, Any]) -> list[dict[str, Any]]:
    data = raw.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("metrics response missing data payload")
    status = str(raw.get("status") or "")
    if status != "success":
        err = raw.get("error") or raw.get("errorType") or "query failed"
        raise RuntimeError(str(err))
    result = data.get("result")
    if not isinstance(result, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        labels = metric if isinstance(metric, dict) else {}
        values = item.get("values")
        points: list[list[Any]]
        if isinstance(values, list):
            points = [pt for pt in values if isinstance(pt, list) and len(pt) >= 2]
        else:
            single = item.get("value")
            if isinstance(single, list) and len(single) >= 2:
                points = [single]
            else:
                points = []
        normalized.append({"labels": labels, "values": points})
    return normalized


def _series_summary(points: list[list[Any]]) -> dict[str, Any]:
    samples: list[float] = []
    ts_values: list[int] = []
    for point in points:
        ts = _float_or_none(point[0])
        value = _float_or_none(point[1])
        if ts is None or value is None:
            continue
        ts_values.append(int(ts))
        samples.append(value)
    if not samples:
        return {
            "sample_count": 0,
            "latest": None,
            "min": None,
            "max": None,
            "mean": None,
            "delta": None,
            "trend": "unknown",
        }
    first = samples[0]
    latest = samples[-1]
    delta = latest - first
    trend = "flat"
    if delta > 1e-9:
        trend = "up"
    elif delta < -1e-9:
        trend = "down"
    return {
        "sample_count": len(samples),
        "latest": latest,
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "delta": delta,
        "trend": trend,
        "start_ts": ts_values[0] if ts_values else None,
        "end_ts": ts_values[-1] if ts_values else None,
    }


def _query_range(
    *,
    query: str,
    start_ts: int,
    end_ts: int,
    step_s: int,
    backend_cfg: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    url, headers = _query_range_url(backend_cfg, query, start_ts, end_ts, step_s)
    raw = _http_get_json(url=url, headers=headers, timeout_s=timeout_s)
    series = _normalize_series(raw)
    summaries: list[dict[str, Any]] = []
    for item in series:
        labels = cast(dict[str, Any], item.get("labels") or {})
        values = cast(list[list[Any]], item.get("values") or [])
        summaries.append({"labels": labels, "summary": _series_summary(values)})
    return {
        "backend": backend_cfg.get("backend"),
        "query": query,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "step_s": step_s,
        "series_count": len(summaries),
        "series": summaries,
    }


def _default_query(env_name: str, fallback: str) -> str:
    return str(os.getenv(env_name, fallback)).strip() or fallback


def _parse_iso_ts(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _daily_date_to_ts(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    for suffix in ("T00:00:00+00:00", "T00:00:00Z"):
        ts = _parse_iso_ts(f"{text}{suffix}")
        if ts is not None:
            return ts
    return None


def _result_has_samples(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    series = payload.get("series")
    if not isinstance(series, list):
        return False
    for item in series:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if not isinstance(summary, dict):
            continue
        sample_count = _float_or_none(summary.get("sample_count"))
        if sample_count is not None and sample_count > 0:
            return True
    return False


def _local_series_result(
    *,
    metric_name: str,
    query: str,
    points: list[list[Any]],
    backend: str,
    start_ts: int,
    end_ts: int,
    step_s: int,
) -> dict[str, Any]:
    labels = {"source": "runtime_report_fallback", "metric": metric_name}
    series: list[dict[str, Any]] = []
    if points:
        series.append({"labels": labels, "summary": _series_summary(points)})
    return {
        "backend": backend,
        "query": query,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "step_s": step_s,
        "series_count": len(series),
        "series": series,
        "source": "runtime_report_fallback",
    }


def _runtime_fallback_series(
    args: dict[str, Any],
    *,
    query_map: dict[str, str],
    backend: str,
    start_ts: int,
    end_ts: int,
    step_s: int,
) -> dict[str, dict[str, Any]]:
    report_path = Path(
        str(
            args.get("runtime_report_path")
            or os.getenv("AI_TRADING_RUNTIME_DAILY_REPORT_PATH")
            or "/var/lib/ai-trading-bot/runtime/daily_performance_report.json"
        )
    )
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(report, dict):
        return {}

    execution = report.get("execution_vs_alpha")
    go_no_go = report.get("go_no_go")
    if not isinstance(execution, dict):
        execution = {}
    if not isinstance(go_no_go, dict):
        go_no_go = {}
    observed = go_no_go.get("observed")
    if not isinstance(observed, dict):
        observed = {}

    daily = execution.get("daily")
    if not isinstance(daily, list):
        daily = []

    capture_points: list[list[Any]] = []
    for row in daily:
        if not isinstance(row, dict):
            continue
        day_ts = _daily_date_to_ts(row.get("date"))
        expected = _float_or_none(row.get("expected_edge_per_accept_bps"))
        realized = _float_or_none(row.get("realized_net_edge_bps"))
        if day_ts is None or expected is None or realized is None:
            continue
        if expected == 0.0:
            continue
        capture_points.append([day_ts, realized / expected])

    slippage_value = _float_or_none(execution.get("slippage_drag_bps"))
    if slippage_value is None:
        slippage_value = _float_or_none(observed.get("slippage_drag_bps"))
    slippage_points: list[list[Any]] = []
    if slippage_value is not None:
        if capture_points:
            slippage_points = [[point[0], slippage_value] for point in capture_points]
        else:
            slippage_points = [[end_ts, slippage_value]]

    reject_value = _float_or_none(observed.get("order_reject_rate_pct"))
    reject_points: list[list[Any]] = []
    if reject_value is not None:
        reject_points = [[end_ts, reject_value]]

    fallback: dict[str, dict[str, Any]] = {}
    if slippage_points:
        fallback["slippage_drag_bps"] = _local_series_result(
            metric_name="slippage_drag_bps",
            query=query_map["slippage_drag_bps"],
            points=slippage_points,
            backend=backend,
            start_ts=start_ts,
            end_ts=end_ts,
            step_s=step_s,
        )
    if capture_points:
        fallback["execution_capture_ratio"] = _local_series_result(
            metric_name="execution_capture_ratio",
            query=query_map["execution_capture_ratio"],
            points=capture_points,
            backend=backend,
            start_ts=start_ts,
            end_ts=end_ts,
            step_s=step_s,
        )
    if reject_points:
        fallback["order_reject_rate_pct"] = _local_series_result(
            metric_name="order_reject_rate_pct",
            query=query_map["order_reject_rate_pct"],
            points=reject_points,
            backend=backend,
            start_ts=start_ts,
            end_ts=end_ts,
            step_s=step_s,
        )
    return fallback


def tool_metrics_backend_status(args: dict[str, Any]) -> dict[str, Any]:
    backend = _resolve_backend(args)
    details = dict(backend)
    token = str(details.get("grafana_api_token") or "")
    if token:
        details["grafana_api_token"] = f"{token[:8]}...{token[-4:]}"
    return {
        "backend": backend.get("backend"),
        "configured": backend.get("backend") in {"prometheus", "grafana"},
        "details": details,
    }


def tool_query_promql_range(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise RuntimeError("query_promql_range requires args.query")
    backend = _resolve_backend(args)
    if backend.get("backend") == "none":
        raise RuntimeError("no metrics backend configured")

    duration_m = int(args.get("duration_minutes") or 60)
    step_s = int(args.get("step_s") or 60)
    end_ts = int(args.get("end_ts") or _now_unix())
    start_ts = int(args.get("start_ts") or (end_ts - max(1, duration_m) * 60))
    timeout_s = float(args.get("timeout_s") or 8.0)
    return _query_range(
        query=query,
        start_ts=start_ts,
        end_ts=end_ts,
        step_s=step_s,
        backend_cfg=backend,
        timeout_s=timeout_s,
    )


def tool_execution_trends_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    backend = _resolve_backend(args)
    if backend.get("backend") == "none":
        raise RuntimeError("no metrics backend configured")

    duration_m = int(args.get("duration_minutes") or 120)
    step_s = int(args.get("step_s") or 60)
    end_ts = int(args.get("end_ts") or _now_unix())
    start_ts = int(args.get("start_ts") or (end_ts - max(1, duration_m) * 60))
    timeout_s = float(args.get("timeout_s") or 8.0)

    query_map = {
        "slippage_drag_bps": str(
            args.get("slippage_query")
            or _default_query("AI_TRADING_PROMQL_SLIPPAGE_DRAG_BPS", "ai_trading_slippage_drag_bps")
        ),
        "execution_capture_ratio": str(
            args.get("capture_query")
            or _default_query(
                "AI_TRADING_PROMQL_EXECUTION_CAPTURE_RATIO",
                "ai_trading_execution_capture_ratio",
            )
        ),
        "order_reject_rate_pct": str(
            args.get("reject_query")
            or _default_query(
                "AI_TRADING_PROMQL_ORDER_REJECT_RATE_PCT",
                "ai_trading_order_reject_rate_pct",
            )
        ),
    }

    metrics: dict[str, Any] = {}
    for metric_name, query in query_map.items():
        try:
            result = _query_range(
                query=query,
                start_ts=start_ts,
                end_ts=end_ts,
                step_s=step_s,
                backend_cfg=backend,
                timeout_s=timeout_s,
            )
            metrics[metric_name] = result
        except Exception as exc:  # pragma: no cover - defensive runtime capture
            metrics[metric_name] = {"error": str(exc), "query": query}

    fallback = _runtime_fallback_series(
        args,
        query_map=query_map,
        backend=str(backend.get("backend") or "none"),
        start_ts=start_ts,
        end_ts=end_ts,
        step_s=step_s,
    )
    fallback_used = False
    for metric_name, fallback_payload in fallback.items():
        current = metrics.get(metric_name)
        if _result_has_samples(current):
            continue
        if isinstance(current, dict) and "error" in current:
            fallback_payload["fallback_reason"] = str(current.get("error") or "backend_query_error")
        else:
            fallback_payload["fallback_reason"] = "backend_no_series"
        metrics[metric_name] = fallback_payload
        fallback_used = True

    return {
        "backend": backend.get("backend"),
        "window": {
            "duration_minutes": duration_m,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "step_s": step_s,
        },
        "fallback_used": fallback_used,
        "metrics": metrics,
    }


TOOLS = {
    "metrics_backend_status": tool_metrics_backend_status,
    "query_promql_range": tool_query_promql_range,
    "execution_trends_snapshot": tool_execution_trends_snapshot,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "metrics_backend_status",
        "description": "Show whether Prometheus/Grafana query backend is configured.",
    },
    {
        "name": "query_promql_range",
        "description": "Execute a PromQL query_range via Prometheus or Grafana datasource proxy.",
    },
    {
        "name": "execution_trends_snapshot",
        "description": "Fetch slippage/capture/reject trend queries for quick execution diagnostics.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Metrics query MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_metrics_query",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
