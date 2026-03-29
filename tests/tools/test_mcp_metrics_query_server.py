from __future__ import annotations

from tools import mcp_metrics_query_server as metrics_srv


def test_series_summary_reports_up_trend() -> None:
    points = [[1, "1.0"], [2, "2.0"], [3, "3.5"]]
    summary = metrics_srv._series_summary(points)
    assert summary["sample_count"] == 3
    assert summary["latest"] == 3.5
    assert summary["trend"] == "up"
    assert summary["delta"] == 2.5


def test_query_promql_range_parses_matrix(monkeypatch) -> None:
    def _fake_http(url: str, headers=None, timeout_s: float = 8.0):
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"job": "ai-trading"},
                        "values": [[1000, "0.10"], [1060, "0.15"], [1120, "0.20"]],
                    }
                ],
            },
        }

    monkeypatch.setattr(metrics_srv, "_http_get_json", _fake_http)

    payload = metrics_srv.tool_query_promql_range(
        {
            "backend": "prometheus",
            "prometheus_url": "http://127.0.0.1:9090",
            "query": "ai_trading_execution_capture_ratio",
            "start_ts": 1000,
            "end_ts": 1120,
            "step_s": 60,
        }
    )
    assert payload["backend"] == "prometheus"
    assert payload["series_count"] == 1
    first = payload["series"][0]
    assert first["labels"]["job"] == "ai-trading"
    assert first["summary"]["latest"] == 0.2


def test_metrics_backend_status_unconfigured() -> None:
    payload = metrics_srv.tool_metrics_backend_status({})
    assert payload["configured"] is False
    assert payload["backend"] == "none"


def test_execution_trends_snapshot_falls_back_to_runtime_report(monkeypatch) -> None:
    def _fake_query_range(*, query, start_ts, end_ts, step_s, backend_cfg, timeout_s):
        return {
            "backend": backend_cfg.get("backend"),
            "query": query,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "step_s": step_s,
            "series_count": 0,
            "series": [],
        }

    def _fake_runtime_fallback(
        args,
        *,
        query_map,
        backend,
        start_ts,
        end_ts,
        step_s,
    ):
        return {
            "execution_capture_ratio": {
                "backend": backend,
                "query": query_map["execution_capture_ratio"],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "step_s": step_s,
                "series_count": 1,
                "series": [
                    {
                        "labels": {"source": "runtime_report_fallback"},
                        "summary": {
                            "sample_count": 2,
                            "latest": 0.21,
                            "min": 0.11,
                            "max": 0.21,
                            "mean": 0.16,
                            "delta": 0.1,
                            "trend": "up",
                        },
                    }
                ],
                "source": "runtime_report_fallback",
            }
        }

    monkeypatch.setattr(metrics_srv, "_query_range", _fake_query_range)
    monkeypatch.setattr(metrics_srv, "_runtime_fallback_series", _fake_runtime_fallback)

    payload = metrics_srv.tool_execution_trends_snapshot(
        {
            "backend": "prometheus",
            "prometheus_url": "http://127.0.0.1:9090",
            "duration_minutes": 60,
            "step_s": 60,
        }
    )

    assert payload["fallback_used"] is True
    capture = payload["metrics"]["execution_capture_ratio"]
    assert capture["source"] == "runtime_report_fallback"
    assert capture["series_count"] == 1
    assert capture["series"][0]["summary"]["latest"] == 0.21
