from __future__ import annotations

from tools import mcp_market_events_server as market_srv


def test_fetch_events_normalizes_payload(monkeypatch) -> None:
    def _fake_http(url: str, timeout_s: float = 8.0):
        _ = url, timeout_s
        return {
            "events": [
                {
                    "name": "FOMC Rate Decision",
                    "datetime": "2026-03-30T18:00:00Z",
                    "impact": "high",
                    "symbols": ["SPY", "QQQ"],
                }
            ]
        }

    monkeypatch.setattr(market_srv, "_http_get_json", _fake_http)
    payload = market_srv.tool_fetch_events({"url": "https://example.test/events.json"})
    assert payload["configured"] is True
    assert payload["count"] == 1
    event = payload["events"][0]
    assert event["title"] == "FOMC Rate Decision"
    assert event["impact_level"] == "high"
    assert event["symbols"] == ["SPY", "QQQ"]


def test_market_risk_window_flags_high_impact(monkeypatch) -> None:
    monkeypatch.setattr(
        market_srv,
        "tool_fetch_events",
        lambda args: {
            "configured": True,
            "events": [
                {
                    "title": "CPI Release",
                    "timestamp": 1_800_000_000,
                    "impact_level": "high",
                    "impact": "high",
                }
            ],
        },
    )
    monkeypatch.setattr(
        market_srv,
        "tool_market_sessions",
        lambda args: {"sessions": [{"session_date": "2026-03-30"}]},
    )
    monkeypatch.setattr(market_srv, "_now_ts", lambda args: 1_799_999_000)
    payload = market_srv.tool_market_risk_window({"horizon_hours": 2})
    assert payload["risk_level"] == "high"
    assert payload["high_impact_count"] == 1
