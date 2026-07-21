from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

import pandas as pd
import pytest

from ai_trading.analytics.opportunity_markouts import resolve_opportunity_markouts
from ai_trading.tools.opportunity_markout_report import main


def _decision(
    *,
    correlation_id: str,
    symbol: str,
    source_timestamp: datetime,
    decision_timestamp: datetime | None = None,
    side: str = "buy",
    submitted: bool = False,
    opportunity_eligible: bool = True,
) -> dict[str, object]:
    source_iso = source_timestamp.isoformat()
    decision_iso = (decision_timestamp or source_timestamp).isoformat()
    return {
        "schema_version": "2.0.0",
        "correlation_id": correlation_id,
        "symbol": symbol,
        "bar_ts": source_iso,
        "order": None,
        "metrics": {
            "correlation_id": correlation_id,
            "source_timestamp": source_iso,
            "decision_ts": decision_iso,
            "opportunity_eligible": opportunity_eligible,
            "reference_price": 100.0,
            "spread_bps": 2.0,
            "quote_age_ms": 250.0,
            "order_type": "limit",
            "session_regime": "midday",
            "market_regime": "sideways",
            "execution_profile": "passive",
        },
        "decision_journal": {
            "correlation_id": correlation_id,
            "symbol": symbol,
            "bar_ts": source_iso,
            "source_timestamp": source_iso,
            "decision_ts": decision_iso,
            "event": "metrics_improvement_controlled_skip",
            "submitted": submitted,
            "signal": {"symbol": symbol, "side": side},
            "risk_decision": {
                "gates": ["METRICS_IMPROVEMENT_CONTROLLED_SKIP"]
            },
            "metadata": {
                "opportunity_eligible": opportunity_eligible,
                "reference_price": 100.0,
                "spread_bps": 2.0,
                "quote_age_ms": 250.0,
                "order_type": "limit",
                "session_regime": "midday",
                "market_regime": "sideways",
                "execution_profile": "passive",
            },
        },
    }


def _bars(
    timestamps: list[datetime] | pd.DatetimeIndex,
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": closes},
        index=pd.DatetimeIndex(timestamps),
    )


def test_shadow_markouts_cover_every_horizon_once_and_stay_non_promotional() -> None:
    source_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    decision = _decision(
        correlation_id="opp_aapl_1430",
        symbol="AAPL",
        source_timestamp=source_ts,
    )
    decisions = [
        decision,
        decision,
        _decision(
            correlation_id="opp_googl_1430",
            symbol="GOOGL",
            source_timestamp=source_ts,
        ),
        _decision(
            correlation_id="opp_ineligible",
            symbol="MSFT",
            source_timestamp=source_ts,
            opportunity_eligible=False,
        ),
    ]
    bars = _bars(
        pd.date_range(source_ts, periods=7, freq="min"),
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
    )

    report = resolve_opportunity_markouts(
        decisions,
        {"aapl": bars, "GOOGL": bars},
        fee_bps=0.5,
        slippage_bps=1.0,
    )

    assert report["eligible_opportunities"] == 1
    assert report["duplicate_decision_rows_discarded"] == 1
    assert report["outcomes_emitted"] == 3
    assert report["outcome_ids_unique"] is True
    assert report["label_status_counts"] == {"resolved": 3}
    outcomes = report["outcomes"]
    assert [row["horizon_bars"] for row in outcomes] == [1, 3, 5]
    assert len({row["outcome_id"] for row in outcomes}) == 3
    assert outcomes[0]["gross_markout_bps"] == pytest.approx(100.0)
    assert outcomes[0]["round_trip_cost_bps"] == pytest.approx(5.0)
    assert outcomes[0]["net_markout_bps"] == pytest.approx(95.0)
    assert outcomes[0]["controlled_skip"] is True
    assert outcomes[0]["submitted"] is False
    assert all(row["evidence_partition"] == "shadow" for row in outcomes)
    assert all(row["fill_based_evidence"] is False for row in outcomes)
    assert all(row["promotion_eligible"] is False for row in outcomes)
    assert all(row["runtime_authority"] is False for row in outcomes)
    assert all(row["live_money_authority"] is False for row in outcomes)

    rerun = resolve_opportunity_markouts(decisions, {"AAPL": bars}, fee_bps=0.5, slippage_bps=1.0)
    assert rerun["outcomes"] == outcomes


def test_shadow_markouts_distinguish_unavailable_and_censored_labels() -> None:
    aapl_ts = datetime(2026, 7, 21, 14, 29, tzinfo=UTC)
    amzn_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    msft_ts = datetime(2026, 7, 21, 19, 58, tzinfo=UTC)
    decisions = [
        _decision(
            correlation_id="opp_aapl_missing",
            symbol="AAPL",
            source_timestamp=aapl_ts,
        ),
        _decision(
            correlation_id="opp_amzn_gap",
            symbol="AMZN",
            source_timestamp=amzn_ts,
        ),
        _decision(
            correlation_id="opp_msft_close",
            symbol="MSFT",
            source_timestamp=msft_ts,
        ),
    ]
    frames = {
        "AAPL": _bars(
            pd.date_range(aapl_ts.replace(minute=30), periods=7, freq="min"),
            [100.0] * 7,
        ),
        "AMZN": _bars(
            [
                amzn_ts,
                amzn_ts.replace(minute=31),
                amzn_ts.replace(minute=33),
                amzn_ts.replace(minute=34),
                amzn_ts.replace(minute=35),
            ],
            [100.0, 101.0, 103.0, 104.0, 105.0],
        ),
        "MSFT": _bars(
            pd.date_range(msft_ts, periods=6, freq="min"),
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        ),
    }

    report = resolve_opportunity_markouts(decisions, frames)

    assert report["outcomes_emitted"] == 9
    assert report["label_status_counts"] == {
        "censored": 4,
        "resolved": 2,
        "unavailable": 3,
    }
    assert report["label_reason_counts"] == {
        "non_contiguous_future_bars": 2,
        "ok": 2,
        "session_boundary": 2,
        "source_bar_missing": 3,
    }


def test_shadow_markouts_reject_source_data_after_decision() -> None:
    source_ts = datetime(2026, 7, 21, 14, 31, tzinfo=UTC)
    decision_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    report = resolve_opportunity_markouts(
        [
            _decision(
                correlation_id="opp_future_source",
                symbol="AAPL",
                source_timestamp=source_ts,
                decision_timestamp=decision_ts,
            )
        ],
        {
            "AAPL": _bars(
                pd.date_range(source_ts, periods=7, freq="min"),
                [100.0] * 7,
            )
        },
    )

    assert report["label_status_counts"] == {"unavailable": 3}
    assert report["label_reason_counts"] == {"source_timestamp_after_decision": 3}


def test_opportunity_markout_cli_writes_dated_and_latest_artifacts(tmp_path) -> None:
    source_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        json.dumps(
            _decision(
                correlation_id="opp_aapl_cli",
                symbol="AAPL",
                source_timestamp=source_ts,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range(source_ts, periods=7, freq="min"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        }
    ).to_csv(bars_dir / "AAPL.csv", index=False)
    output = tmp_path / "opportunity_markouts_20260721.json"
    latest = tmp_path / "opportunity_markouts_latest.json"

    rc = main(
        [
            "--report-date",
            "2026-07-21",
            "--decisions-jsonl",
            str(decisions_path),
            "--bars-dir",
            str(bars_dir),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert output.read_bytes() == latest.read_bytes()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["report_date"] == "2026-07-21"
    assert payload["decision_rows_scanned"] == 1
    assert payload["outcomes_emitted"] == 3
    assert payload["promotion_eligible"] is False


def test_opportunity_markout_cli_verifies_historical_manifest_hash(tmp_path) -> None:
    source_ts = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        json.dumps(
            _decision(
                correlation_id="opp_manifest",
                symbol="AAPL",
                source_timestamp=source_ts,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "verified-bars"
    dataset_dir.mkdir()
    csv_path = dataset_dir / "AAPL.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range(source_ts, periods=7, freq="min"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        }
    ).to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "historical.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "cache_key": "cache-1",
                "quality_passed": True,
                "authority": {
                    "research_only": True,
                    "evidence_type": "historical_research",
                    "promotion_eligible": False,
                    "promotion_authority": False,
                    "live_money_authority": False,
                    "runtime_fill_authority": False,
                },
                "symbols": [
                    {
                        "symbol": "AAPL",
                        "csv_path": str(csv_path),
                        "content_sha256": digest,
                        "quality_passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "markouts.json"

    assert main(
        [
            "--report-date",
            "2026-07-21",
            "--decisions-jsonl",
            str(decisions_path),
            "--historical-backfill-json",
            str(manifest_path),
            "--symbols",
            "AAPL",
            "--output-json",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["outcomes_emitted"] == 3
    assert payload["bars_provenance"]["quality_passed"] is True
    assert payload["bars_provenance"]["promotion_eligible"] is False

    csv_path.write_text("timestamp,close\ncorrupt,0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--report-date",
                "2026-07-21",
                "--decisions-jsonl",
                str(decisions_path),
                "--historical-backfill-json",
                str(manifest_path),
                "--symbols",
                "AAPL",
            ]
        )
