from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import (
    huggingface_cache_materializer,
    huggingface_candidate_intake,
    huggingface_research_discovery,
)


def _fixture_results() -> dict[str, object]:
    return {
        "candidates": [
            {
                "id": "open/finance-timeseries",
                "repo_type": "model",
                "tags": ["finance", "time-series", "license:apache-2.0"],
                "pipeline_tag": "tabular-classification",
                "downloads": 2500,
                "likes": 12,
                "cardData": {"license": "apache-2.0"},
            },
            {
                "id": "closed/gated-stock-model",
                "repo_type": "model",
                "tags": ["stock", "forecasting"],
                "downloads": 4000,
                "gated": True,
                "cardData": {"license": "mit"},
            },
            {
                "id": "unknown/no-license",
                "repo_type": "dataset",
                "tags": ["finance", "dataset"],
                "downloads": 20,
                "cardData": {},
            },
        ]
    }


def test_huggingface_discovery_is_research_only_and_scores_offline_results() -> None:
    report = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results=_fixture_results(),
        enabled=False,
        max_results=10,
    )

    assert report["status"] == "discovered"
    assert report["runtime_authority"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["summary"]["candidate_count"] == 3
    assert report["candidates"][0]["repo_id"] == "open/finance-timeseries"
    gated = next(row for row in report["candidates"] if row["repo_id"] == "closed/gated-stock-model")
    assert gated["blocked_reasons"] == ["gated_access_not_allowed"]


def test_huggingface_discovery_disabled_without_offline_fixture_blocks_softly() -> None:
    report = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        enabled=False,
        use_hf_api=False,
    )

    assert report["status"] == "disabled"
    assert report["blocked_reasons"] == ["hf_research_disabled"]
    assert report["research_only"] is True


def test_huggingface_discovery_cli_writes_dated_and_latest(tmp_path: Path) -> None:
    fixture = tmp_path / "hf_results.json"
    output = tmp_path / "hf_discovery.json"
    latest = tmp_path / "hf_discovery_latest.json"
    fixture.write_text(json.dumps(_fixture_results()), encoding="utf-8")

    rc = huggingface_research_discovery.main(
        [
            "--report-date",
            "2026-05-09",
            "--offline-results-json",
            str(fixture),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "huggingface_research_discovery"
    assert payload["paths"]["latest"] == str(latest)
    assert latest.is_file()


def test_huggingface_candidate_intake_blocks_license_and_keeps_runtime_disallowed() -> None:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results=_fixture_results(),
        max_results=10,
    )

    intake = huggingface_candidate_intake.build_huggingface_candidate_intake(
        report_date="2026-05-09",
        discovery=discovery,
        require_card=True,
    )

    assert intake["runtime_authority"] is False
    assert intake["promotion_authority"] is False
    assert intake["summary"]["ready"] == 1
    assert intake["summary"]["blocked"] == 2
    blocked = {row["repo_id"]: row["blocked_reasons"] for row in intake["candidates"] if row["status"] == "blocked"}
    assert "gated_access_not_allowed" in blocked["closed/gated-stock-model"]
    assert "missing_license" in blocked["unknown/no-license"]
    ready = next(row for row in intake["candidates"] if row["status"] == "ready_for_manual_review")
    assert ready["intake"]["runtime_use_allowed"] is False
    assert ready["intake"]["materialization_allowed"] is True
    assert intake["experiment_ledger_entry"]["workflow"] == "huggingface_candidate_intake"


def test_huggingface_candidate_intake_cli_selects_requested_candidate(tmp_path: Path) -> None:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results=_fixture_results(),
        max_results=10,
    )
    discovery_path = tmp_path / "discovery.json"
    output = tmp_path / "intake.json"
    latest = tmp_path / "intake_latest.json"
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

    rc = huggingface_candidate_intake.main(
        [
            "--report-date",
            "2026-05-09",
            "--discovery-json",
            str(discovery_path),
            "--candidate-id",
            "open/finance-timeseries",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["selected"] == 1
    assert payload["candidates"][0]["repo_id"] == "open/finance-timeseries"
    assert latest.is_file()


def test_huggingface_cache_materializer_dry_run_does_not_create_cache(tmp_path: Path) -> None:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results=_fixture_results(),
        max_results=10,
    )
    intake = huggingface_candidate_intake.build_huggingface_candidate_intake(
        report_date="2026-05-09",
        discovery=discovery,
    )
    cache_dir = tmp_path / "cache"

    report = huggingface_cache_materializer.build_huggingface_cache_materialization(
        report_date="2026-05-09",
        intake=intake,
        cache_dir=cache_dir,
        dry_run=True,
    )

    assert report["status"] == "planned"
    assert report["summary"]["dry_run"] is True
    assert report["artifacts"][0]["runtime_use_allowed"] is False
    assert not cache_dir.exists()


def test_huggingface_cache_materializer_local_source_records_manifest(tmp_path: Path) -> None:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results=_fixture_results(),
        max_results=10,
    )
    intake = huggingface_candidate_intake.build_huggingface_candidate_intake(
        report_date="2026-05-09",
        discovery=discovery,
    )
    source = tmp_path / "source" / "open_finance-timeseries"
    source.mkdir(parents=True)
    (source / "README.md").write_text("model card", encoding="utf-8")

    report = huggingface_cache_materializer.build_huggingface_cache_materialization(
        report_date="2026-05-09",
        intake=intake,
        cache_dir=tmp_path / "cache",
        allow_downloads=True,
        local_source_dir=tmp_path / "source",
    )

    assert report["status"] == "materialized"
    assert report["summary"]["materialized"] == 1
    assert report["artifacts"][0]["manifest"]["file_count"] == 1
    assert report["live_money_authority"] is False
