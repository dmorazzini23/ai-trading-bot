from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import (
    huggingface_cache_materializer,
    huggingface_candidate_intake,
    huggingface_research_discovery,
    huggingface_sentiment_benchmark,
    huggingface_sentiment_predictions,
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
    assert report["non_authoritative"] is True
    assert report["external_network_allowed"] is False
    assert report["summary"]["candidate_count"] == 3
    assert report["candidates"][0]["repo_id"] == "open/finance-timeseries"
    assert report["candidates"][0]["non_authoritative"] is True
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


def test_huggingface_api_requires_enabled_explicit_opt_in() -> None:
    calls: list[str] = []

    report = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        enabled=False,
        use_hf_api=True,
        fetch_json=lambda url, *_args: calls.append(url) or [],
    )

    assert report["status"] == "disabled"
    assert report["blocked_reasons"] == ["hf_research_disabled"]
    assert report["query"]["explicit_opt_in"] is False
    assert report["external_network_allowed"] is False
    assert calls == []


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


def _sentiment_intake_fixture() -> dict[str, object]:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results={
            "candidates": [
                {
                    "id": "open/finance-sentiment-model",
                    "repo_type": "model",
                    "tags": ["finance", "sentiment", "license:apache-2.0"],
                    "pipeline_tag": "text-classification",
                    "downloads": 100,
                    "cardData": {"license": "apache-2.0"},
                }
            ]
        },
        max_results=10,
    )
    return huggingface_candidate_intake.build_huggingface_candidate_intake(
        report_date="2026-05-09",
        discovery=discovery,
    )


def test_huggingface_sentiment_benchmark_scores_current_path_with_fake_analyzer() -> None:
    intake = _sentiment_intake_fixture()

    def fake_analyzer(text: str) -> dict[str, object]:
        positive_terms = (
            "stronger",
            "buyback",
            "raised",
            "upgraded",
            "won",
            "lift revenue",
            "resilient",
            "upside",
        )
        negative_terms = (
            "fell",
            "investigation",
            "missed",
            "contracted",
            "downgraded",
            "outage",
            "antitrust",
            "fines",
        )
        if any(term in text for term in positive_terms):
            return {"available": True, "score": 0.7, "confidence": 0.8, "latency_ms": 3.0}
        if any(term in text for term in negative_terms):
            return {"available": True, "score": -0.8, "confidence": 0.85, "latency_ms": 4.0}
        return {"available": True, "score": 0.0, "confidence": 0.6, "latency_ms": 2.0}

    report = huggingface_sentiment_benchmark.build_huggingface_sentiment_benchmark(
        report_date="2026-05-09",
        intake=intake,
        current_analyzer=fake_analyzer,
    )

    assert report["status"] == "evaluated"
    assert report["runtime_authority"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["current_sentiment_path"]["coverage"] == 1.0
    assert report["current_sentiment_path"]["accuracy_on_covered"] == 1.0
    assert report["summary"]["candidates_needing_predictions"] == 1
    assert report["candidate_results"][0]["recommendation"] == "materialize_or_generate_offline_predictions"


def test_huggingface_sentiment_benchmark_compares_candidate_prediction_fixture() -> None:
    intake = _sentiment_intake_fixture()
    samples = {
        "samples": [
            {"id": "s1", "text": "good earnings", "expected_label": "positive"},
            {"id": "s2", "text": "bad downgrade", "expected_label": "negative"},
            {"id": "s3", "text": "meeting notice", "expected_label": "neutral"},
        ]
    }
    predictions = {
        "predictions": [
            {
                "repo_id": "open/finance-sentiment-model",
                "samples": [
                    {"sample_id": "s1", "score": 0.9, "confidence": 0.9, "latency_ms": 12},
                    {"sample_id": "s2", "score": -0.7, "confidence": 0.7, "latency_ms": 13},
                    {"sample_id": "s3", "score": 0.0, "confidence": 0.8, "latency_ms": 11},
                ],
            }
        ]
    }

    report = huggingface_sentiment_benchmark.build_huggingface_sentiment_benchmark(
        report_date="2026-05-09",
        intake=intake,
        samples_payload=samples,
        candidate_predictions_payload=predictions,
        current_analyzer=lambda _text: {
            "available": True,
            "score": 0.0,
            "confidence": 0.5,
            "latency_ms": 1,
        },
    )

    candidate = report["candidate_results"][0]
    assert candidate["coverage"] == 1.0
    assert candidate["accuracy_on_covered"] == 1.0
    assert candidate["recommendation"] == "consider_deeper_weekend_testing"
    assert report["summary"]["deeper_weekend_testing_candidates"] == ["open/finance-sentiment-model"]


def test_huggingface_sentiment_benchmark_distinguishes_dataset_review_from_missing_model_predictions() -> None:
    discovery = huggingface_research_discovery.build_huggingface_research_discovery(
        report_date="2026-05-09",
        offline_results={
            "candidates": [
                {
                    "id": "open/finance-sentiment-model",
                    "repo_type": "model",
                    "tags": ["finance", "sentiment", "license:apache-2.0"],
                    "pipeline_tag": "text-classification",
                    "downloads": 100,
                    "cardData": {"license": "apache-2.0"},
                },
                {
                    "id": "open/finance-sentiment-dataset",
                    "repo_type": "dataset",
                    "tags": ["finance", "sentiment", "license:apache-2.0"],
                    "downloads": 100,
                    "cardData": {"license": "apache-2.0"},
                },
            ]
        },
        max_results=10,
    )
    intake = huggingface_candidate_intake.build_huggingface_candidate_intake(
        report_date="2026-05-09",
        discovery=discovery,
    )
    samples = {"samples": [{"id": "s1", "text": "good earnings", "expected_label": "positive"}]}
    predictions = {
        "predictions": [
            {
                "repo_id": "open/finance-sentiment-model",
                "samples": [{"sample_id": "s1", "score": 0.9, "confidence": 0.9}],
            }
        ]
    }

    report = huggingface_sentiment_benchmark.build_huggingface_sentiment_benchmark(
        report_date="2026-05-09",
        intake=intake,
        samples_payload=samples,
        candidate_predictions_payload=predictions,
        current_analyzer=lambda _text: {
            "available": True,
            "score": 0.9,
            "confidence": 0.9,
        },
    )

    assert report["summary"]["candidates_needing_predictions"] == 0
    assert report["summary"]["dataset_or_inspect_candidates_remaining"] == 1
    assert report["operator_action"] == "review_remaining_dataset_or_inspect_candidates"


def test_huggingface_sentiment_benchmark_cli_writes_artifacts(tmp_path: Path) -> None:
    intake_path = tmp_path / "intake.json"
    output = tmp_path / "benchmark.json"
    latest = tmp_path / "benchmark_latest.json"
    intake_path.write_text(json.dumps(_sentiment_intake_fixture()), encoding="utf-8")

    rc = huggingface_sentiment_benchmark.main(
        [
            "--report-date",
            "2026-05-09",
            "--intake-json",
            str(intake_path),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "huggingface_sentiment_benchmark"
    assert payload["research_only"] is True
    assert payload["runtime_authority"] is False
    assert payload["paths"]["latest"] == str(latest)
    assert latest.is_file()


def test_huggingface_sentiment_predictions_require_explicit_download_opt_in(tmp_path: Path) -> None:
    calls: list[str] = []

    report = huggingface_sentiment_predictions.build_huggingface_sentiment_predictions(
        report_date="2026-05-09",
        intake=_sentiment_intake_fixture(),
        cache_dir=tmp_path / "cache",
        allow_downloads=False,
        pipeline_factory=lambda repo_id, _cache_dir, _allow_downloads: calls.append(repo_id) or (lambda _text: []),
    )

    assert report["status"] == "blocked"
    assert report["runtime_authority"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["summary"]["blocked_candidates"] == 1
    assert report["predictions"][0]["blocked_reasons"] == ["hf_downloads_disabled"]
    assert calls == []


def test_huggingface_sentiment_predictions_generate_fixture_predictions(tmp_path: Path) -> None:
    def fake_pipeline_factory(_repo_id: str, _cache_dir: Path, allow_downloads: bool):
        assert allow_downloads is True

        def classify(text: str):
            if any(
                term in text
                for term in (
                    "stronger",
                    "buyback",
                    "raised",
                    "upgraded",
                    "won",
                    "lift revenue",
                    "resilient",
                    "upside",
                )
            ):
                return [
                    {"label": "positive", "score": 0.91},
                    {"label": "negative", "score": 0.03},
                    {"label": "neutral", "score": 0.06},
                ]
            if any(
                term in text
                for term in (
                    "fell",
                    "investigation",
                    "missed",
                    "contracted",
                    "downgraded",
                    "outage",
                    "antitrust",
                    "fines",
                )
            ):
                return [
                    {"label": "positive", "score": 0.04},
                    {"label": "negative", "score": 0.9},
                    {"label": "neutral", "score": 0.06},
                ]
            return [
                {"label": "positive", "score": 0.05},
                {"label": "negative", "score": 0.04},
                {"label": "neutral", "score": 0.91},
            ]

        return classify

    report = huggingface_sentiment_predictions.build_huggingface_sentiment_predictions(
        report_date="2026-05-09",
        intake=_sentiment_intake_fixture(),
        cache_dir=tmp_path / "cache",
        allow_downloads=True,
        pipeline_factory=fake_pipeline_factory,
    )

    assert report["status"] == "evaluated"
    assert report["summary"]["evaluated_candidates"] == 1
    assert report["predictions"][0]["samples"][0]["label"] == "positive"
    assert report["predictions"][0]["runtime_authority"] is False


def test_huggingface_sentiment_predictions_cli_writes_artifacts(tmp_path: Path) -> None:
    intake_path = tmp_path / "intake.json"
    output = tmp_path / "predictions.json"
    latest = tmp_path / "predictions_latest.json"
    intake_path.write_text(json.dumps(_sentiment_intake_fixture()), encoding="utf-8")

    rc = huggingface_sentiment_predictions.main(
        [
            "--report-date",
            "2026-05-09",
            "--intake-json",
            str(intake_path),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "huggingface_sentiment_predictions"
    assert payload["research_only"] is True
    assert payload["runtime_authority"] is False
    assert payload["paths"]["latest"] == str(latest)
    assert latest.is_file()
