"""Materialize approved Hugging Face research candidates into an offline cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_ARTIFACT_TYPE = "huggingface_cache_materialization"
_SCHEMA_VERSION = "1.0.0"
_DEFAULT_OUTPUT_DIR = "runtime/research_reports/huggingface"
_DEFAULT_CACHE_DIR = "runtime/research_reports/huggingface/cache"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _compact_date(value: str) -> str:
    return str(value or "").replace("-", "") or _utc_now().strftime("%Y%m%d")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _truthy(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        _DEFAULT_OUTPUT_DIR,
        default_relative=_DEFAULT_OUTPUT_DIR,
        for_write=True,
    )
    return (
        root / f"hf_cache_materialization_{_compact_date(report_date)}.json",
        root.parent / "latest" / "hf_cache_materialization_latest.json",
    )


def _default_intake_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/research_reports/latest/hf_candidate_intake_latest.json",
        default_relative="runtime/research_reports/latest/hf_candidate_intake_latest.json",
    )


def _default_cache_dir(raw: str | Path | None = None) -> Path:
    value = str(
        raw
        or get_env("AI_TRADING_HF_CACHE_DIR", _DEFAULT_CACHE_DIR, cast=str, resolve_aliases=False)
        or _DEFAULT_CACHE_DIR
    )
    return resolve_runtime_artifact_path(value, default_relative=_DEFAULT_CACHE_DIR, for_write=True)


def _approved_candidates(intake: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = intake.get("candidates")
    if not isinstance(raw, list):
        return []
    approved: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        details = item.get("intake")
        if (
            item.get("status") == "ready_for_manual_review"
            and isinstance(details, Mapping)
            and bool(details.get("materialization_allowed"))
            and not bool(details.get("runtime_use_allowed"))
        ):
            approved.append(dict(item))
    return approved


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120] or "hf_candidate"


def _pinned_revision(candidate: Mapping[str, Any], fallback: str = "") -> str:
    revision = str(candidate.get("sha") or candidate.get("revision") or fallback or "").strip()
    return "" if revision.lower() in {"", "main", "master", "latest"} else revision


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _manifest_for_path(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if path.exists():
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            files.append(
                {
                    "path": str(child.relative_to(path)),
                    "size_bytes": size,
                    "sha256": _sha256_file(child),
                }
            )
    return {
        "file_count": len(files),
        "size_bytes": int(sum(int(item.get("size_bytes") or 0) for item in files)),
        "files": files[:100],
        "truncated": len(files) > 100,
    }


def _copy_local_source(source: Path, target: Path) -> dict[str, Any]:
    if not source.exists():
        return {"status": "blocked", "blocked_reasons": ["local_source_missing"]}
    target.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, target / source.name)
    else:
        for child in source.rglob("*"):
            if not child.is_file():
                continue
            destination = target / child.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
    manifest = _manifest_for_path(target)
    return {"status": "materialized", "manifest": manifest}


def _materialize_from_hf(
    *,
    repo_id: str,
    repo_type: str,
    target: Path,
    revision: str,
    allow_patterns: Sequence[str],
    ignore_patterns: Sequence[str],
) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        return {"status": "blocked", "blocked_reasons": ["huggingface_hub_not_installed"]}
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset" if repo_type == "dataset" else "model",
            revision=revision or None,
            local_dir=str(target),
            allow_patterns=list(allow_patterns) or None,
            ignore_patterns=list(ignore_patterns) or None,
        )
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - external client/network behavior
        return {"status": "blocked", "blocked_reasons": [f"huggingface_download_failed:{type(exc).__name__}"]}
    manifest = _manifest_for_path(Path(local_path))
    return {"status": "materialized", "manifest": manifest}


def build_huggingface_cache_materialization(
    *,
    report_date: str,
    intake: Mapping[str, Any],
    cache_dir: Path,
    allow_downloads: bool = False,
    use_hf_api: bool = False,
    dry_run: bool = False,
    local_source_dir: Path | None = None,
    max_candidates: int = 5,
    revision: str = "main",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at.astimezone(UTC) if generated_at else _utc_now()
    candidates = _approved_candidates(intake)[: max(0, int(max_candidates))]
    artifacts: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    if not allow_downloads and not dry_run:
        blocked_reasons.append("hf_downloads_disabled")
    for candidate in candidates:
        repo_id = str(candidate.get("repo_id") or candidate.get("hf_id") or "")
        repo_type = str(candidate.get("resource_type") or "model")
        candidate_revision = _pinned_revision(candidate, revision)
        target = cache_dir / _safe_name(repo_id)
        row: dict[str, Any] = {
            "repo_id": repo_id,
            "hf_id": repo_id,
            "resource_type": repo_type,
            "revision": candidate_revision or None,
            "local_path": str(target),
            "runtime_use_allowed": False,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "provider_authority": False,
        }
        if dry_run:
            status = "planned" if candidate_revision else "blocked"
            row.update({"status": status, "manifest": {"file_count": 0, "size_bytes": 0, "files": []}})
            if not candidate_revision:
                row["blocked_reasons"] = ["hf_revision_unpinned"]
        elif not allow_downloads:
            row.update({"status": "blocked", "blocked_reasons": ["hf_downloads_disabled"]})
        elif not candidate_revision:
            row.update({"status": "blocked", "blocked_reasons": ["hf_revision_unpinned"]})
        elif local_source_dir is not None:
            source = local_source_dir / _safe_name(repo_id)
            result = _copy_local_source(source, target)
            row.update(result)
        elif use_hf_api:
            row.update(
                _materialize_from_hf(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    target=target,
                    revision=candidate_revision,
                    allow_patterns=list(allow_patterns or []),
                    ignore_patterns=list(ignore_patterns or []),
                )
            )
        else:
            row.update({"status": "blocked", "blocked_reasons": ["hf_api_not_enabled"]})
        artifacts.append(row)
    materialized = sum(1 for item in artifacts if item.get("status") == "materialized")
    blocked = sum(1 for item in artifacts if item.get("status") == "blocked")
    status = "empty" if not candidates else "planned" if dry_run else "materialized" if materialized else "blocked"
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "report_date": str(report_date),
        "generated_at": _iso(generated),
        "status": status,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "cache_dir": str(cache_dir),
        "summary": {
            "approved_candidates": len(candidates),
            "materialized": materialized,
            "blocked": blocked,
            "dry_run": bool(dry_run),
            "downloads_enabled": bool(allow_downloads),
        },
        "artifacts": artifacts,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "research_only": True,
        "operator_action": "review_cached_research_artifacts_offline" if materialized else "no_hf_cache_action",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--intake-json", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-source-dir", type=Path, default=None)
    parser.add_argument("--use-hf-api", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    allow_downloads = bool(args.allow_downloads) or _truthy(
        get_env("AI_TRADING_HF_ALLOW_DOWNLOADS", "0", cast=str, resolve_aliases=False)
    )
    report = build_huggingface_cache_materialization(
        report_date=str(args.report_date),
        intake=_read_json(args.intake_json or _default_intake_path()),
        cache_dir=_default_cache_dir(args.cache_dir),
        allow_downloads=allow_downloads,
        use_hf_api=bool(args.use_hf_api),
        dry_run=bool(args.dry_run),
        local_source_dir=args.local_source_dir,
        max_candidates=int(args.max_candidates),
        revision=str(args.revision or "main"),
        allow_patterns=list(args.allow_pattern or []),
        ignore_patterns=list(args.ignore_pattern or []),
    )
    default_output, default_latest = _default_paths(str(args.report_date))
    output = args.output_json or default_output
    latest = args.latest_json or default_latest
    report.setdefault("paths", {})
    report["paths"].update({"dated": str(output), "latest": str(latest)})
    _write_json(output, report)
    _write_json(latest, report)
    sys.stdout.write(
        json.dumps(
            {
                "status": report["status"],
                "materialized": report["summary"]["materialized"],
                "output_json": str(output),
                "latest_json": str(latest),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
