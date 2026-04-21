from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import joblib

from ai_trading.logging import get_logger
from ai_trading.models.artifacts import write_artifact_manifest

logger = get_logger(__name__)


def _load_legacy_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def migrate_pickle_artifact(
    source: Path,
    destination: Path,
    *,
    kind: str,
) -> Path:
    """Convert a legacy pickle artifact into a supported format."""
    payload = _load_legacy_pickle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "model":
        joblib.dump(payload, destination)
        write_artifact_manifest(
            model_path=destination,
            model_version="pickle-migration-v1",
            metadata={"source_path": str(source), "migration_kind": kind},
        )
        return destination
    if kind in {"checkpoint", "history"}:
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return destination
    raise ValueError(f"Unsupported migration kind: {kind}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy pickle artifacts into supported formats.",
    )
    parser.add_argument("source", type=Path, help="Legacy pickle artifact path.")
    parser.add_argument("destination", type=Path, help="Converted artifact output path.")
    parser.add_argument(
        "--kind",
        choices=("model", "checkpoint", "history"),
        required=True,
        help="Artifact kind to convert.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = migrate_pickle_artifact(
        args.source.expanduser().resolve(),
        args.destination.expanduser().resolve(),
        kind=str(args.kind),
    )
    logger.info(
        "PICKLE_ARTIFACT_MIGRATED",
        extra={
            "source": str(args.source),
            "destination": str(output),
            "kind": str(args.kind),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
