"""Startup run-manifest writer for reproducibility and auditability."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = (completed.stdout or "").strip()
    return value or None


def _safe_account_id(cfg: Any) -> str | None:
    account = getattr(cfg, "account_id", None)
    if account:
        text = str(account)
        return text[-6:] if len(text) > 6 else text
    key = str(getattr(cfg, "alpaca_api_key", "") or "")
    if not key:
        return None
    return key[-6:] if len(key) > 6 else key


def _redacted_cfg(cfg: Any) -> dict[str, Any]:
    snapshot_fn = getattr(cfg, "snapshot_sanitized", None)
    if callable(snapshot_fn):
        return dict(snapshot_fn())
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        raw = dict(to_dict())
    else:
        raw = dict(getattr(cfg, "__dict__", {}))

    redacted: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key).lower()
        if any(token in key_text for token in ("secret", "token", "password", "key")):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _config_hash(cfg_payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(cfg_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enabled_flags(cfg_payload: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    for key, value in cfg_payload.items():
        if isinstance(value, bool) and value:
            flags.append(str(key))
    return sorted(flags)


def _default_manifest_path() -> str:
    """Return configured run-manifest path with env fallback."""

    try:
        value = get_env("AI_TRADING_RUN_MANIFEST_PATH", "runtime/run_manifest.json")
    except Exception:
        value = "runtime/run_manifest.json"
    normalized = str(value or "").strip()
    return normalized or "runtime/run_manifest.json"


def _resolve_manifest_path(cfg: Any, explicit_path: str | None) -> Path:
    """Resolve run-manifest target path independent of process CWD."""

    configured = explicit_path
    if not configured:
        configured = str(getattr(cfg, "run_manifest_path", "") or "").strip()
    if not configured:
        configured = _default_manifest_path()
    target = Path(str(configured)).expanduser()
    if target.is_absolute():
        return target
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / target).resolve()


def build_run_manifest(
    cfg: Any,
    *,
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_payload = _redacted_cfg(cfg)
    mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode,
        "account_id": _safe_account_id(cfg),
        "resolved_config_hash": _config_hash(cfg_payload),
        "enabled_feature_flags": _enabled_flags(cfg_payload),
        "git_commit_hash": _git_commit_hash(),
        "runtime_contract": dict(runtime_contract or {"stubs_enabled": False}),
    }
    return manifest


def write_run_manifest(
    cfg: Any,
    *,
    runtime_contract: Mapping[str, Any] | None = None,
    path: str | None = None,
) -> Path:
    manifest = build_run_manifest(cfg, runtime_contract=runtime_contract)
    target = _resolve_manifest_path(cfg, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    logger.info("RUN_MANIFEST_WRITTEN", extra={"path": str(target)})
    return target
