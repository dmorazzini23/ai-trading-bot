"""Execution runtime metadata helpers extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import copy
import importlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _copy_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return value


def _normalize_symbol_mapping(raw: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return normalized
    for raw_key, value in raw.items():
        key = str(raw_key).strip().upper()
        if not key:
            continue
        normalized[key] = _copy_value(value)
    return normalized


def _normalize_symbol_float_mapping(raw: Any) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if not isinstance(raw, Mapping):
        return normalized
    for raw_key, value in raw.items():
        key = str(raw_key).strip().upper()
        if not key:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        normalized[key] = float(numeric)
    return normalized


def _normalize_capability_mapping(raw: Any) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    if not isinstance(raw, Mapping):
        return normalized
    for raw_key, value in raw.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        normalized[key] = bool(value)
    return normalized


def _copy_plain_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    copied: dict[str, Any] = {}
    for raw_key, value in raw.items():
        key = str(raw_key).strip()
        if not key:
            continue
        copied[key] = _copy_value(value)
    return copied


def _normalize_symbol_list(raw: Iterable[Any] | None) -> list[str]:
    if raw is None:
        return []
    return sorted(
        {
            str(symbol).strip().upper()
            for symbol in raw
            if str(symbol).strip()
        }
    )


@dataclass(slots=True)
class ExecutionPreRankRuntimeState:
    weights: dict[str, Any]
    runtime_rank: dict[str, Any]
    opportunity_quality: dict[str, Any]
    prerank_cycle: int
    last_selected_cycles: dict[str, int]


@dataclass(frozen=True, slots=True)
class ExecutionCandidateRankingRuntimeState:
    opportunity_quality_by_symbol: dict[str, float]
    opportunity_allowed_symbols: list[str]
    opportunity_quality_gate: dict[str, Any]
    candidate_rank: dict[str, float]
    candidate_expected_net_edge_bps: dict[str, float]
    candidate_expected_capture_bps: dict[str, float]
    candidate_rank_realism_factor: dict[str, float]
    candidate_rank_learning_signals: dict[str, Any]
    candidate_rank_context: dict[str, Any]

    @classmethod
    def from_payloads(
        cls,
        *,
        opportunity_quality_by_symbol: Mapping[str, Any] | None = None,
        opportunity_allowed_symbols: Iterable[Any] | None = None,
        opportunity_quality_gate: Mapping[str, Any] | None = None,
        candidate_rank: Mapping[str, Any] | None = None,
        candidate_expected_net_edge: Mapping[str, Any] | None = None,
        candidate_expected_capture: Mapping[str, Any] | None = None,
        edge_realism_rank_factor_by_symbol: Mapping[str, Any] | None = None,
        counterfactual_signal_by_symbol: Mapping[str, Any] | None = None,
        rank_context: Mapping[str, Any] | None = None,
    ) -> ExecutionCandidateRankingRuntimeState:
        """Normalize candidate-ranking runtime metadata at one typed boundary."""

        return cls(
            opportunity_quality_by_symbol=_normalize_symbol_float_mapping(
                opportunity_quality_by_symbol
            ),
            opportunity_allowed_symbols=_normalize_symbol_list(opportunity_allowed_symbols),
            opportunity_quality_gate=_copy_plain_mapping(opportunity_quality_gate),
            candidate_rank=_normalize_symbol_float_mapping(candidate_rank),
            candidate_expected_net_edge_bps=_normalize_symbol_float_mapping(
                candidate_expected_net_edge
            ),
            candidate_expected_capture_bps=_normalize_symbol_float_mapping(
                candidate_expected_capture
            ),
            candidate_rank_realism_factor=_normalize_symbol_float_mapping(
                edge_realism_rank_factor_by_symbol
            ),
            candidate_rank_learning_signals=_normalize_symbol_mapping(
                counterfactual_signal_by_symbol
            ),
            candidate_rank_context=_copy_plain_mapping(rank_context),
        )

    def to_runtime_attributes(self) -> dict[str, Any]:
        """Return the legacy runtime attribute payload shape."""

        return {
            "execution_opportunity_quality_by_symbol": dict(
                self.opportunity_quality_by_symbol
            ),
            "execution_opportunity_quality_allowed_symbols": list(
                self.opportunity_allowed_symbols
            ),
            "execution_opportunity_quality_gate": dict(self.opportunity_quality_gate),
            "execution_candidate_rank": dict(self.candidate_rank),
            "execution_candidate_rank_expected_edge_bps": dict(
                self.candidate_expected_net_edge_bps
            ),
            "execution_candidate_rank_expected_capture_bps": dict(
                self.candidate_expected_capture_bps
            ),
            "execution_candidate_rank_realism_factor": dict(
                self.candidate_rank_realism_factor
            ),
            "execution_candidate_rank_learning_signals": dict(
                self.candidate_rank_learning_signals
            ),
            "execution_candidate_rank_context": dict(self.candidate_rank_context),
        }


def resolve_order_type_capabilities(runtime: Any) -> Mapping[str, bool] | None:
    """Resolve broker order-type capabilities with runtime cache hardening."""

    be = _bot_engine()

    runtime_caps = _normalize_capability_mapping(
        getattr(runtime, "broker_order_type_capabilities", None)
    )
    if runtime_caps:
        setattr(runtime, "broker_order_type_capabilities", dict(runtime_caps))
        return runtime_caps

    exec_engine = getattr(runtime, "execution_engine", None) or getattr(
        runtime,
        "exec_engine",
        None,
    )
    for attr in ("broker_order_type_capabilities", "order_type_capabilities"):
        candidate = _normalize_capability_mapping(getattr(exec_engine, attr, None))
        if candidate:
            setattr(runtime, "broker_order_type_capabilities", dict(candidate))
            return candidate

    env_caps: dict[str, bool] = {}
    env_defined = False
    env_map = {
        "limit": "AI_TRADING_BROKER_SUPPORTS_LIMIT",
        "market": "AI_TRADING_BROKER_SUPPORTS_MARKET",
        "stop": "AI_TRADING_BROKER_SUPPORTS_STOP",
        "stop_limit": "AI_TRADING_BROKER_SUPPORTS_STOP_LIMIT",
        "trailing_stop": "AI_TRADING_BROKER_SUPPORTS_TRAILING_STOP",
        "bracket": "AI_TRADING_BROKER_SUPPORTS_BRACKET",
        "oco": "AI_TRADING_BROKER_SUPPORTS_OCO",
        "oto": "AI_TRADING_BROKER_SUPPORTS_OTO",
    }
    for capability, env_key in env_map.items():
        raw_value = be.get_env(env_key, None)
        if raw_value is None:
            continue
        env_defined = True
        env_caps[capability] = bool(be.get_env(env_key, "0", cast=bool))
    if env_defined and env_caps:
        setattr(runtime, "broker_order_type_capabilities", dict(env_caps))
        be.logger.info(
            "ORDER_TYPE_CAPABILITIES_CONFIGURED",
            extra={"capabilities": env_caps},
        )
        return env_caps

    engine_module = ""
    if exec_engine is not None:
        engine_module = str(getattr(exec_engine.__class__, "__module__", "") or "")
    if engine_module.startswith("ai_trading.execution.live_trading"):
        inferred_caps = {
            "limit": True,
            "market": True,
            "stop": True,
            "stop_limit": True,
            "trailing_stop": True,
            "bracket": False,
            "oco": False,
            "oto": False,
        }
        setattr(runtime, "broker_order_type_capabilities", dict(inferred_caps))
        be.logger.warning(
            "ORDER_TYPE_CAPABILITIES_INFERRED",
            extra={
                "engine_module": engine_module,
                "capabilities": inferred_caps,
                "note": "Set AI_TRADING_BROKER_SUPPORTS_* for explicit capability config.",
            },
        )
        return inferred_caps

    return None


def load_execution_prerank_runtime_state(runtime: Any) -> ExecutionPreRankRuntimeState:
    """Return copied/normalized prerank runtime metadata for the current cycle."""

    be = _bot_engine()
    weights = _normalize_symbol_mapping(getattr(runtime, "portfolio_weights", None))
    runtime_rank = _normalize_symbol_mapping(
        getattr(runtime, "execution_candidate_rank", None)
    )
    opportunity_quality = _normalize_symbol_mapping(
        getattr(runtime, "execution_opportunity_quality_by_symbol", None)
    )

    cycle_candidate = getattr(runtime, "_execution_prerank_cycle_idx", 0)
    try:
        prerank_cycle = max(int(cycle_candidate), 0) + 1
    except (TypeError, ValueError):
        prerank_cycle = 1
    try:
        setattr(runtime, "_execution_prerank_cycle_idx", prerank_cycle)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        be.logger.debug("EXECUTION_CANDIDATE_PRERANK_CYCLE_SET_FAILED", exc_info=True)

    last_selected_cycles: dict[str, int] = {}
    history_candidate = getattr(runtime, "_execution_candidate_last_selected_cycle", None)
    if isinstance(history_candidate, Mapping):
        for raw_symbol, raw_cycle in history_candidate.items():
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                continue
            try:
                parsed_cycle = int(raw_cycle)
            except (TypeError, ValueError):
                continue
            if parsed_cycle > 0:
                last_selected_cycles[symbol] = parsed_cycle

    return ExecutionPreRankRuntimeState(
        weights=weights,
        runtime_rank=runtime_rank,
        opportunity_quality=opportunity_quality,
        prerank_cycle=prerank_cycle,
        last_selected_cycles=last_selected_cycles,
    )


def store_execution_prerank_runtime_state(
    runtime: Any,
    *,
    selected_symbols: Iterable[str],
    prerank_cycle: int,
    last_selected_cycles: Mapping[str, int],
) -> None:
    """Persist normalized candidate selection history for future prerank cycles."""

    be = _bot_engine()
    cycle_value = max(int(prerank_cycle), 0)
    history = {
        str(symbol).strip().upper(): int(cycle)
        for symbol, cycle in last_selected_cycles.items()
        if str(symbol).strip() and int(cycle) > 0
    }
    if cycle_value > 0:
        for raw_symbol in selected_symbols:
            symbol = str(raw_symbol).strip().upper()
            if symbol:
                history[symbol] = cycle_value
    try:
        setattr(runtime, "_execution_candidate_last_selected_cycle", history)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        be.logger.debug("EXECUTION_CANDIDATE_PRERANK_HISTORY_SET_FAILED", exc_info=True)


def store_execution_candidate_ranking_runtime_state(
    runtime: Any,
    *,
    opportunity_quality_by_symbol: Mapping[str, Any],
    opportunity_allowed_symbols: Iterable[str],
    opportunity_quality_gate: Mapping[str, Any],
    candidate_rank: Mapping[str, Any],
    candidate_expected_net_edge: Mapping[str, Any],
    candidate_expected_capture: Mapping[str, Any],
    edge_realism_rank_factor_by_symbol: Mapping[str, Any],
    counterfactual_signal_by_symbol: Mapping[str, Any],
    rank_context: Mapping[str, Any],
) -> None:
    """Persist copied execution-ranking metadata onto runtime."""

    ranking_state = ExecutionCandidateRankingRuntimeState.from_payloads(
        opportunity_quality_by_symbol=opportunity_quality_by_symbol,
        opportunity_allowed_symbols=opportunity_allowed_symbols,
        opportunity_quality_gate=opportunity_quality_gate,
        candidate_rank=candidate_rank,
        candidate_expected_net_edge=candidate_expected_net_edge,
        candidate_expected_capture=candidate_expected_capture,
        edge_realism_rank_factor_by_symbol=edge_realism_rank_factor_by_symbol,
        counterfactual_signal_by_symbol=counterfactual_signal_by_symbol,
        rank_context=rank_context,
    )
    for attribute, value in ranking_state.to_runtime_attributes().items():
        setattr(runtime, attribute, value)


__all__ = [
    "ExecutionCandidateRankingRuntimeState",
    "ExecutionPreRankRuntimeState",
    "load_execution_prerank_runtime_state",
    "resolve_order_type_capabilities",
    "store_execution_candidate_ranking_runtime_state",
    "store_execution_prerank_runtime_state",
]
