"""One registered development-only model trial; never activates a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.logging import get_logger
from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_COLUMNS
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import (
    claim_campaign_trial, finish_campaign_trial, register_campaign,
)
from ai_trading.tools.research_feasibility import (
    FEATURE_LOOKBACK, IntradayGrid, chronological_folds, intraday_schedule,
    load_governed_development_timestamps,
)
from ai_trading.tools.stock_development_readiness import valid_bars
from ai_trading.utils.market_calendar import is_trading_day, session_info

LOGGER = get_logger(__name__)
CONTRACT_HASH = '7e41c0bb3b658ebc9e1f21fbf453098fc7aaf3490a66c5a9268f3b19b63f68e5'
FEATURES = list(DAY_SLEEVE_ML_FEATURE_COLUMNS)
LEGACY_CONTRACT = Path('config/model_replacement_campaign.json')
LEGACY_DIRECTORY = 'artifacts/model_replacement_20260921'
LEGACY_CLASSIFIER: dict[str, Any] = {'family': 'logistic_regression',
                                     'params': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 42}}
CLASSIFIER_FAMILIES = ('logistic_regression', 'hist_gradient_boosting')


@dataclass(frozen=True)
class TrialSpec:
    """Contract values that fix labels, the classifier and the bootstrap."""

    grid: IntradayGrid = IntradayGrid(5)
    cost_bps: float = 10.0
    threshold: float = .5
    classifier_family: str = 'logistic_regression'
    classifier_params: tuple[tuple[str, Any], ...] = tuple(sorted(LEGACY_CLASSIFIER['params'].items()))
    seed: int = 42
    replicates: int = 10000


def contract_spec(contract: dict[str, Any], *, legacy: bool = False) -> TrialSpec:
    """Only the registered Sept 21 contract may rely on its historical defaults."""
    grid = IntradayGrid.from_protocol(contract)
    classifier = contract.get('classifier')
    if not legacy and (grid is None or classifier is None):
        raise ValueError('contract must declare exit_horizon_minutes and classifier')
    classifier = LEGACY_CLASSIFIER if classifier is None else classifier
    if (not isinstance(classifier, dict) or set(classifier) != {'family', 'params'}
            or classifier['family'] not in CLASSIFIER_FAMILIES or not isinstance(classifier['params'], dict)):
        raise ValueError('unsupported classifier specification')
    replicates = int(contract['bootstrap']['replicates'])
    if replicates <= 0 or replicates % 100:
        raise ValueError('bootstrap replicates must be a positive multiple of 100')
    return TrialSpec(grid=grid or IntradayGrid(5), cost_bps=float(contract['cost_bps']),
                     threshold=float(contract['decision_threshold']), classifier_family=classifier['family'],
                     classifier_params=tuple(sorted(classifier['params'].items())),
                     seed=int(contract['bootstrap']['seed']), replicates=replicates)


def build_classifier(spec: TrialSpec) -> Any:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(spec.classifier_params)
    if spec.classifier_family == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**params))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def evidence_signature(contract_hash: str, provenance: dict[str, Any], code_hashes: dict[str, str]) -> str:
    stable = {key: value for key, value in provenance.items() if key != 'generated_at'}
    return hashlib.sha256(json.dumps(dict(contract=contract_hash, provenance=stable,
                                         code=code_hashes), sort_keys=True).encode()).hexdigest()


def regular_bars(frame: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    """Require every source minute, including early closes, before aggregation."""
    parts = []
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('source timestamps must be unique and ordered')
    for day in sessions:
        info = session_info(pd.Timestamp(day).date())
        expected = pd.date_range(info.start_utc, info.end_utc, freq='min', inclusive='left')
        part = frame.loc[(frame.index >= expected[0]) & (frame.index <= expected[-1])]
        if not part.index.equals(expected) or not valid_bars(part).all():
            raise ValueError(f'incomplete or invalid regular session: {day}')
        parts.append(part.resample('5min', origin=expected[0]).agg(
            dict(open='first', high='max', low='min', close='last', volume='sum')))
    return pd.concat(parts)


def labeled_rows(minutes: pd.DataFrame, bars: pd.DataFrame, *, symbol: str,
                 spec: TrialSpec = TrialSpec()) -> pd.DataFrame:
    """Canonical features computed independently from the actual rolling history."""
    rows = []
    cost = spec.cost_bps
    for i, stamp, decision, entry, exit_at in intraday_schedule(bars.index, spec.grid):
        history = bars.iloc[bars.index.searchsorted(decision - FEATURE_LOOKBACK):i + 1]
        if entry not in minutes.index or exit_at not in minutes.index:
            raise ValueError('missing entry or exit bar')
        features = build_day_sleeve_features(history).iloc[0]
        entry_price, exit_price = float(minutes.loc[entry, 'open']), float(minutes.loc[exit_at, 'open'])
        gross = (exit_price / entry_price - 1) * 10000
        rows.append(dict(symbol=symbol, session=str(stamp.tz_convert('America/New_York').date()),
                         feature_start=history.index[0], bar_start=stamp, decision_at=decision,
                         entry_at=entry, exit_at=exit_at, entry_price=entry_price, exit_price=exit_price,
                         gross_bps=gross, net_bps=gross - cost, label=int(gross > cost),
                         **{c: float(features[c]) for c in FEATURES}))
        if len(rows) % 2000 == 0:
            LOGGER.info('REPLACEMENT_FEATURE_PROGRESS', extra={'symbol': symbol, 'rows': len(rows)})
    if not rows:
        raise ValueError('no supported feature and label rows')
    return pd.DataFrame(rows)


def validate_rows(rows: pd.DataFrame, contract: dict[str, Any], spec: TrialSpec = TrialSpec()) -> None:
    if rows.empty or rows.duplicated(['symbol', 'decision_at']).any():
        raise ValueError('empty or duplicate derived observations')
    if set(rows.symbol) != set(contract['symbols']):
        raise ValueError('derived symbols differ from contract')
    if not np.isfinite(rows[FEATURES + ['entry_price', 'exit_price', 'gross_bps', 'net_bps']]).all().all():
        raise ValueError('nonfinite derived values')
    if not rows.session.between(contract['development_start'], contract['development_end']).all():
        raise ValueError('derived data outside development')
    for col in ['feature_start', 'bar_start', 'decision_at', 'entry_at', 'exit_at']:
        if rows[col].dt.tz is None or rows[col].isna().any():
            raise ValueError('derived timestamps must be UTC aware and nonmissing')
    if not (rows.feature_start <= rows.bar_start).all():
        raise ValueError('feature history starts after decision bar')
    if not (rows.decision_at == rows.bar_start + pd.Timedelta(minutes=5, seconds=2)).all():
        raise ValueError('decision finality mismatch')
    if not (rows.entry_at == rows.decision_at.dt.ceil('min')).all():
        raise ValueError('entry must be strictly after finalized decision')
    if not (rows.exit_at == rows.entry_at + pd.Timedelta(minutes=spec.grid.exit_minutes)).all():
        raise ValueError('exit horizon mismatch')
    cutoffs = {day: spec.grid.cutoff(pd.Timestamp(day).date()) for day in rows.session.unique()}
    if not (rows.exit_at < pd.to_datetime(rows.session.map(cutoffs), utc=True)).all():
        raise ValueError('exit at or after flatten cutoff')
    if not (rows.exit_at.dt.tz_convert('America/New_York').dt.date.astype(str) == rows.session).all():
        raise ValueError('label crosses session date')
    gross = (rows.exit_price / rows.entry_price - 1) * 10000
    if not np.allclose(rows.gross_bps, gross) or not np.allclose(rows.net_bps, gross - spec.cost_bps):
        raise ValueError('label arithmetic mismatch')
    if not (rows.label == (rows.net_bps > 0).astype(int)).all():
        raise ValueError('label class mismatch')
    for _, group in rows.sort_values('entry_at').groupby('symbol'):
        if (group.entry_at.iloc[1:].to_numpy() < group.exit_at.iloc[:-1].to_numpy()).any():
            raise ValueError('overlapping opportunities')


def splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Six fixed session chunks: first train, next five expanding test folds."""
    days = sorted(rows.session.unique())
    if len(days) < 30:
        raise ValueError('insufficient development sessions')
    folds = []
    for embargo_day, chunk in chronological_folds(days):
        cutoff = pd.Timestamp(session_info(pd.Timestamp(embargo_day).date()).start_utc)
        train = np.flatnonzero((rows.session < embargo_day) & (rows.exit_at < cutoff))
        test = np.flatnonzero(rows.session.isin(chunk))
        if len(train) == 0 or len(test) < 250 or rows.iloc[train].label.nunique() != 2:
            raise ValueError('unsupported fold before trial claim')
        folds.append((train, test))
    return folds


def bootstrap_lower(rows: pd.DataFrame, values: np.ndarray, *, seed: int = 42, replicates: int = 10000) -> float:
    grouped = pd.DataFrame({'session': rows.session.to_numpy(), 'value': values}).groupby('session').value.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    means = []
    sums, counts = grouped['sum'].to_numpy(), grouped['count'].to_numpy()
    for _ in range(replicates // 100):
        indices = rng.integers(0, len(grouped), size=(100, len(grouped)))
        means.extend((sums[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    return float(np.quantile(means, .025))


def evaluate(rows: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]],
             spec: TrialSpec = TrialSpec()) -> dict[str, Any]:
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    results: list[dict[str, Any]] = []
    predictions = []
    for number, (train, test) in enumerate(folds, 1):
        model = build_classifier(spec)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(rows.iloc[train][FEATURES], rows.iloc[train].label)
        scores = model.predict_proba(rows.iloc[test][FEATURES])[:, 1]
        observed = rows.iloc[test].copy()
        observed['probability'] = scores
        observed['selected'] = scores >= spec.threshold
        observed['strategy_bps'] = np.where(observed.selected, observed.net_bps, 0.)
        observed['fold'] = number
        predictions.append(observed)
        results.append(dict(fold=number, train_rows=len(train), test_rows=len(test),
                            train_end=str(rows.iloc[train].session.max()),
                            test_start=str(observed.session.min()), test_end=str(observed.session.max()),
                            selected_trades=int(observed.selected.sum()),
                            mean_net_bps_per_opportunity=float(observed.strategy_bps.mean())))
    oof = pd.concat(predictions, ignore_index=True)
    lower = bootstrap_lower(oof, oof.strategy_bps.to_numpy(), seed=spec.seed, replicates=spec.replicates)
    improvement = oof.strategy_bps - oof.net_bps
    supported = all(f['selected_trades'] >= 250 for f in results)
    passed = (supported and sum(f['mean_net_bps_per_opportunity'] > 0 for f in results) >= 4
              and float(oof.strategy_bps.mean()) > 0 and lower > 0 and float(improvement.mean()) > 0)
    return dict(status=('development_screen_passed' if passed else 'hypothesis_rejected' if supported else 'inconclusive_budget_consumed'),
                folds=results, oof_rows=len(oof), selected_trades=int(oof.selected.sum()),
                mean_net_bps_per_opportunity=float(oof.strategy_bps.mean()),
                always_long_net_bps_per_opportunity=float(oof.net_bps.mean()),
                paired_improvement_bps=float(improvement.mean()), bootstrap_lower_95_bps=lower,
                development_screen_passed=bool(passed), promotion_authority=False, runtime_authority=False,
                qualification_status='not_qualified', existing_qualification_gates='not_evaluated_no_promotion_authority',
                holdout_evaluated=False, model_artifact_saved=False,
                evidence_type='historical_bar_open_proxy_not_executable_fills',
                _predictions=oof)


def run(root: Path, *, fit: bool, contract_path: Path = LEGACY_CONTRACT) -> dict[str, Any]:
    root = root.resolve()
    contract_path = (root / contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    legacy = canonical == CONTRACT_HASH
    if not legacy and contract_path == (root / LEGACY_CONTRACT).resolve():
        raise ValueError('replacement contract differs from registered specification')
    if not legacy and not re.fullmatch(r'[a-z0-9_]+', str(contract.get('campaign_id', ''))):
        raise ValueError('contract campaign_id must name its artifact directory')
    spec = contract_spec(contract, legacy=legacy)
    build_classifier(spec)  # Reject an invalid classifier before any data read or claim.
    directory = root / (LEGACY_DIRECTORY if legacy else f"artifacts/{contract['campaign_id']}")
    state_path = directory / 'campaign_state.json'
    # Require the original registration. A different root must not create a budget.
    if not state_path.is_file():
        raise ValueError('original campaign registration is required')
    state = register_campaign(state_path, contract)
    if state['trials']:
        return {'status': 'budget_already_claimed', 'trials': state['trials']}
    _, provenance = load_governed_development_timestamps(contract, root / contract['acquisition'])
    days = [str(d.date()) for d in pd.date_range(contract['development_start'], contract['development_end']) if is_trading_day(d.date())]
    code_root = Path(__file__).resolve().parents[1]
    code_files = ['tools/model_replacement_trial.py', 'features/day_sleeve.py', 'features/indicators.py',
                  'models/contracts.py', 'tools/research_feasibility.py', 'tools/stock_development_readiness.py',
                  'indicators/__init__.py', 'utils/market_calendar.py']
    code_hashes = {p: digest(code_root / p) for p in code_files}
    signature = evidence_signature(canonical, provenance, code_hashes)
    parts = []
    for symbol in contract['symbols']:
        output, manifest = directory / f'{symbol}.parquet', directory / f'{symbol}.manifest.json'
        if output.exists() and manifest.exists():
            meta = json.loads(manifest.read_text())
            if meta.get('signature') == signature and meta.get('sha256') == digest(output):
                parts.append(pd.read_parquet(output))
                continue
        source = provenance['sources'][symbol]
        path = Path(source['path'])
        if digest(path) != source['sha256']:
            raise ValueError('source changed before feature construction')
        minutes = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        minutes.index = pd.DatetimeIndex(pd.to_datetime(minutes.pop('timestamp'), utc=True))
        if minutes.index.min() < pd.Timestamp(contract['development_start'], tz='UTC') or minutes.index.max() >= pd.Timestamp(contract['development_end'], tz='UTC') + pd.Timedelta(days=1):
            raise ValueError('source outside development interval')
        part = labeled_rows(minutes, regular_bars(minutes, days), symbol=symbol, spec=spec)
        if digest(path) != source['sha256']:
            raise ValueError('source changed during feature construction')
        temporary = output.with_suffix('.parquet.tmp')
        part.to_parquet(temporary, index=False)
        temporary.replace(output)
        write_json(manifest, dict(signature=signature, sha256=digest(output), source=source,
                                  code=code_hashes, rows=len(part), promotion_authority=False))
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True).sort_values(['decision_at', 'symbol']).reset_index(drop=True)
    validate_rows(rows, contract, spec)
    folds = splits(rows)
    preflight = dict(status='preflight_passed', signature=signature, rows=len(rows),
                     folds=len(folds), provenance=provenance, code=code_hashes,
                     trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    write_json(directory / 'preflight.json', preflight)
    if not fit:
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract['hypotheses'][0], evidence_signature=signature,
                         evaluation_start=contract['development_start'], evaluation_end=contract['development_end'], quality_passed=True)
    # A crash after claim intentionally prevents automatic refitting.
    report = evaluate(rows, folds, spec)
    predictions = report.pop('_predictions')
    predictions.to_parquet(directory / 'oof_predictions.parquet', index=False)
    report.update(signature=signature, contract_hash=canonical, predictions_sha256=digest(directory / 'oof_predictions.parquet'))
    report_path = directory / 'report.json'
    write_json(report_path, report)
    finish_campaign_trial(state_path, evidence_signature=signature, decision=report['status'], report_path=report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--contract', type=Path, default=LEGACY_CONTRACT,
                        help='campaign contract, relative to the repository root unless absolute')
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args()
    result = run(args.repository_root, fit=args.fit, contract_path=args.contract)
    LOGGER.info('REPLACEMENT_TRIAL_RESULT', extra={'result': result})


if __name__ == '__main__':
    main()
