"""Portfolio rebalancing utilities with tax awareness and advanced features."""
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any
import numpy as np
try:  # pragma: no cover - Alpaca optional
    from alpaca.common.exceptions import APIError as _ImportedAlpacaAPIError
except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # ImportError
    raise RuntimeError(
        "alpaca-py==0.42.1 is required; install with `pip install alpaca-py==0.42.1`"
    ) from exc
else:
    APIError = _ImportedAlpacaAPIError
from ai_trading.config import get_settings
import importlib

def _resolve_portfolio_callable(name: str):
    module = importlib.import_module("ai_trading.portfolio")
    attr = getattr(module, name)
    if not callable(attr):
        raise AttributeError(f"ai_trading.portfolio.{name} is not callable")
    return attr

try:  # pragma: no cover - allow tests to inject stubs
    _compute_portfolio_weights = _resolve_portfolio_callable("compute_portfolio_weights")
except AI_TRADING_FALLBACK_EXCEPTIONS:
    _compute_portfolio_weights = None

def compute_portfolio_weights(*args, **kwargs):
    """Call ``ai_trading.portfolio.compute_portfolio_weights`` with lazy resolution."""
    global _compute_portfolio_weights
    if _compute_portfolio_weights is None:
        _compute_portfolio_weights = _resolve_portfolio_callable("compute_portfolio_weights")
    return _compute_portfolio_weights(*args, **kwargs)
from ai_trading.settings import get_rebalance_interval_min
from ai_trading.utils.time import safe_utcnow
from ai_trading.config.management import get_env

logger = get_logger(__name__)

def apply_no_trade_bands(current: dict[str, float], target: dict[str, float], band_bps: float | dict[str, float]=25.0) -> dict[str, float]:
    """
    Suppress small reallocations that are inside a no-trade band (in basis points).
    Example: band_bps=25 means ignore deltas smaller than 0.25% absolute weight.
    """
    def _band_for(sym: str) -> float:
        try:
            if isinstance(band_bps, dict):
                return float(band_bps.get(sym, 25.0)) / 10000.0
            return float(band_bps) / 10000.0
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return 0.0025
    out = {}
    for sym, tgt in target.items():
        cur = current.get(sym, 0.0)
        band = _band_for(sym)
        if abs(tgt - cur) < band:
            out[sym] = cur
        else:
            out[sym] = tgt
    return out
from ai_trading.core.constants import RISK_PARAMETERS
try:  # pragma: no cover - allow tests to inject stubs
    _create_portfolio_optimizer = _resolve_portfolio_callable("create_portfolio_optimizer")
except AI_TRADING_FALLBACK_EXCEPTIONS:
    _create_portfolio_optimizer = None

def create_portfolio_optimizer(*args, **kwargs):
    """Return the portfolio optimizer, ensuring the real module is loaded."""
    global _create_portfolio_optimizer
    if _create_portfolio_optimizer is None:
        _create_portfolio_optimizer = _resolve_portfolio_callable("create_portfolio_optimizer")
    return _create_portfolio_optimizer(*args, **kwargs)
from ai_trading.risk.adaptive_sizing import AdaptivePositionSizer
from ai_trading.strategies.regime_detector import create_regime_detector

def rebalance_interval_min() -> int:
    return int(get_rebalance_interval_min())

_last_rebalance = safe_utcnow()
_rebalancer: "TaxAwareRebalancer | None" = None


def init_rebalancer() -> "TaxAwareRebalancer":
    """Initialize the global tax-aware rebalancer."""
    global _rebalancer
    if _rebalancer is None:
        _rebalancer = TaxAwareRebalancer()
        logger.info('Portfolio-first trading capabilities loaded')
    return _rebalancer


def _rebalance_order_side(current_quantity: float, trade_quantity: int) -> str:
    """Return the order side needed to move a signed position toward target."""
    if trade_quantity > 0:
        return "buy_to_cover" if current_quantity < 0.0 else "buy"
    if trade_quantity < 0:
        return "sell_short" if current_quantity <= 0.0 else "sell"
    return "hold"


def _extract_position_quantity(position: Any) -> float:
    """Extract signed share quantity from runtime position payloads."""
    if isinstance(position, dict):
        raw_quantity = position.get('quantity', position.get('qty', 0.0))
    else:
        raw_quantity = getattr(position, 'quantity', getattr(position, 'qty', position))
    return float(raw_quantity)


def _finite_positive(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _rebalance_basis(position: Any) -> tuple[float | None, datetime | None]:
    if isinstance(position, dict):
        raw_price = position.get('entry_price', position.get('purchase_price'))
        raw_date = position.get('entry_date', position.get('purchase_date'))
    else:
        raw_price = getattr(position, 'entry_price', getattr(position, 'purchase_price', None))
        raw_date = getattr(position, 'entry_date', getattr(position, 'purchase_date', None))
    entry_price = _finite_positive(raw_price)
    if isinstance(raw_date, datetime):
        entry_date = raw_date if raw_date.tzinfo is not None else raw_date.replace(tzinfo=UTC)
    else:
        try:
            entry_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            entry_date = None
        if entry_date is not None and entry_date.tzinfo is None:
            entry_date = entry_date.replace(tzinfo=UTC)
    return entry_price, entry_date


def _has_rebalance_basis(position: Any) -> bool:
    entry_price, entry_date = _rebalance_basis(position)
    return entry_price is not None and entry_date is not None


class TaxAwareRebalancer:
    """
    Tax-aware portfolio rebalancing with loss harvesting and wash sale avoidance.

    Implements sophisticated rebalancing that considers tax implications,
    capital gains/losses, and optimal timing for portfolio adjustments.
    """

    def __init__(self, tax_rate_short: float=0.37, tax_rate_long: float=0.2):
        """Initialize tax-aware rebalancer."""
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        self.wash_sale_days = 31
        settings = get_settings()
        if settings.ENABLE_PORTFOLIO_FEATURES:
            self.adaptive_sizer = AdaptivePositionSizer()
            self.max_portfolio_risk = RISK_PARAMETERS['MAX_PORTFOLIO_RISK']
        self.holding_period_long = 365
        self._portfolio_optimizer: Any | None = None
        self._regime_detector: Any | None = None
        logger.info(
            f'TaxAwareRebalancer initialized with tax rates: short={tax_rate_short:.1%}, long={tax_rate_long:.1%}'
        )

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def _within_wash_sale_window(self, value: Any) -> bool:
        event_dt = self._coerce_datetime(value)
        if event_dt is None:
            return False
        return (datetime.now(UTC) - event_dt).days < self.wash_sale_days

    def _has_recent_loss_sale(self, position: dict[str, Any]) -> bool:
        for key in ("last_loss_sale_date", "last_harvest_sale_date", "last_sale_date"):
            if self._within_wash_sale_window(position.get(key)):
                return True
        return False

    def _has_recent_replacement_purchase(self, position: dict[str, Any]) -> bool:
        for key in ("replacement_purchase_date", "last_purchase_date", "last_buy_date"):
            if self._within_wash_sale_window(position.get(key)):
                return True
        return False

    @property
    def portfolio_optimizer(self):
        if self._portfolio_optimizer is None:
            self._portfolio_optimizer = create_portfolio_optimizer()
        return self._portfolio_optimizer

    @property
    def regime_detector(self):
        if self._regime_detector is None:
            self._regime_detector = create_regime_detector()
        return self._regime_detector

    def calculate_tax_impact(self, position: dict[str, Any], current_price: float) -> dict[str, Any]:
        """
        Calculate tax impact of selling a position.

        Args:
            position: Position data including entry price, quantity, entry date
            current_price: Current market price

        Returns:
            Dictionary with tax impact analysis
        """
        try:
            entry_price = position.get('entry_price', 0.0)
            quantity = position.get('quantity', 0)
            entry_date = position.get('entry_date')
            if not all([entry_price, quantity, current_price, entry_date]):
                return {'error': 'Missing position data'}
            total_gain_loss = (current_price - entry_price) * quantity
            gain_loss_per_share = current_price - entry_price
            gain_loss_pct = gain_loss_per_share / entry_price if entry_price > 0 else 0
            holding_days = (datetime.now(UTC) - entry_date).days
            is_long_term = holding_days >= self.holding_period_long
            applicable_tax_rate = self.tax_rate_long if is_long_term else self.tax_rate_short
            tax_liability = max(0, total_gain_loss * applicable_tax_rate)
            proceeds_before_tax = current_price * quantity
            after_tax_proceeds = proceeds_before_tax - tax_liability
            return {'total_gain_loss': total_gain_loss, 'gain_loss_pct': gain_loss_pct, 'holding_days': holding_days, 'is_long_term': is_long_term, 'applicable_tax_rate': applicable_tax_rate, 'tax_liability': tax_liability, 'proceeds_before_tax': proceeds_before_tax, 'after_tax_proceeds': after_tax_proceeds, 'tax_efficiency_score': self._calculate_tax_efficiency(total_gain_loss, is_long_term)}
        except (KeyError, ValueError, TypeError) as e:
            logger.error('CALCULATE_TAX_IMPACT_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'error': str(e)}

    def identify_loss_harvesting_opportunities(self, portfolio_positions: dict[str, dict], current_prices: dict[str, float]) -> list[dict]:
        """
        Identify tax loss harvesting opportunities.

        Args:
            portfolio_positions: Dictionary of current positions
            current_prices: Current market prices for positions

        Returns:
            List of loss harvesting opportunities
        """
        try:
            opportunities = []
            for symbol, position in portfolio_positions.items():
                current_price = current_prices.get(symbol, 0)
                if current_price <= 0:
                    continue
                tax_impact = self.calculate_tax_impact(position, current_price)
                if tax_impact.get('error'):
                    continue
                total_loss = tax_impact.get('total_gain_loss', 0)
                if total_loss < 0:
                    last_sale_date = position.get('last_sale_date')
                    can_harvest = True
                    if last_sale_date and self._within_wash_sale_window(last_sale_date):
                        can_harvest = False
                    if self._has_recent_replacement_purchase(position):
                        can_harvest = False
                    if can_harvest:
                        tax_rate = tax_impact.get('applicable_tax_rate', 0)
                        tax_benefit = abs(total_loss) * tax_rate
                        opportunity = {'symbol': symbol, 'position': position, 'current_price': current_price, 'total_loss': total_loss, 'tax_benefit': tax_benefit, 'is_long_term': tax_impact.get('is_long_term', False), 'holding_days': tax_impact.get('holding_days', 0), 'priority_score': self._calculate_harvest_priority(total_loss, tax_benefit, position)}
                        opportunities.append(opportunity)
            opportunities.sort(key=lambda x: x['priority_score'], reverse=True)
            return opportunities
        except (KeyError, ValueError, TypeError) as e:
            logger.error('LOSS_HARVEST_OPS_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return []

    def calculate_optimal_rebalance(self, current_positions: dict[str, dict], target_weights: dict[str, float], current_prices: dict[str, float], account_equity: float) -> dict[str, Any]:
        """
        Calculate tax-optimal rebalancing trades.

        Args:
            current_positions: Current portfolio positions
            target_weights: Target portfolio weights
            current_prices: Current market prices
            account_equity: Total account equity

        Returns:
            Dictionary with optimal rebalancing plan
        """
        try:
            current_weights: dict[str, float] = {}
            if not (math.isfinite(float(account_equity)) and float(account_equity) > 0.0):
                logger.warning('SIZING_SKIPPED', extra={'reason': 'invalid_account_equity'})
                return {'error': 'invalid_account_equity', 'rebalance_trades': []}
            for symbol, position in current_positions.items():
                current_price = float(current_prices.get(symbol, np.nan))
                if not (math.isfinite(current_price) and current_price > 0.0):
                    logger.warning('SIZING_SKIPPED', extra={'reason': 'invalid_price', 'symbol': symbol})
                    continue
                position_value = _extract_position_quantity(position) * current_price
                current_weights[symbol] = position_value / float(account_equity)
            rebalance_trades: list[dict[str, Any]] = []
            total_tax_impact = 0.0
            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight
                if abs(weight_diff) > 0.01:
                    current_price = float(current_prices.get(symbol, np.nan))
                    if not (math.isfinite(current_price) and current_price > 0.0):
                        logger.warning('SIZING_SKIPPED', extra={'reason': 'invalid_price', 'symbol': symbol})
                        continue
                    target_value = target_weight * account_equity
                    current_value = current_weight * account_equity
                    trade_value = target_value - current_value
                    trade_quantity = int(trade_value / current_price)
                    if trade_quantity != 0:
                        position = current_positions.get(symbol, {})
                        if not isinstance(position, dict):
                            position = {
                                'quantity': _extract_position_quantity(position),
                                'entry_price': getattr(position, 'entry_price', getattr(position, 'purchase_price', 0.0)),
                                'entry_date': getattr(position, 'entry_date', getattr(position, 'purchase_date', None)),
                            }
                        current_quantity = _extract_position_quantity(position)
                        order_side = _rebalance_order_side(current_quantity, trade_quantity)
                        has_rebalance_basis = _has_rebalance_basis(position)
                        tax_impact: dict[str, Any] = {
                            'tax_liability': 0,
                            'is_optimal_timing': True,
                            'tax_basis_available': has_rebalance_basis,
                        }
                        if trade_quantity > 0 and self._has_recent_loss_sale(position):
                            logger.info('REBALANCE_BUY_SKIPPED_WASH_SALE', extra={'symbol': symbol})
                            continue
                        if trade_quantity < 0 and current_quantity > 0.0 and symbol in current_positions:
                            if has_rebalance_basis:
                                sell_quantity = min(abs(trade_quantity), current_quantity)
                                partial_position = position.copy()
                                partial_position['quantity'] = sell_quantity
                                tax_impact = self.calculate_tax_impact(partial_position, current_price)
                                tax_impact['tax_basis_available'] = True
                                total_tax_impact += tax_impact.get('tax_liability', 0)
                                holding_days = tax_impact.get('holding_days', 0)
                                total_gain_loss = tax_impact.get('total_gain_loss', 0)
                                if total_gain_loss < 0 and self._has_recent_replacement_purchase(position):
                                    logger.info('REBALANCE_SELL_SKIPPED_WASH_SALE', extra={'symbol': symbol})
                                    continue
                                if holding_days > 300 and holding_days < 365 and (total_gain_loss > 0) and (not tax_impact.get('is_long_term', False)):
                                    tax_impact['is_optimal_timing'] = False
                                    tax_impact['delay_recommendation'] = 365 - holding_days
                            else:
                                tax_impact = {
                                    'tax_liability': 0,
                                    'is_optimal_timing': False,
                                    'tax_basis_available': False,
                                    'reason': 'missing_rebalance_basis',
                                }
                        trade = {'symbol': symbol, 'current_weight': current_weight, 'target_weight': target_weight, 'weight_diff': weight_diff, 'trade_quantity': trade_quantity, 'trade_value': trade_value, 'current_price': current_price, 'side': order_side, 'tax_impact': tax_impact, 'priority': self._calculate_rebalance_priority(weight_diff, tax_impact)}
                        rebalance_trades.append(trade)
            rebalance_trades.sort(key=lambda x: float(x.get('priority', 0.0)), reverse=True)
            return {'rebalance_trades': rebalance_trades, 'total_tax_impact': total_tax_impact, 'current_weights': current_weights, 'target_weights': target_weights, 'portfolio_drift': self._calculate_portfolio_drift(current_weights, target_weights), 'tax_efficiency_score': self._calculate_overall_tax_efficiency(rebalance_trades), 'recommendations': self._generate_rebalance_recommendations(rebalance_trades)}
        except (KeyError, ValueError, TypeError) as e:
            logger.error('CALC_OPTIMAL_REBALANCE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'error': str(e), 'rebalance_trades': []}

    def _calculate_tax_efficiency(self, gain_loss: float, is_long_term: bool) -> float:
        """Calculate tax efficiency score for a position (0-1)."""
        try:
            if gain_loss <= 0:
                return 1.0
            base_score = 0.8 if is_long_term else 0.4
            gain_magnitude_penalty = min(0.3, gain_loss / 10000)
            return max(0, base_score - gain_magnitude_penalty)
        except (KeyError, ValueError, TypeError) as e:
            logger.error('TAX_EFFICIENCY_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return 0.5

    def _calculate_harvest_priority(self, total_loss: float, tax_benefit: float, position: dict) -> float:
        """Calculate priority score for loss harvesting."""
        try:
            base_score = min(1000, abs(tax_benefit))
            loss_bonus = min(500, abs(total_loss) * 0.1)
            entry_date = position.get('entry_date')
            if entry_date:
                days_held = (datetime.now(UTC) - entry_date).days
                recency_penalty = max(0, (31 - days_held) * 10) if days_held < 31 else 0
            else:
                recency_penalty = 0
            return float(base_score + loss_bonus - recency_penalty)
        except (KeyError, ValueError, TypeError) as e:
            logger.error('HARVEST_PRIORITY_FAILED', exc_info=True, extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return 0

    def _calculate_rebalance_priority(self, weight_diff: float, tax_impact: dict) -> float:
        """Calculate priority score for rebalancing trades."""
        try:
            deviation_score = abs(weight_diff) * 100
            tax_liability = tax_impact.get('tax_liability', 0)
            is_optimal_timing = tax_impact.get('is_optimal_timing', True)
            tax_penalty = float(tax_liability) * 0.1
            timing_bonus = 20.0 if bool(is_optimal_timing) else -50.0
            return float(deviation_score - tax_penalty + timing_bonus)
        except (KeyError, ValueError, TypeError) as e:
            logger.error('REBALANCE_PRIORITY_FAILED', exc_info=True, extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return 0

    def _calculate_portfolio_drift(self, current_weights: dict[str, float], target_weights: dict[str, float]) -> float:
        """Calculate overall portfolio drift from target."""
        try:
            total_drift = 0.0
            all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
            for symbol in all_symbols:
                current = current_weights.get(symbol, 0.0)
                target = target_weights.get(symbol, 0.0)
                total_drift += abs(current - target)
            return float(total_drift / 2.0)
        except (KeyError, ValueError, TypeError) as e:
            logger.error('PORTFOLIO_DRIFT_FAILED', exc_info=True, extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return 0

    def _calculate_overall_tax_efficiency(self, rebalance_trades: list[dict]) -> float:
        """Calculate overall tax efficiency of rebalancing plan."""
        try:
            if not rebalance_trades:
                return 1.0
            efficiency_scores = []
            for trade in rebalance_trades:
                tax_impact = trade.get('tax_impact', {})
                is_optimal = tax_impact.get('is_optimal_timing', True)
                tax_liability = tax_impact.get('tax_liability', 0)
                trade_value = abs(trade.get('trade_value', 1))
                if is_optimal and tax_liability == 0:
                    trade_efficiency = 1.0
                else:
                    tax_drag = tax_liability / trade_value if trade_value > 0 else 0
                    timing_penalty = 0 if is_optimal else 0.3
                    trade_efficiency = max(0, 1.0 - tax_drag - timing_penalty)
                efficiency_scores.append(trade_efficiency)
            return sum(efficiency_scores) / len(efficiency_scores)
        except (KeyError, ValueError, TypeError) as e:
            logger.error('TAX_EFFICIENCY_FAILED', exc_info=True, extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return 0.5

    def _generate_rebalance_recommendations(self, rebalance_trades: list[dict]) -> list[str]:
        """Generate recommendations for tax-optimal rebalancing."""
        recommendations = []
        try:
            for trade in rebalance_trades:
                symbol = trade.get('symbol', '')
                tax_impact = trade.get('tax_impact', {})
                if not tax_impact.get('is_optimal_timing', True):
                    delay_days = tax_impact.get('delay_recommendation', 0)
                    if delay_days > 0:
                        recommendations.append(f'Consider delaying sale of {symbol} by {delay_days} days for long-term capital gains treatment')
                tax_liability = tax_impact.get('tax_liability', 0)
                trade_value = abs(trade.get('trade_value', 1))
                if tax_liability > trade_value * 0.1:
                    recommendations.append(f'High tax impact for {symbol}: ${tax_liability:,.0f} ({tax_liability / trade_value:.1%} of trade value)')
            total_sells = sum((1 for trade in rebalance_trades if trade.get('trade_quantity', 0) < 0))
            if total_sells > 5:
                recommendations.append('Consider spreading sales across multiple periods to manage tax impact')
            if not recommendations:
                recommendations.append('Rebalancing plan appears tax-efficient')
            return recommendations
        except (KeyError, ValueError, TypeError) as e:
            logger.error('REBALANCE_RECOMMENDATIONS_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return ['Manual review recommended due to analysis error']

def rebalance_portfolio(ctx) -> None:
    """Enhanced portfolio rebalancing with tax awareness."""
    logger.info('Starting enhanced portfolio rebalancing')
    try:
        settings = get_settings()
        if settings.ENABLE_PORTFOLIO_FEATURES and hasattr(ctx, 'account_equity'):
            tax_rebalancer = TaxAwareRebalancer()
            current_positions = getattr(ctx, 'current_positions', {})
            target_weights = getattr(ctx, 'target_weights', {})
            current_prices = getattr(ctx, 'current_prices', {})
            account_equity = getattr(ctx, 'account_equity', 0)
            if all([current_positions, target_weights, current_prices, account_equity]):
                rebalance_plan = tax_rebalancer.calculate_optimal_rebalance(current_positions, target_weights, current_prices, account_equity)
                logger.info(f"Tax-aware rebalancing complete: drift={rebalance_plan.get('portfolio_drift', 0):.3f}, tax_impact=${rebalance_plan.get('total_tax_impact', 0):,.0f}")
                ctx.rebalance_plan = rebalance_plan
                return
    except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as e:
        logger.warning('ENHANCED_REBALANCE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
    logger.info('Using basic portfolio rebalancing')

def enhanced_maybe_rebalance(ctx) -> None:
    """Enhanced rebalance check with tax optimization and market conditions."""
    global _last_rebalance
    now = safe_utcnow()
    if now - _last_rebalance >= timedelta(minutes=rebalance_interval_min()):
        try:
            portfolio = getattr(ctx, 'portfolio_weights', {})
            if not portfolio:
                init_rebalancer()
                portfolio_first_rebalance(ctx)
                _last_rebalance = now
                return
            current = compute_portfolio_weights(ctx, list(portfolio.keys()))
            drift = max((abs(current.get(s, 0) - portfolio.get(s, 0)) for s in current)) if current else 0.0
            settings = get_settings()
            drift_threshold = settings.portfolio_drift_threshold
            if settings.ENABLE_PORTFOLIO_FEATURES:
                init_rebalancer()
                should_rebalance, reason = _check_portfolio_first_rebalancing(ctx, drift, drift_threshold)
                if should_rebalance:
                    portfolio_first_rebalance(ctx)
                    _last_rebalance = now
                    logger.info(f'PORTFOLIO_FIRST_REBALANCING_EXECUTED | {reason}')
            elif drift > drift_threshold:
                rebalance_portfolio(ctx)
                _last_rebalance = now
        except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as e:
            logger.error('ENHANCED_REBALANCE_LOOP_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            try:
                rebalance_portfolio(ctx)
                _last_rebalance = now
            except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as fallback_error:
                logger.error('FALLBACK_REBALANCE_FAILED', extra={'cause': fallback_error.__class__.__name__, 'detail': str(fallback_error)})

def portfolio_first_rebalance(ctx) -> None:
    """
    Portfolio-first rebalancing that integrates with portfolio optimization.

    This function serves as the primary trading mechanism, implementing
    quarterly tax-optimized rebalancing with intelligent decision making.
    """
    global _rebalancer
    if _rebalancer is None:
        raise RuntimeError('init_rebalancer() must be called before portfolio_first_rebalance')
    try:
        settings = get_settings()
        if not settings.ENABLE_PORTFOLIO_FEATURES:
            logger.info('Portfolio-first not enabled, using standard rebalancing')
            rebalance_portfolio(ctx)
            return
        current_positions = _get_current_positions_for_rebalancing(ctx)
        target_weights = _get_target_weights_for_rebalancing(ctx)
        market_data = _prepare_rebalancing_market_data(ctx)
        if not current_positions or not target_weights or not market_data.get('prices'):
            logger.warning('PORTFOLIO_FIRST_REBALANCING_SKIPPED', extra={'reason': 'missing_rebalance_evidence'})
            ctx.rebalance_plan = {'error': 'missing_rebalance_evidence', 'rebalance_trades': []}
            return
        regime, regime_metrics = _rebalancer.regime_detector.detect_current_regime(market_data)
        dynamic_thresholds = _rebalancer.regime_detector.calculate_dynamic_thresholds(regime, regime_metrics)
        _rebalancer.portfolio_optimizer.improvement_threshold = dynamic_thresholds.minimum_improvement_threshold
        _rebalancer.portfolio_optimizer.rebalance_drift_threshold = dynamic_thresholds.rebalance_drift_threshold
        current_prices = market_data.get('prices', {})
        should_rebalance, rebalance_reason = _rebalancer.portfolio_optimizer.should_trigger_rebalance(
            current_positions, target_weights, current_prices
        )
        if should_rebalance:
            account_equity = 0.0
            missing_prices: list[str] = []
            for symbol, pos in current_positions.items():
                current_price = _finite_positive(current_prices.get(symbol))
                if current_price is None:
                    missing_prices.append(symbol)
                    continue
                account_equity += abs(pos) * current_price
            if missing_prices or account_equity <= 0.0:
                logger.warning(
                    'PORTFOLIO_FIRST_REBALANCING_SKIPPED',
                    extra={'reason': 'missing_price_evidence', 'symbols': missing_prices},
                )
                ctx.rebalance_plan = {'error': 'missing_price_evidence', 'rebalance_trades': []}
                return
            formatted_positions = {}
            for symbol, quantity in current_positions.items():
                if quantity != 0:
                    current_price = _finite_positive(current_prices.get(symbol))
                    if current_price is None:
                        continue
                    formatted_positions[symbol] = {
                        'quantity': quantity,
                    }
            rebalance_plan = _rebalancer.calculate_optimal_rebalance(
                formatted_positions, target_weights, current_prices, account_equity
            )
            logger.info(
                'PORTFOLIO_FIRST_REBALANCING_COMPLETE',
                extra={
                    'reason': rebalance_reason,
                    'market_regime': regime.value,
                    'portfolio_drift': rebalance_plan.get('portfolio_drift', 0),
                    'total_tax_impact': rebalance_plan.get('total_tax_impact', 0),
                    'rebalance_trades': len(rebalance_plan.get('rebalance_trades', [])),
                    'tax_efficiency': rebalance_plan.get('tax_efficiency', 0),
                },
            )
            ctx.rebalance_plan = rebalance_plan
            ctx.last_portfolio_rebalance = datetime.now(UTC)
        else:
            logger.info(f'PORTFOLIO_FIRST_REBALANCING_SKIPPED | {rebalance_reason}')
    except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as e:
        logger.error('PORTFOLIO_FIRST_REBALANCE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
        try:
            rebalance_portfolio(ctx)
        except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as fallback_error:
            logger.error(
                'PORTFOLIO_FIRST_FALLBACK_FAILED',
                extra={'cause': fallback_error.__class__.__name__, 'detail': str(fallback_error)},
            )

def _check_portfolio_first_rebalancing(ctx, current_drift: float, drift_threshold: float) -> tuple:
    """Check if portfolio-first rebalancing should be triggered."""
    try:
        if current_drift > drift_threshold:
            return (True, f'Drift {current_drift:.3f} exceeds threshold {drift_threshold:.3f}')
        last_rebalance = getattr(ctx, 'last_portfolio_rebalance', None)
        if last_rebalance is None:
            return (True, 'No previous rebalancing recorded')
        days_since_rebalance = (datetime.now(UTC) - last_rebalance).days
        if days_since_rebalance >= 90:
            return (True, f'Quarterly rebalance due ({days_since_rebalance} days since last)')
        if _rebalancer is not None:
            try:
                market_data = _prepare_rebalancing_market_data(ctx)
                if market_data:
                    regime, metrics = _rebalancer.regime_detector.detect_current_regime(market_data)
                    if regime.value == 'crisis':
                        return (True, 'Crisis regime detected - protective rebalancing')
                    if (
                        metrics.volatility_regime.value in ['extremely_high', 'extremely_low']
                        and metrics.regime_confidence > 0.8
                    ):
                        return (True, f'Extreme volatility regime: {metrics.volatility_regime.value}')
            except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as e:
                logger.debug(
                    'REGIME_REBALANCE_CHECK_FAILED',
                    extra={'cause': e.__class__.__name__, 'detail': str(e)},
                )
        return (False, f'No rebalancing needed (drift={current_drift:.3f}, days={days_since_rebalance})')
    except (KeyError, ValueError, TypeError, APIError, TimeoutError, ConnectionError, OSError) as e:
        logger.error('CHECK_PORTFOLIO_REBALANCE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
        return (current_drift > drift_threshold, 'Error in analysis, using basic drift check')

def _get_current_positions_for_rebalancing(ctx) -> dict:
    """Get current positions formatted for portfolio rebalancing."""
    try:
        positions = {}
        if hasattr(ctx, 'portfolio_positions'):
            positions = ctx.portfolio_positions.copy()
        elif hasattr(ctx, 'current_positions'):
            positions = ctx.current_positions.copy()
        elif hasattr(ctx, 'positions'):
            positions = ctx.positions.copy()
        filtered_positions = {}
        for symbol, quantity in positions.items():
            try:
                qty = _extract_position_quantity(quantity)
                if abs(qty) > 0.001:
                    filtered_positions[symbol] = qty
            except (ValueError, TypeError):
                continue
        return filtered_positions
    except (APIError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        logger.error('GET_CURRENT_POSITIONS_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
        return {}

def _get_target_weights_for_rebalancing(ctx) -> dict:
    """Get target portfolio weights for rebalancing."""
    try:
        if hasattr(ctx, 'target_weights'):
            target_weights = getattr(ctx, 'target_weights', {})
            if isinstance(target_weights, dict):
                normalized = {}
                for sym, weight in target_weights.items():
                    numeric = float(weight)
                    if math.isfinite(numeric):
                        normalized[str(sym)] = numeric
                return normalized
            return {}
        return {}
    except (KeyError, ValueError, TypeError) as e:
        logger.error('GET_TARGET_WEIGHTS_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
        return {}

def _prepare_rebalancing_market_data(ctx) -> dict:
    """Prepare market data for rebalancing analysis."""
    try:
        market_data: dict[str, dict[str, Any]] = {
            'prices': {},
            'returns': {},
            'volumes': {},
            'correlations': {},
            'volatility': {},
        }
        current_positions = _get_current_positions_for_rebalancing(ctx)
        symbols = set(current_positions.keys())
        symbols.add('SPY')
        for symbol in symbols:
            try:
                fetcher = getattr(ctx, 'data_fetcher', None)
                if not fetcher:
                    continue
                df = fetcher.get_daily_df(ctx, symbol)
                if df is not None and len(df) > 0:
                    if 'close' not in df.columns:
                        logger.debug('REBALANCE_PRICE_EVIDENCE_MISSING', extra={'symbol': symbol})
                        continue
                    current_price = _finite_positive(df['close'].iloc[-1])
                    if current_price is None:
                        logger.debug('REBALANCE_PRICE_EVIDENCE_INVALID', extra={'symbol': symbol})
                        continue
                    market_data['prices'][symbol] = current_price
                    if 'close' in df.columns and len(df) > 1:
                        prices = df['close'].values[-100:]
                        returns = []
                        for i in range(1, len(prices)):
                            if prices[i - 1] > 0:
                                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
                        market_data['returns'][symbol] = returns
                    if 'volume' in df.columns:
                        market_data['volumes'][symbol] = df['volume'].tail(20).mean()
            except (APIError, TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.debug('REBALANCE_DATA_FETCH_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
        return market_data
    except (APIError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        logger.error('PREPARE_MARKET_DATA_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
        return {}

def maybe_rebalance(ctx) -> None:
    """Rebalance when interval has elapsed."""
    global _last_rebalance
    now = safe_utcnow()
    if now - _last_rebalance >= timedelta(minutes=rebalance_interval_min()):
        settings = get_settings()
        portfolio = getattr(ctx, 'portfolio_weights', {})
        if not portfolio:
            rebalance_portfolio(ctx)
            _last_rebalance = now
        else:
            current = compute_portfolio_weights(ctx, list(portfolio.keys()))
            drift = max((abs(current.get(s, 0) - portfolio.get(s, 0)) for s in current)) if current else 0.0
            if drift > settings.portfolio_drift_threshold:
                rebalance_portfolio(ctx)
                _last_rebalance = now

def start_rebalancer(ctx) -> threading.Thread:
    """Run :func:`maybe_rebalance` every minute in a background thread."""

    def loop() -> None:
        while True:
            try:
                maybe_rebalance(ctx)
            except StopIteration:
                logger.debug('Rebalancer loop stopped by test')
                break
            except (ValueError, KeyError, TypeError, OSError, APIError, TimeoutError, ConnectionError) as exc:
                logger.error('REBALANCER_LOOP_ERROR', extra={'cause': exc.__class__.__name__, 'detail': str(exc)})
            settings = get_settings()
            sleep_interval = settings.rebalance_sleep_seconds
            if get_env("PYTEST_CURRENT_TEST", "", cast=str, resolve_aliases=False) or 'test' in str(ctx).lower():
                sleep_interval = 1
            time.sleep(sleep_interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
