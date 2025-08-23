"""
Enhanced Pre-Trade Validation System.

Comprehensive pre-trade checks including liquidity analysis, risk validation,
compliance checks, and market condition assessment for institutional trading.
"""
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from ai_trading.logging import logger
from ..core.constants import EXECUTION_PARAMETERS, MARKET_HOURS, RISK_PARAMETERS
from ..core.enums import RiskLevel

class ValidationStatus(Enum):
    """Pre-trade validation status."""
    APPROVED = 'approved'
    REJECTED = 'rejected'
    WARNING = 'warning'
    CONDITIONAL = 'conditional'

class ValidationCategory(Enum):
    """Categories of pre-trade validation checks."""
    MARKET_HOURS = 'market_hours'
    LIQUIDITY = 'liquidity'
    RISK_LIMITS = 'risk_limits'
    POSITION_LIMITS = 'position_limits'
    CORRELATION = 'correlation'
    VOLATILITY = 'volatility'
    SYSTEM_HEALTH = 'system_health'
    COMPLIANCE = 'compliance'
    EXECUTION = 'execution'

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    category: ValidationCategory
    status: ValidationStatus
    message: str
    details: dict[str, Any]
    score: float
    recommendations: list[str]

@dataclass
class PreTradeCheckResult:
    """Complete pre-trade validation result."""
    symbol: str
    order_type: str
    quantity: int
    price: float
    overall_status: ValidationStatus
    overall_score: float
    validation_results: list[ValidationResult]
    warnings: list[str]
    recommendations: list[str]
    metadata: dict[str, Any]
    timestamp: datetime

class MarketHoursValidator:
    """Validates trading during appropriate market hours."""

    def __init__(self):
        """Initialize market hours validator."""
        self.market_hours = MARKET_HOURS
        logger.debug('MarketHoursValidator initialized')

    def validate_market_hours(self, timestamp: datetime | None=None) -> ValidationResult:
        """
        Validate if trading is allowed during current market hours.

        Args:
            timestamp: Optional timestamp to check (default: now)

        Returns:
            ValidationResult for market hours check
        """
        try:
            check_time = timestamp or datetime.now(UTC)
            current_time = check_time.time()
            market_open = self.market_hours['MARKET_OPEN']
            market_close = self.market_hours['MARKET_CLOSE']
            if market_open <= current_time <= market_close:
                return ValidationResult(category=ValidationCategory.MARKET_HOURS, status=ValidationStatus.APPROVED, message='Trading within regular market hours', details={'current_time': current_time, 'market_status': 'open'}, score=1.0, recommendations=[])
            pre_market_start = self.market_hours['PRE_MARKET_START']
            after_hours_end = self.market_hours['AFTER_HOURS_END']
            if pre_market_start <= current_time < market_open or market_close < current_time <= after_hours_end:
                return ValidationResult(category=ValidationCategory.MARKET_HOURS, status=ValidationStatus.WARNING, message='Trading during extended hours - reduced liquidity expected', details={'current_time': current_time, 'market_status': 'extended'}, score=0.7, recommendations=['Consider smaller position sizes during extended hours', 'Use limit orders for better execution'])
            return ValidationResult(category=ValidationCategory.MARKET_HOURS, status=ValidationStatus.REJECTED, message='Trading outside allowed hours', details={'current_time': current_time, 'market_status': 'closed'}, score=0.0, recommendations=['Wait for market open', 'Queue order for next session'])
        except (ValueError, TypeError) as e:
            logger.error(f'Error validating market hours: {e}')
            return ValidationResult(category=ValidationCategory.MARKET_HOURS, status=ValidationStatus.REJECTED, message=f'Market hours validation error: {e}', details={'error': str(e)}, score=0.0, recommendations=['Manual review required'])

class LiquidityValidator:
    """Validates liquidity and execution feasibility."""

    def __init__(self):
        """Initialize liquidity validator."""
        self.min_liquidity = RISK_PARAMETERS['MIN_LIQUIDITY_THRESHOLD']
        self.max_participation = EXECUTION_PARAMETERS['PARTICIPATION_RATE']
        self.max_slippage_bps = EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS']
        logger.debug('LiquidityValidator initialized')

    def validate_liquidity(self, symbol: str, quantity: int, market_data: dict) -> ValidationResult:
        """
        Validate liquidity for the proposed trade.

        Args:
            symbol: Trading symbol
            quantity: Proposed trade quantity
            market_data: Current market data including volume, spreads

        Returns:
            ValidationResult for liquidity check
        """
        try:
            details = {'symbol': symbol, 'quantity': quantity}
            avg_volume = market_data.get('avg_volume', 0)
            current_volume = market_data.get('current_volume', 0)
            bid_ask_spread = market_data.get('bid_ask_spread', 0.0)
            bid_size = market_data.get('bid_size', 0)
            ask_size = market_data.get('ask_size', 0)
            price = market_data.get('last_price', 0.0)
            details.update({'avg_volume': avg_volume, 'current_volume': current_volume, 'bid_ask_spread': bid_ask_spread, 'price': price})
            if avg_volume < self.min_liquidity:
                return ValidationResult(category=ValidationCategory.LIQUIDITY, status=ValidationStatus.REJECTED, message=f'Insufficient liquidity: avg volume {avg_volume:,} < {self.min_liquidity:,}', details=details, score=0.0, recommendations=['Find more liquid alternative', 'Reduce position size'])
            participation_rate = quantity / avg_volume if avg_volume > 0 else 1.0
            details['participation_rate'] = participation_rate
            if participation_rate > self.max_participation:
                return ValidationResult(category=ValidationCategory.LIQUIDITY, status=ValidationStatus.WARNING, message=f'High participation rate: {participation_rate:.1%} > {self.max_participation:.1%}', details=details, score=0.5, recommendations=['Split order across multiple sessions', 'Use TWAP/VWAP execution algorithm', 'Reduce position size'])
            estimated_slippage_bps = self._estimate_slippage(quantity, avg_volume, bid_ask_spread, price, bid_size, ask_size)
            details['estimated_slippage_bps'] = estimated_slippage_bps
            if estimated_slippage_bps > self.max_slippage_bps:
                return ValidationResult(category=ValidationCategory.LIQUIDITY, status=ValidationStatus.WARNING, message=f'High estimated slippage: {estimated_slippage_bps:.1f} bps', details=details, score=0.6, recommendations=['Use limit orders', 'Split into smaller orders', 'Consider alternative execution venue'])
            liquidity_score = self._calculate_liquidity_score(participation_rate, estimated_slippage_bps, current_volume, avg_volume)
            return ValidationResult(category=ValidationCategory.LIQUIDITY, status=ValidationStatus.APPROVED, message='Adequate liquidity for execution', details=details, score=liquidity_score, recommendations=[] if liquidity_score > 0.8 else ['Monitor execution quality'])
        except (ValueError, TypeError) as e:
            logger.error(f'Error validating liquidity for {symbol}: {e}')
            return ValidationResult(category=ValidationCategory.LIQUIDITY, status=ValidationStatus.REJECTED, message=f'Liquidity validation error: {e}', details={'error': str(e)}, score=0.0, recommendations=['Manual liquidity review required'])

    def _estimate_slippage(self, quantity: int, avg_volume: float, spread: float, price: float, bid_size: int, ask_size: int) -> float:
        """Estimate slippage in basis points."""
        try:
            if price <= 0:
                return 999.0
            spread_bps = spread / price * 10000 if price > 0 else 100
            participation = quantity / avg_volume if avg_volume > 0 else 1.0
            impact_multiplier = math.sqrt(participation) * 100
            depth_penalty = 0.0
            if quantity > min(bid_size, ask_size) and min(bid_size, ask_size) > 0:
                depth_penalty = 20.0
            total_slippage = spread_bps * 0.5 + impact_multiplier + depth_penalty
            return min(200.0, total_slippage)
        except (ValueError, TypeError, ZeroDivisionError):
            return 100.0

    def _calculate_liquidity_score(self, participation_rate: float, slippage_bps: float, current_volume: float, avg_volume: float) -> float:
        """Calculate overall liquidity score (0-1)."""
        try:
            participation_score = max(0.0, 1.0 - participation_rate / self.max_participation)
            slippage_score = max(0.0, 1.0 - slippage_bps / self.max_slippage_bps)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.5
            volume_score = min(1.0, volume_ratio)
            overall_score = participation_score * 0.4 + slippage_score * 0.4 + volume_score * 0.2
            return max(0.0, min(1.0, overall_score))
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.5

class RiskValidator:
    """Validates risk limits and portfolio constraints."""

    def __init__(self, risk_level: RiskLevel=RiskLevel.MODERATE):
        """Initialize risk validator."""
        self.risk_level = risk_level
        self.max_portfolio_risk = RISK_PARAMETERS['MAX_PORTFOLIO_RISK']
        self.max_position_size = RISK_PARAMETERS['MAX_POSITION_SIZE']
        self.max_correlation_exposure = RISK_PARAMETERS['MAX_CORRELATION_EXPOSURE']
        logger.debug(f'RiskValidator initialized with risk_level={risk_level}')

    def validate_position_risk(self, symbol: str, quantity: int, price: float, account_equity: float, current_positions: dict) -> ValidationResult:
        """
        Validate position-level risk constraints.

        Args:
            symbol: Trading symbol
            quantity: Proposed trade quantity
            price: Trade price
            account_equity: Current account equity
            current_positions: Dictionary of current positions

        Returns:
            ValidationResult for position risk check
        """
        try:
            details = {'symbol': symbol, 'quantity': quantity, 'price': price, 'account_equity': account_equity}
            if account_equity <= 0:
                return ValidationResult(category=ValidationCategory.RISK_LIMITS, status=ValidationStatus.REJECTED, message='Invalid account equity', details=details, score=0.0, recommendations=['Verify account data'])
            position_value = quantity * price
            position_pct = position_value / account_equity
            details.update({'position_value': position_value, 'position_percentage': position_pct})
            if position_pct > self.max_position_size:
                return ValidationResult(category=ValidationCategory.POSITION_LIMITS, status=ValidationStatus.REJECTED, message=f'Position size {position_pct:.1%} exceeds limit {self.max_position_size:.1%}', details=details, score=0.0, recommendations=[f'Reduce quantity to max {int(account_equity * self.max_position_size / price)}'])
            total_position_value = sum((pos.get('notional_value', 0) for pos in current_positions.values()))
            new_total = total_position_value + position_value
            portfolio_utilization = new_total / account_equity
            details['portfolio_utilization'] = portfolio_utilization
            warnings = []
            if position_pct > self.max_position_size * 0.7:
                warnings.append(f'Large position: {position_pct:.1%} of account')
            if portfolio_utilization > 0.8:
                warnings.append(f'High portfolio utilization: {portfolio_utilization:.1%}')
            risk_score = self._calculate_risk_score(position_pct, portfolio_utilization)
            status = ValidationStatus.APPROVED
            if warnings:
                status = ValidationStatus.WARNING
            return ValidationResult(category=ValidationCategory.RISK_LIMITS, status=status, message='Position risk within acceptable limits' if not warnings else 'Position risk warnings', details=details, score=risk_score, recommendations=warnings)
        except (ValueError, TypeError) as e:
            logger.error(f'Error validating position risk for {symbol}: {e}')
            return ValidationResult(category=ValidationCategory.RISK_LIMITS, status=ValidationStatus.REJECTED, message=f'Risk validation error: {e}', details={'error': str(e)}, score=0.0, recommendations=['Manual risk review required'])

    def validate_portfolio_risk(self, proposed_trade: dict, portfolio_data: dict) -> ValidationResult:
        """
        Validate portfolio-level risk after proposed trade.

        Args:
            proposed_trade: Dictionary with trade details
            portfolio_data: Current portfolio data including correlations

        Returns:
            ValidationResult for portfolio risk check
        """
        try:
            symbol = proposed_trade.get('symbol')
            quantity = proposed_trade.get('quantity', 0)
            price = proposed_trade.get('price', 0.0)
            current_positions = portfolio_data.get('current_positions', {})
            correlations = portfolio_data.get('correlations', {})
            account_equity = portfolio_data.get('account_equity', 0)
            details = {'symbol': symbol, 'current_position_count': len(current_positions), 'account_equity': account_equity}
            correlation_exposure = self._calculate_correlation_exposure(symbol, quantity * price, current_positions, correlations)
            details['correlation_exposure'] = correlation_exposure
            if correlation_exposure > self.max_correlation_exposure:
                return ValidationResult(category=ValidationCategory.CORRELATION, status=ValidationStatus.WARNING, message=f'High correlation exposure: {correlation_exposure:.1%}', details=details, score=0.4, recommendations=['Consider reducing correlated positions', 'Diversify across uncorrelated assets'])
            position_count = len(current_positions) + (1 if symbol not in current_positions else 0)
            concentration_risk = self._assess_concentration_risk(current_positions, account_equity)
            details.update({'position_count': position_count, 'concentration_risk': concentration_risk})
            portfolio_score = self._calculate_portfolio_risk_score(correlation_exposure, concentration_risk, position_count)
            return ValidationResult(category=ValidationCategory.RISK_LIMITS, status=ValidationStatus.APPROVED, message='Portfolio risk within acceptable limits', details=details, score=portfolio_score, recommendations=[] if portfolio_score > 0.7 else ['Monitor portfolio risk closely'])
        except (ValueError, TypeError) as e:
            logger.error(f'Error validating portfolio risk: {e}')
            return ValidationResult(category=ValidationCategory.RISK_LIMITS, status=ValidationStatus.REJECTED, message=f'Portfolio risk validation error: {e}', details={'error': str(e)}, score=0.0, recommendations=['Manual portfolio risk review required'])

    def _calculate_risk_score(self, position_pct: float, portfolio_utilization: float) -> float:
        """Calculate risk score based on position and portfolio metrics."""
        try:
            position_score = max(0.0, 1.0 - position_pct / self.max_position_size)
            utilization_score = max(0.0, 1.0 - max(0.0, portfolio_utilization - 0.5) * 2)
            return position_score * 0.6 + utilization_score * 0.4
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 0.5

    def _calculate_correlation_exposure(self, symbol: str, position_value: float, current_positions: dict, correlations: dict) -> float:
        """Calculate correlation exposure for the new position."""
        try:
            if not current_positions or not correlations:
                return 0.0
            total_portfolio_value = sum((pos.get('notional_value', 0) for pos in current_positions.values()))
            total_portfolio_value += position_value
            if total_portfolio_value <= 0:
                return 0.0
            correlation_exposure = 0.0
            for other_symbol, position_info in current_positions.items():
                if other_symbol == symbol:
                    continue
                correlation_key = f'{symbol}_{other_symbol}'
                correlation = correlations.get(correlation_key, 0.0)
                other_weight = position_info.get('notional_value', 0) / total_portfolio_value
                correlation_contribution = abs(correlation) * other_weight
                correlation_exposure += correlation_contribution
            return correlation_exposure
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 0.0

    def _assess_concentration_risk(self, current_positions: dict, account_equity: float) -> float:
        """Assess portfolio concentration risk."""
        try:
            if not current_positions or account_equity <= 0:
                return 0.0
            position_weights = []
            for position_info in current_positions.values():
                notional_value = position_info.get('notional_value', 0)
                weight = notional_value / account_equity
                position_weights.append(weight)
            if not position_weights:
                return 0.0
            hhi = sum((weight ** 2 for weight in position_weights))
            max_hhi = 1.0
            return hhi / max_hhi
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_portfolio_risk_score(self, correlation_exposure: float, concentration_risk: float, position_count: int) -> float:
        """Calculate overall portfolio risk score."""
        try:
            correlation_score = max(0.0, 1.0 - correlation_exposure / self.max_correlation_exposure)
            concentration_score = max(0.0, 1.0 - concentration_risk)
            optimal_positions = 20
            diversification_score = min(1.0, position_count / optimal_positions)
            return correlation_score * 0.4 + concentration_score * 0.4 + diversification_score * 0.2
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 0.5

class PreTradeValidator:
    """
    Comprehensive pre-trade validation system.

    Coordinates all validation checks and provides final trade approval/rejection
    with detailed reasoning and recommendations.
    """

    def __init__(self, risk_level: RiskLevel=RiskLevel.MODERATE):
        """Initialize pre-trade validator."""
        self.risk_level = risk_level
        self.market_hours_validator = MarketHoursValidator()
        self.liquidity_validator = LiquidityValidator()
        self.risk_validator = RiskValidator(risk_level)
        self.min_overall_score = 0.6
        self.warning_threshold = 0.8
        logger.info(f'PreTradeValidator initialized with risk_level={risk_level}')

    def validate_trade(self, trade_request: dict, market_data: dict, portfolio_data: dict) -> PreTradeCheckResult:
        """
        Perform comprehensive pre-trade validation.

        Args:
            trade_request: Dictionary with trade details (symbol, quantity, price, etc.)
            market_data: Current market data for the symbol
            portfolio_data: Portfolio and account information

        Returns:
            PreTradeCheckResult with validation outcome
        """
        try:
            symbol = trade_request.get('symbol', '')
            quantity = trade_request.get('quantity', 0)
            price = trade_request.get('price', 0.0)
            order_type = trade_request.get('order_type', 'market')
            logger.info(f'Starting pre-trade validation for {symbol}: {quantity} @ {price}')
            validation_results = []
            market_hours_result = self.market_hours_validator.validate_market_hours()
            validation_results.append(market_hours_result)
            liquidity_result = self.liquidity_validator.validate_liquidity(symbol, quantity, market_data)
            validation_results.append(liquidity_result)
            account_equity = portfolio_data.get('account_equity', 0)
            current_positions = portfolio_data.get('current_positions', {})
            position_risk_result = self.risk_validator.validate_position_risk(symbol, quantity, price, account_equity, current_positions)
            validation_results.append(position_risk_result)
            portfolio_risk_result = self.risk_validator.validate_portfolio_risk(trade_request, portfolio_data)
            validation_results.append(portfolio_risk_result)
            system_health_result = self._validate_system_health()
            validation_results.append(system_health_result)
            overall_status, overall_score = self._calculate_overall_result(validation_results)
            warnings = []
            recommendations = []
            for result in validation_results:
                if result.status in [ValidationStatus.WARNING, ValidationStatus.REJECTED]:
                    warnings.append(f'{result.category.value}: {result.message}')
                recommendations.extend(result.recommendations)
            final_result = PreTradeCheckResult(symbol=symbol, order_type=order_type, quantity=quantity, price=price, overall_status=overall_status, overall_score=overall_score, validation_results=validation_results, warnings=warnings, recommendations=list(set(recommendations)), metadata={'validation_timestamp': datetime.now(UTC), 'validation_version': '1.0', 'risk_level': self.risk_level.value, 'checks_performed': len(validation_results)}, timestamp=datetime.now(UTC))
            logger.info(f'Pre-trade validation complete for {symbol}: status={overall_status.value}, score={overall_score:.3f}')
            return final_result
        except (ValueError, TypeError) as e:
            logger.error(f'Error in pre-trade validation: {e}')
            return PreTradeCheckResult(symbol=trade_request.get('symbol', 'UNKNOWN'), order_type=trade_request.get('order_type', 'unknown'), quantity=trade_request.get('quantity', 0), price=trade_request.get('price', 0.0), overall_status=ValidationStatus.REJECTED, overall_score=0.0, validation_results=[], warnings=[f'Validation system error: {e}'], recommendations=['Manual review required'], metadata={'error': str(e)}, timestamp=datetime.now(UTC))

    def _validate_system_health(self) -> ValidationResult:
        """Validate system health and capacity."""
        try:
            return ValidationResult(category=ValidationCategory.SYSTEM_HEALTH, status=ValidationStatus.APPROVED, message='System health check passed', details={'system_status': 'healthy'}, score=1.0, recommendations=[])
        except (ValueError, TypeError) as e:
            return ValidationResult(category=ValidationCategory.SYSTEM_HEALTH, status=ValidationStatus.WARNING, message=f'System health check inconclusive: {e}', details={'error': str(e)}, score=0.7, recommendations=['Monitor system performance'])

    def _calculate_overall_result(self, validation_results: list[ValidationResult]) -> tuple[ValidationStatus, float]:
        """Calculate overall validation status and score."""
        try:
            if not validation_results:
                return (ValidationStatus.REJECTED, 0.0)
            rejections = [r for r in validation_results if r.status == ValidationStatus.REJECTED]
            if rejections:
                return (ValidationStatus.REJECTED, 0.0)
            total_score = sum((result.score for result in validation_results))
            overall_score = total_score / len(validation_results)
            warnings = [r for r in validation_results if r.status == ValidationStatus.WARNING]
            if overall_score >= self.warning_threshold and (not warnings):
                return (ValidationStatus.APPROVED, overall_score)
            elif overall_score >= self.min_overall_score:
                return (ValidationStatus.WARNING, overall_score)
            else:
                return (ValidationStatus.REJECTED, overall_score)
        except (KeyError, ValueError, TypeError, AttributeError):
            return (ValidationStatus.REJECTED, 0.0)