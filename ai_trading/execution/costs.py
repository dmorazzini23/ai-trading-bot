"""
Symbol-aware cost model for tracking and applying trading costs.

Maintains per-symbol cost parameters (half_spread_bps, slip_k) and applies
them in both backtesting and live position sizing.
"""
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_pandas

pd = load_pandas()
logger = get_logger(__name__)


@lru_cache(maxsize=None)
def _load_numpy() -> ModuleType:
    """Return :mod:`numpy` if available."""
    if find_spec("numpy") is None:
        raise ModuleNotFoundError("numpy is required for cost calculations")
    import numpy as np  # type: ignore

    return np

@dataclass
class SymbolCosts:
    """Per-symbol cost parameters."""
    symbol: str
    half_spread_bps: float
    slip_k: float
    commission_bps: float = 0.0
    min_commission: float = 0.0
    borrow_fee_bps: float = 0.0
    overnight_bps: float = 0.0
    locate_required: bool = False
    updated_at: datetime = None
    sample_count: int = 0

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now(UTC)

    @property
    def total_cost_bps(self) -> float:
        """Base total cost in basis points (spread + commission)."""
        return self.half_spread_bps * 2 + self.commission_bps

    def slippage_cost_bps(self, volume_ratio: float=1.0) -> float:
        """
        Calculate slippage cost based on volume ratio.

        Args:
            volume_ratio: Trade volume / typical volume

        Returns:
            Slippage cost in basis points
        """
        np = _load_numpy()
        return self.slip_k * np.sqrt(max(volume_ratio, 0.1))

    def total_execution_cost_bps(self, volume_ratio: float=1.0) -> float:
        """
        Calculate total execution cost including slippage.

        Args:
            volume_ratio: Trade volume / typical volume

        Returns:
            Total execution cost in basis points
        """
        return self.total_cost_bps + self.slippage_cost_bps(volume_ratio)

    def borrow_cost_bps(self, days_held: float=1.0) -> float:
        """
        Calculate borrow cost for short positions.

        Args:
            days_held: Number of days position is held

        Returns:
            Borrow cost in basis points
        """
        return self.borrow_fee_bps * days_held

    def overnight_cost_bps(self, days_held: float=1.0) -> float:
        """
        Calculate overnight carry cost.

        Args:
            days_held: Number of days position is held

        Returns:
            Overnight cost in basis points
        """
        return self.overnight_bps * days_held

    def total_holding_cost_bps(self, days_held: float=1.0, is_short: bool=False) -> float:
        """
        Calculate total cost of holding position.

        Args:
            days_held: Number of days position is held
            is_short: Whether position is short

        Returns:
            Total holding cost in basis points
        """
        cost = self.overnight_cost_bps(days_held)
        if is_short:
            cost += self.borrow_cost_bps(days_held)
        return cost

class SymbolCostModel:
    """
    Manages per-symbol cost models and persistence.

    Tracks execution costs for each symbol and updates models based on
    realized vs expected costs.
    """

    def __init__(self, data_path: str='artifacts/microstructure'):
        """
        Initialize cost model.

        Args:
            data_path: Path to store microstructure data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self._costs: dict[str, SymbolCosts] = {}
        self._default_costs = SymbolCosts(symbol='DEFAULT', half_spread_bps=2.0, slip_k=1.5, commission_bps=0.5, min_commission=1.0, borrow_fee_bps=10.0, overnight_bps=0.5, locate_required=False)
        self._load_cost_data()

    def _load_cost_data(self) -> None:
        """Load cost data from disk."""
        cost_file = self.data_path / 'symbol_costs.json'
        if cost_file.exists():
            try:
                with open(cost_file) as f:
                    data = json.load(f)
                for symbol, cost_data in data.items():
                    if 'updated_at' in cost_data and isinstance(cost_data['updated_at'], str):
                        cost_data['updated_at'] = datetime.fromisoformat(cost_data['updated_at'])
                    self._costs[symbol] = SymbolCosts(**cost_data)
                self.logger.info(f'Loaded cost data for {len(self._costs)} symbols')
            except (ValueError, TypeError) as e:
                self.logger.error(f'Failed to load cost data: {e}')

    def _save_cost_data(self) -> None:
        """Save cost data to disk."""
        cost_file = self.data_path / 'symbol_costs.json'
        try:
            data = {}
            for symbol, costs in self._costs.items():
                cost_dict = asdict(costs)
                if cost_dict['updated_at']:
                    cost_dict['updated_at'] = cost_dict['updated_at'].isoformat()
                data[symbol] = cost_dict
            with open(cost_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f'Saved cost data for {len(self._costs)} symbols')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Failed to save cost data: {e}')

    def get_costs(self, symbol: str) -> SymbolCosts:
        """
        Get cost parameters for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            SymbolCosts for the symbol
        """
        symbol = symbol.upper()
        if symbol in self._costs:
            return self._costs[symbol]
        default = SymbolCosts(symbol=symbol, half_spread_bps=self._default_costs.half_spread_bps, slip_k=self._default_costs.slip_k, commission_bps=self._default_costs.commission_bps, min_commission=self._default_costs.min_commission, borrow_fee_bps=self._default_costs.borrow_fee_bps, overnight_bps=self._default_costs.overnight_bps, locate_required=self._default_costs.locate_required)
        self._costs[symbol] = default
        return default

    def update_costs(self, symbol: str, realized_cost_bps: float, volume_ratio: float=1.0, learning_rate: float=0.1) -> None:
        """
        Update cost model based on realized execution cost.

        Args:
            symbol: Trading symbol
            realized_cost_bps: Actual execution cost in bps
            volume_ratio: Trade volume / typical volume
            learning_rate: Learning rate for cost updates
        """
        symbol = symbol.upper()
        current_costs = self.get_costs(symbol)
        expected_cost_bps = current_costs.total_execution_cost_bps(volume_ratio)
        cost_error = realized_cost_bps - expected_cost_bps
        if volume_ratio > 1.0:
            np = _load_numpy()
            slippage_update = cost_error * learning_rate / np.sqrt(volume_ratio)
            new_slip_k = max(0.1, current_costs.slip_k + slippage_update)
        else:
            spread_update = cost_error / 2 * learning_rate
            new_half_spread = max(0.1, current_costs.half_spread_bps + spread_update)
            current_costs.half_spread_bps = new_half_spread
            new_slip_k = current_costs.slip_k
        updated_costs = SymbolCosts(symbol=symbol, half_spread_bps=current_costs.half_spread_bps, slip_k=new_slip_k, commission_bps=current_costs.commission_bps, min_commission=current_costs.min_commission, borrow_fee_bps=current_costs.borrow_fee_bps, overnight_bps=current_costs.overnight_bps, locate_required=current_costs.locate_required, updated_at=datetime.now(UTC), sample_count=current_costs.sample_count + 1)
        self._costs[symbol] = updated_costs
        self.logger.debug(f'Updated costs for {symbol}: spread={updated_costs.half_spread_bps:.2f}bps, slip_k={updated_costs.slip_k:.2f}, samples={updated_costs.sample_count}')
        self._save_cost_data()

    def calculate_position_impact(self, symbol: str, position_value: float, volume_ratio: float=1.0) -> dict[str, float]:
        """
        Calculate cost impact for a position.

        Args:
            symbol: Trading symbol
            position_value: Position value in dollars
            volume_ratio: Trade volume / typical volume

        Returns:
            Dict with cost breakdown
        """
        costs = self.get_costs(symbol)
        total_cost_bps = costs.total_execution_cost_bps(volume_ratio)
        cost_dollars = position_value * (total_cost_bps / 10000)
        if cost_dollars < costs.min_commission:
            cost_dollars = costs.min_commission
            effective_bps = cost_dollars / position_value * 10000
        else:
            effective_bps = total_cost_bps
        return {'cost_bps': total_cost_bps, 'cost_dollars': cost_dollars, 'effective_bps': effective_bps, 'spread_bps': costs.half_spread_bps * 2, 'slippage_bps': costs.slippage_cost_bps(volume_ratio), 'commission_bps': costs.commission_bps}

    def adjust_position_size(self, symbol: str, target_size: float, max_cost_bps: float=20.0, volume_ratio: float=1.0) -> tuple[float, dict[str, float]]:
        """
        Adjust position size based on cost constraints.

        Args:
            symbol: Trading symbol
            target_size: Target position size
            max_cost_bps: Maximum acceptable cost in bps
            volume_ratio: Expected volume ratio

        Returns:
            Tuple of (adjusted_size, cost_info)
        """
        if target_size == 0:
            return (0.0, {})
        costs = self.get_costs(symbol)
        total_cost_bps = costs.total_execution_cost_bps(volume_ratio)
        if total_cost_bps <= max_cost_bps:
            cost_info = self.calculate_position_impact(symbol, abs(target_size), volume_ratio)
            return (target_size, cost_info)
        cost_ratio = max_cost_bps / total_cost_bps
        adjusted_size = target_size * cost_ratio
        cost_info = self.calculate_position_impact(symbol, abs(adjusted_size), volume_ratio)
        cost_info['original_size'] = target_size
        cost_info['scaling_factor'] = cost_ratio
        self.logger.warning(f'Scaled down position for {symbol} by {cost_ratio:.2%} due to high costs ({total_cost_bps:.1f}bps > {max_cost_bps}bps)')
        return (adjusted_size, cost_info)

    def save_daily_snapshot(self, trading_date: date | None=None) -> str:
        """
        Save daily snapshot of cost parameters.

        Args:
            trading_date: Date for snapshot (defaults to today)

        Returns:
            Path to saved snapshot
        """
        if trading_date is None:
            trading_date = date.today()
        snapshot_file = self.data_path / f"{trading_date.strftime('%Y%m%d')}.parquet"
        records = []
        for _symbol, costs in self._costs.items():
            record = asdict(costs)
            record['date'] = trading_date
            records.append(record)
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(snapshot_file, compression='snappy')
            self.logger.info(f'Saved cost snapshot to {snapshot_file}')
            return str(snapshot_file)
        return ''

    def get_cost_statistics(self) -> dict[str, float]:
        """Get summary statistics for all symbols."""
        if not self._costs:
            return {}
        np = _load_numpy()
        spreads = [c.half_spread_bps * 2 for c in self._costs.values()]
        slippages = [c.slip_k for c in self._costs.values()]
        return {
            'num_symbols': len(self._costs),
            'avg_spread_bps': np.mean(spreads),
            'median_spread_bps': np.median(spreads),
            'avg_slip_k': np.mean(slippages),
            'median_slip_k': np.median(slippages),
            'max_spread_bps': np.max(spreads),
            'min_spread_bps': np.min(spreads),
        }

    def check_short_availability(self, symbol: str) -> tuple[bool, str]:
        """
        Check if symbol is available for short selling.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (available, reason)
        """
        costs = self.get_costs(symbol)
        if costs.locate_required:
            if costs.borrow_fee_bps > 50.0:
                return (False, 'Hard to borrow - high fee')
            elif costs.borrow_fee_bps > 20.0:
                return (True, 'Available but expensive')
        return (True, 'Available')

    def calculate_holding_cost(self, symbol: str, position_value: float, days_held: float=1.0, is_short: bool=False) -> dict[str, float]:
        """
        Calculate cost of holding position overnight.

        Args:
            symbol: Trading symbol
            position_value: Position value in dollars
            days_held: Number of days position is held
            is_short: Whether position is short

        Returns:
            Dict with holding cost breakdown
        """
        costs = self.get_costs(symbol)
        overnight_cost_bps = costs.overnight_cost_bps(days_held)
        overnight_cost_dollars = position_value * (overnight_cost_bps / 10000)
        borrow_cost_bps = 0.0
        borrow_cost_dollars = 0.0
        if is_short:
            borrow_cost_bps = costs.borrow_cost_bps(days_held)
            borrow_cost_dollars = position_value * (borrow_cost_bps / 10000)
        total_cost_bps = overnight_cost_bps + borrow_cost_bps
        total_cost_dollars = overnight_cost_dollars + borrow_cost_dollars
        return {'overnight_cost_bps': overnight_cost_bps, 'overnight_cost_dollars': overnight_cost_dollars, 'borrow_cost_bps': borrow_cost_bps, 'borrow_cost_dollars': borrow_cost_dollars, 'total_holding_cost_bps': total_cost_bps, 'total_holding_cost_dollars': total_cost_dollars, 'days_held': days_held, 'is_short': is_short}

    def adjust_for_holding_costs(self, symbol: str, target_size: float, expected_holding_days: float=1.0, max_holding_cost_bps: float=5.0, is_short: bool=False) -> tuple[float, dict[str, float]]:
        """
        Adjust position size considering holding costs.

        Args:
            symbol: Trading symbol
            target_size: Target position size
            expected_holding_days: Expected days to hold position
            max_holding_cost_bps: Maximum acceptable holding cost
            is_short: Whether position is short

        Returns:
            Tuple of (adjusted_size, cost_info)
        """
        if target_size == 0:
            return (0.0, {})
        costs = self.get_costs(symbol)
        holding_cost_bps = costs.total_holding_cost_bps(expected_holding_days, is_short)
        if is_short:
            available, reason = self.check_short_availability(symbol)
            if not available:
                self.logger.warning(f'Cannot short {symbol}: {reason}')
                return (0.0, {'rejected': True, 'reason': reason})
        if holding_cost_bps <= max_holding_cost_bps:
            cost_info = self.calculate_holding_cost(symbol, abs(target_size), expected_holding_days, is_short)
            return (target_size, cost_info)
        cost_ratio = max_holding_cost_bps / holding_cost_bps
        adjusted_size = target_size * cost_ratio
        cost_info = self.calculate_holding_cost(symbol, abs(adjusted_size), expected_holding_days, is_short)
        cost_info['original_size'] = target_size
        cost_info['scaling_factor'] = cost_ratio
        self.logger.warning(f"Scaled down {('short' if is_short else 'long')} position for {symbol} by {cost_ratio:.2%} due to high holding costs ({holding_cost_bps:.1f}bps > {max_holding_cost_bps}bps)")
        return (adjusted_size, cost_info)
_global_cost_model: SymbolCostModel | None = None

def get_cost_model() -> SymbolCostModel:
    """Get or create global cost model instance."""
    global _global_cost_model
    if _global_cost_model is None:
        _global_cost_model = SymbolCostModel()
    return _global_cost_model

def get_symbol_costs(symbol: str) -> SymbolCosts:
    """
    Convenience function to get symbol costs.

    Args:
        symbol: Trading symbol

    Returns:
        SymbolCosts for the symbol
    """
    model = get_cost_model()
    return model.get_costs(symbol)

def calculate_execution_cost(symbol: str, position_value: float, volume_ratio: float=1.0) -> dict[str, float]:
    """
    Convenience function to calculate execution costs.

    Args:
        symbol: Trading symbol
        position_value: Position value in dollars
        volume_ratio: Trade volume / typical volume

    Returns:
        Dict with cost breakdown
    """
    model = get_cost_model()
    return model.calculate_position_impact(symbol, position_value, volume_ratio)