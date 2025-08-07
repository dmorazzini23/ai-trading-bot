#!/usr/bin/env python3.12
"""
Symbol-aware cost model for tracking and applying trading costs.

Maintains per-symbol cost parameters (half_spread_bps, slip_k) and applies
them in both backtesting and live position sizing.
"""

import logging
import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SymbolCosts:
    """Per-symbol cost parameters."""
    symbol: str
    half_spread_bps: float  # Half spread in basis points
    slip_k: float          # Slippage coefficient (bps per √volume)
    commission_bps: float = 0.0  # Commission in basis points
    min_commission: float = 0.0  # Minimum commission per trade
    updated_at: datetime = None
    sample_count: int = 0   # Number of trades used to estimate
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
    
    @property
    def total_cost_bps(self) -> float:
        """Base total cost in basis points (spread + commission)."""
        return self.half_spread_bps * 2 + self.commission_bps
    
    def slippage_cost_bps(self, volume_ratio: float = 1.0) -> float:
        """
        Calculate slippage cost based on volume ratio.
        
        Args:
            volume_ratio: Trade volume / typical volume
            
        Returns:
            Slippage cost in basis points
        """
        return self.slip_k * np.sqrt(max(volume_ratio, 0.1))
    
    def total_execution_cost_bps(self, volume_ratio: float = 1.0) -> float:
        """
        Calculate total execution cost including slippage.
        
        Args:
            volume_ratio: Trade volume / typical volume
            
        Returns:
            Total execution cost in basis points
        """
        return self.total_cost_bps + self.slippage_cost_bps(volume_ratio)


class SymbolCostModel:
    """
    Manages per-symbol cost models and persistence.
    
    Tracks execution costs for each symbol and updates models based on
    realized vs expected costs.
    """
    
    def __init__(self, data_path: str = "artifacts/microstructure"):
        """
        Initialize cost model.
        
        Args:
            data_path: Path to store microstructure data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cost data cache
        self._costs: Dict[str, SymbolCosts] = {}
        self._default_costs = SymbolCosts(
            symbol="DEFAULT",
            half_spread_bps=2.0,  # 2 bps half spread
            slip_k=1.5,           # 1.5 bps per √volume
            commission_bps=0.5,   # 0.5 bps commission
            min_commission=1.0    # $1 minimum
        )
        
        # Load existing cost data
        self._load_cost_data()
    
    def _load_cost_data(self) -> None:
        """Load cost data from disk."""
        cost_file = self.data_path / "symbol_costs.json"
        
        if cost_file.exists():
            try:
                with open(cost_file, 'r') as f:
                    data = json.load(f)
                
                for symbol, cost_data in data.items():
                    # Convert datetime string back to datetime
                    if 'updated_at' in cost_data and isinstance(cost_data['updated_at'], str):
                        cost_data['updated_at'] = datetime.fromisoformat(cost_data['updated_at'])
                    
                    self._costs[symbol] = SymbolCosts(**cost_data)
                
                self.logger.info(f"Loaded cost data for {len(self._costs)} symbols")
                
            except Exception as e:
                self.logger.error(f"Failed to load cost data: {e}")
    
    def _save_cost_data(self) -> None:
        """Save cost data to disk."""
        cost_file = self.data_path / "symbol_costs.json"
        
        try:
            # Convert to serializable format
            data = {}
            for symbol, costs in self._costs.items():
                cost_dict = asdict(costs)
                # Convert datetime to string
                if cost_dict['updated_at']:
                    cost_dict['updated_at'] = cost_dict['updated_at'].isoformat()
                data[symbol] = cost_dict
            
            with open(cost_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved cost data for {len(self._costs)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to save cost data: {e}")
    
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
        
        # Return default costs for new symbols
        default = SymbolCosts(
            symbol=symbol,
            half_spread_bps=self._default_costs.half_spread_bps,
            slip_k=self._default_costs.slip_k,
            commission_bps=self._default_costs.commission_bps,
            min_commission=self._default_costs.min_commission
        )
        
        self._costs[symbol] = default
        return default
    
    def update_costs(
        self,
        symbol: str,
        realized_cost_bps: float,
        volume_ratio: float = 1.0,
        learning_rate: float = 0.1
    ) -> None:
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
        
        # Calculate expected cost
        expected_cost_bps = current_costs.total_execution_cost_bps(volume_ratio)
        
        # Update with exponential moving average
        cost_error = realized_cost_bps - expected_cost_bps
        
        # Decompose error into spread and slippage components
        if volume_ratio > 1.0:
            # High volume - likely slippage issue
            slippage_update = cost_error * learning_rate / np.sqrt(volume_ratio)
            new_slip_k = max(0.1, current_costs.slip_k + slippage_update)
        else:
            # Normal volume - likely spread issue
            spread_update = (cost_error / 2) * learning_rate  # Half spread
            new_half_spread = max(0.1, current_costs.half_spread_bps + spread_update)
            current_costs.half_spread_bps = new_half_spread
            new_slip_k = current_costs.slip_k
        
        # Update costs
        updated_costs = SymbolCosts(
            symbol=symbol,
            half_spread_bps=current_costs.half_spread_bps,
            slip_k=new_slip_k,
            commission_bps=current_costs.commission_bps,
            min_commission=current_costs.min_commission,
            updated_at=datetime.now(timezone.utc),
            sample_count=current_costs.sample_count + 1
        )
        
        self._costs[symbol] = updated_costs
        
        self.logger.debug(
            f"Updated costs for {symbol}: spread={updated_costs.half_spread_bps:.2f}bps, "
            f"slip_k={updated_costs.slip_k:.2f}, samples={updated_costs.sample_count}"
        )
        
        # Save updated costs
        self._save_cost_data()
    
    def calculate_position_impact(
        self,
        symbol: str,
        position_value: float,
        volume_ratio: float = 1.0
    ) -> Dict[str, float]:
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
        
        # Apply minimum commission if applicable
        if cost_dollars < costs.min_commission:
            cost_dollars = costs.min_commission
            effective_bps = (cost_dollars / position_value) * 10000
        else:
            effective_bps = total_cost_bps
        
        return {
            'cost_bps': total_cost_bps,
            'cost_dollars': cost_dollars,
            'effective_bps': effective_bps,
            'spread_bps': costs.half_spread_bps * 2,
            'slippage_bps': costs.slippage_cost_bps(volume_ratio),
            'commission_bps': costs.commission_bps
        }
    
    def adjust_position_size(
        self,
        symbol: str,
        target_size: float,
        max_cost_bps: float = 20.0,
        volume_ratio: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
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
            return 0.0, {}
        
        costs = self.get_costs(symbol)
        total_cost_bps = costs.total_execution_cost_bps(volume_ratio)
        
        # If costs are within limit, return original size
        if total_cost_bps <= max_cost_bps:
            cost_info = self.calculate_position_impact(symbol, abs(target_size), volume_ratio)
            return target_size, cost_info
        
        # Scale down size to meet cost constraint
        cost_ratio = max_cost_bps / total_cost_bps
        adjusted_size = target_size * cost_ratio
        
        cost_info = self.calculate_position_impact(symbol, abs(adjusted_size), volume_ratio)
        cost_info['original_size'] = target_size
        cost_info['scaling_factor'] = cost_ratio
        
        self.logger.warning(
            f"Scaled down position for {symbol} by {cost_ratio:.2%} due to high costs "
            f"({total_cost_bps:.1f}bps > {max_cost_bps}bps)"
        )
        
        return adjusted_size, cost_info
    
    def save_daily_snapshot(self, trading_date: Optional[date] = None) -> str:
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
        
        # Convert to DataFrame
        records = []
        for symbol, costs in self._costs.items():
            record = asdict(costs)
            record['date'] = trading_date
            records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(snapshot_file, compression='snappy')
            
            self.logger.info(f"Saved cost snapshot to {snapshot_file}")
            return str(snapshot_file)
        
        return ""
    
    def get_cost_statistics(self) -> Dict[str, float]:
        """Get summary statistics for all symbols."""
        if not self._costs:
            return {}
        
        spreads = [c.half_spread_bps * 2 for c in self._costs.values()]
        slippages = [c.slip_k for c in self._costs.values()]
        
        return {
            'num_symbols': len(self._costs),
            'avg_spread_bps': np.mean(spreads),
            'median_spread_bps': np.median(spreads),
            'avg_slip_k': np.mean(slippages),
            'median_slip_k': np.median(slippages),
            'max_spread_bps': np.max(spreads),
            'min_spread_bps': np.min(spreads)
        }


# Global cost model instance
_global_cost_model: Optional[SymbolCostModel] = None


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


def calculate_execution_cost(
    symbol: str,
    position_value: float,
    volume_ratio: float = 1.0
) -> Dict[str, float]:
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