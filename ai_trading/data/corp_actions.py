"""
Corporate actions adjustment pipeline for unified price/volume adjustments.

Provides single source of truth for corporate action adjustments used by
features, labels, and execution sizing to ensure consistency.
"""

import logging
import json
from datetime import date
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    """Single corporate action event."""
    symbol: str
    ex_date: date  # Ex-dividend/split date
    action_type: str  # 'split', 'dividend', 'spin_off', 'merger'
    ratio: float  # Split ratio (e.g., 2.0 for 2:1 split)
    dividend_amount: float = 0.0  # Dividend per share
    cash_amount: float = 0.0  # Cash consideration
    description: str = ""
    source: str = ""  # Data source
    
    @property
    def price_adjustment_factor(self) -> float:
        """
        Calculate price adjustment factor for historical data.
        
        Returns:
            Factor to multiply historical prices by
        """
        if self.action_type == 'split':
            return 1.0 / self.ratio
        elif self.action_type == 'dividend':
            # For dividends, price adjustment is minimal unless very large
            return 1.0  # Simplified - in practice would need reference price
        elif self.action_type == 'spin_off':
            return 1.0  # Requires complex calculation based on spin-off value
        elif self.action_type == 'merger':
            return self.ratio if self.ratio > 0 else 1.0
        else:
            return 1.0
    
    @property
    def volume_adjustment_factor(self) -> float:
        """
        Calculate volume adjustment factor for historical data.
        
        Returns:
            Factor to multiply historical volumes by
        """
        if self.action_type == 'split':
            return self.ratio
        else:
            return 1.0


class CorporateActionRegistry:
    """
    Registry for corporate actions with loading and adjustment capabilities.
    """
    
    def __init__(self, data_path: str = "artifacts/corp_actions"):
        """
        Initialize corporate action registry.
        
        Args:
            data_path: Path to store corporate action data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Registry of actions by symbol
        self._actions: Dict[str, List[CorporateAction]] = {}
        
        # Load existing action data
        self._load_actions()
    
    def _load_actions(self) -> None:
        """Load corporate actions from disk."""
        actions_file = self.data_path / "corp_actions.json"
        
        if actions_file.exists():
            try:
                with open(actions_file, 'r') as f:
                    data = json.load(f)
                
                for symbol, action_list in data.items():
                    self._actions[symbol] = []
                    for action_data in action_list:
                        # Convert date strings back to date objects
                        if isinstance(action_data['ex_date'], str):
                            action_data['ex_date'] = date.fromisoformat(action_data['ex_date'])
                        
                        self._actions[symbol].append(CorporateAction(**action_data))
                
                self.logger.info(f"Loaded {len(self._actions)} symbols with corporate actions")
                
            except Exception as e:
                self.logger.error(f"Error loading corporate actions: {e}")
                self._actions = {}
    
    def _save_actions(self) -> None:
        """Save corporate actions to disk."""
        actions_file = self.data_path / "corp_actions.json"
        
        try:
            # Convert to serializable format
            data = {}
            for symbol, action_list in self._actions.items():
                data[symbol] = []
                for action in action_list:
                    action_dict = asdict(action)
                    # Convert date to string for JSON serialization
                    action_dict['ex_date'] = action.ex_date.isoformat()
                    data[symbol].append(action_dict)
            
            with open(actions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            self.logger.debug(f"Saved corporate actions to {actions_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving corporate actions: {e}")
    
    def add_action(
        self,
        symbol: str,
        ex_date: Union[date, str],
        action_type: str,
        ratio: float,
        dividend_amount: float = 0.0,
        description: str = "",
        source: str = "manual"
    ) -> None:
        """
        Add corporate action to registry.
        
        Args:
            symbol: Trading symbol
            ex_date: Ex-date as date object or ISO string
            action_type: Type of action ('split', 'dividend', etc.)
            ratio: Action ratio
            dividend_amount: Dividend amount per share
            description: Human readable description
            source: Data source
        """
        symbol = symbol.upper()
        
        if isinstance(ex_date, str):
            ex_date = date.fromisoformat(ex_date)
        
        action = CorporateAction(
            symbol=symbol,
            ex_date=ex_date,
            action_type=action_type,
            ratio=ratio,
            dividend_amount=dividend_amount,
            description=description,
            source=source
        )
        
        if symbol not in self._actions:
            self._actions[symbol] = []
        
        # Insert in chronological order
        self._actions[symbol].append(action)
        self._actions[symbol].sort(key=lambda x: x.ex_date)
        
        self.logger.info(f"Added {action_type} action for {symbol} on {ex_date}: {description}")
        self._save_actions()
    
    def get_actions(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """
        Get corporate actions for symbol within date range.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of CorporateAction objects
        """
        symbol = symbol.upper()
        
        if symbol not in self._actions:
            return []
        
        actions = self._actions[symbol]
        
        if start_date is not None:
            actions = [a for a in actions if a.ex_date >= start_date]
        
        if end_date is not None:
            actions = [a for a in actions if a.ex_date <= end_date]
        
        return actions
    
    def get_adjustment_factors(
        self,
        symbol: str,
        reference_date: date,
        target_date: date
    ) -> Tuple[float, float]:
        """
        Get cumulative adjustment factors between two dates.
        
        Args:
            symbol: Trading symbol  
            reference_date: Reference date (usually more recent)
            target_date: Target date to adjust to
            
        Returns:
            Tuple of (price_factor, volume_factor)
        """
        if reference_date == target_date:
            return 1.0, 1.0
        
        # Get actions between the dates
        start_date = min(reference_date, target_date)
        end_date = max(reference_date, target_date)
        
        actions = self.get_actions(symbol, start_date, end_date)
        
        # Filter to actions that occurred between dates (exclusive of end points)
        if reference_date > target_date:
            # Adjusting backwards - include actions after target_date up to reference_date
            relevant_actions = [a for a in actions if target_date < a.ex_date <= reference_date]
        else:
            # Adjusting forwards - include actions after reference_date up to target_date  
            relevant_actions = [a for a in actions if reference_date < a.ex_date <= target_date]
        
        price_factor = 1.0
        volume_factor = 1.0
        
        for action in relevant_actions:
            price_factor *= action.price_adjustment_factor
            volume_factor *= action.volume_adjustment_factor
        
        # If adjusting backwards, invert the factors
        if reference_date > target_date:
            price_factor = 1.0 / price_factor if price_factor != 0 else 1.0
            volume_factor = 1.0 / volume_factor if volume_factor != 0 else 1.0
        
        return price_factor, volume_factor


# Global registry instance
_global_registry: Optional[CorporateActionRegistry] = None


def get_corp_action_registry() -> CorporateActionRegistry:
    """Get or create global corporate action registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CorporateActionRegistry()
    return _global_registry


def adjust_bars(
    bars: pd.DataFrame,
    symbol: str,
    reference_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Adjust OHLCV bars for corporate actions.
    
    Args:
        bars: DataFrame with OHLCV data and datetime index
        symbol: Trading symbol
        reference_date: Reference date for adjustments (defaults to most recent)
        
    Returns:
        DataFrame with adjusted OHLCV data
    """
    if bars.empty:
        return bars
    
    registry = get_corp_action_registry()
    
    # Use most recent date as reference if not specified
    if reference_date is None:
        if hasattr(bars.index, 'date'):
            reference_date = bars.index.date.max()
        else:
            reference_date = pd.to_datetime(bars.index).date.max()
    
    # Create copy to avoid modifying original
    adjusted_bars = bars.copy()
    
    # Get price columns (flexible column naming)
    price_cols = []
    volume_cols = []
    
    for col in bars.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in ['open', 'high', 'low', 'close', 'price', 'adj_close']):
            price_cols.append(col)
        elif any(v in col_lower for v in ['volume', 'vol']):
            volume_cols.append(col)
    
    # Apply adjustments row by row
    for idx, row in bars.iterrows():
        if hasattr(idx, 'date'):
            bar_date = idx.date()
        else:
            bar_date = pd.to_datetime(idx).date()
        
        # Get adjustment factors for this bar
        price_factor, volume_factor = registry.get_adjustment_factors(
            symbol, reference_date, bar_date
        )
        
        # Apply price adjustments
        for col in price_cols:
            if not pd.isna(adjusted_bars.loc[idx, col]):
                adjusted_bars.loc[idx, col] *= price_factor
        
        # Apply volume adjustments
        for col in volume_cols:
            if not pd.isna(adjusted_bars.loc[idx, col]):
                adjusted_bars.loc[idx, col] *= volume_factor
    
    return adjusted_bars


def apply_adjustment_factor(price: float, factor: float) -> float:
    """
    Apply adjustment factor to a price.
    
    Args:
        price: Original price
        factor: Adjustment factor
        
    Returns:
        Adjusted price
    """
    return price * factor


# AI-AGENT-REF: Add some common stock splits for testing
def populate_common_splits():
    """Add some well-known historical stock splits for testing."""
    registry = get_corp_action_registry()
    
    # Tesla 3:1 split in August 2022
    registry.add_action(
        symbol="TSLA",
        ex_date="2022-08-25",
        action_type="split",
        ratio=3.0,
        description="3-for-1 stock split",
        source="manual"
    )
    
    # Apple 4:1 split in August 2020
    registry.add_action(
        symbol="AAPL", 
        ex_date="2020-08-31",
        action_type="split",
        ratio=4.0,
        description="4-for-1 stock split",
        source="manual"
    )
    
    # NVIDIA 4:1 split in July 2021
    registry.add_action(
        symbol="NVDA",
        ex_date="2021-07-20", 
        action_type="split",
        ratio=4.0,
        description="4-for-1 stock split",
        source="manual"
    )