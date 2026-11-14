"""
Corporate actions adjustment pipeline for unified price/volume adjustments.

Provides single source of truth for corporate action adjustments used by
features, labels, and execution sizing to ensure consistency.
"""
from __future__ import annotations

import json
from ai_trading.logging import get_logger
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    import pandas as pd
logger = get_logger(__name__)

@dataclass
class CorporateAction:
    """Single corporate action event."""
    symbol: str
    ex_date: date
    action_type: str
    ratio: float
    dividend_amount: float = 0.0
    cash_amount: float = 0.0
    description: str = ''
    source: str = ''

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
            return 1.0
        elif self.action_type == 'spin_off':
            return 1.0
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

    def __init__(self, data_path: str='artifacts/corp_actions'):
        """
        Initialize corporate action registry.
        
        Args:
            data_path: Path to store corporate action data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self._actions: dict[str, list[CorporateAction]] = {}
        self._load_actions()

    def _load_actions(self) -> None:
        """Load corporate actions from disk."""
        actions_file = self.data_path / 'corp_actions.json'
        if actions_file.exists():
            try:
                with open(actions_file) as f:
                    data = json.load(f)
                for symbol, action_list in data.items():
                    self._actions[symbol] = []
                    for action_data in action_list:
                        if isinstance(action_data['ex_date'], str):
                            action_data['ex_date'] = date.fromisoformat(action_data['ex_date'])
                        self._actions[symbol].append(CorporateAction(**action_data))
                self.logger.info(f'Loaded {len(self._actions)} symbols with corporate actions')
            except (ValueError, TypeError) as e:
                self.logger.error(f'Error loading corporate actions: {e}')
                self._actions = {}

    def _save_actions(self) -> None:
        """Save corporate actions to disk."""
        actions_file = self.data_path / 'corp_actions.json'
        try:
            data: dict[str, list[dict[str, Any]]] = {}
            for symbol, action_list in self._actions.items():
                data[symbol] = []
                for action in action_list:
                    action_dict = asdict(action)
                    action_dict['ex_date'] = action.ex_date.isoformat()
                    data[symbol].append(action_dict)
            with open(actions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.debug(f'Saved corporate actions to {actions_file}')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error saving corporate actions: {e}')

    def add_action(self, symbol: str, ex_date: date | str, action_type: str, ratio: float, dividend_amount: float=0.0, description: str='', source: str='manual') -> None:
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
        action = CorporateAction(symbol=symbol, ex_date=ex_date, action_type=action_type, ratio=ratio, dividend_amount=dividend_amount, description=description, source=source)
        if symbol not in self._actions:
            self._actions[symbol] = []
        self._actions[symbol].append(action)
        self._actions[symbol].sort(key=lambda x: x.ex_date)
        self.logger.info(f'Added {action_type} action for {symbol} on {ex_date}: {description}')
        self._save_actions()

    def get_actions(self, symbol: str, start_date: date | None=None, end_date: date | None=None) -> list[CorporateAction]:
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

    def get_adjustment_factors(self, symbol: str, reference_date: date, target_date: date) -> tuple[float, float]:
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
            return (1.0, 1.0)
        start_date = min(reference_date, target_date)
        end_date = max(reference_date, target_date)
        actions = self.get_actions(symbol, start_date, end_date)
        if reference_date > target_date:
            relevant_actions = [a for a in actions if target_date < a.ex_date <= reference_date]
        else:
            relevant_actions = [a for a in actions if reference_date < a.ex_date <= target_date]
        price_factor = 1.0
        volume_factor = 1.0
        for action in relevant_actions:
            price_factor *= action.price_adjustment_factor
            volume_factor *= action.volume_adjustment_factor
        if reference_date < target_date:
            price_factor = 1.0 / price_factor if price_factor != 0 else 1.0
            volume_factor = 1.0 / volume_factor if volume_factor != 0 else 1.0
        return (price_factor, volume_factor)
_global_registry: CorporateActionRegistry | None = None

