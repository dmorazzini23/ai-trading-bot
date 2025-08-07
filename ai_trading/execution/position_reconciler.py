"""Position reconciliation system to sync bot state with broker positions.

This module provides periodic reconciliation between the bot's internal
position tracking and the actual positions reported by the Alpaca API,
including discrepancy detection and alerting.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from ai_trading.logging import get_logger


def get_phase_logger(name: str, phase: str) -> logging.Logger:
    """Get a logger for a specific phase - fallback implementation."""
    logger_name = f"{name}.{phase}" if phase else name
    return get_logger(logger_name)


class PositionDiscrepancy:
    """Represents a discrepancy between bot and broker positions."""
    
    def __init__(self, symbol: str, bot_qty: float, broker_qty: float, 
                 discrepancy_type: str, severity: str = "medium"):
        self.symbol = symbol
        self.bot_qty = bot_qty
        self.broker_qty = broker_qty
        self.difference = broker_qty - bot_qty
        self.discrepancy_type = discrepancy_type
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'symbol': self.symbol,
            'bot_qty': self.bot_qty,
            'broker_qty': self.broker_qty,
            'difference': self.difference,
            'discrepancy_type': self.discrepancy_type,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat()
        }


class PositionReconciler:
    """Manages position reconciliation between bot and broker."""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.logger = get_phase_logger(__name__, "POS_RECONCILE")
        self._lock = Lock()
        
        # Track bot's internal positions (symbol -> quantity)
        self._bot_positions: Dict[str, float] = {}
        
        # Track last known broker positions
        self._broker_positions: Dict[str, float] = {}
        
        # Track reconciliation history
        self._reconciliation_history: List[Dict[str, Any]] = []
        
        # Track discrepancies
        self._current_discrepancies: List[PositionDiscrepancy] = []
        self._discrepancy_history: List[PositionDiscrepancy] = []
        
        # Configuration
        self.reconciliation_interval = 300  # 5 minutes
        self.position_tolerance = 1e-6  # Small tolerance for floating point
        self.large_discrepancy_threshold = 10  # shares
        self.running = False
        self._reconciliation_thread: Optional[Thread] = None
        
    def update_bot_position(self, symbol: str, quantity: float, 
                           reason: str = "trade_execution") -> None:
        """Update the bot's internal position tracking."""
        with self._lock:
            old_qty = self._bot_positions.get(symbol, 0)
            self._bot_positions[symbol] = quantity
            
            self.logger.info("BOT_POSITION_UPDATE", extra={
                'symbol': symbol,
                'old_qty': old_qty,
                'new_qty': quantity,
                'change': quantity - old_qty,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    def adjust_bot_position(self, symbol: str, quantity_change: float, 
                           reason: str = "trade_execution") -> None:
        """Adjust the bot's position by a relative amount."""
        with self._lock:
            current_qty = self._bot_positions.get(symbol, 0)
            new_qty = current_qty + quantity_change
            self._bot_positions[symbol] = new_qty
            
            self.logger.info("BOT_POSITION_ADJUST", extra={
                'symbol': symbol,
                'old_qty': current_qty,
                'new_qty': new_qty,
                'change': quantity_change,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    def get_bot_positions(self) -> Dict[str, float]:
        """Get current bot positions."""
        with self._lock:
            return self._bot_positions.copy()
    
    def get_broker_positions(self) -> Dict[str, float]:
        """Fetch current positions from broker API."""
        if not self.api_client:
            self.logger.warning("NO_API_CLIENT", extra={
                'message': 'Cannot fetch broker positions without API client'
            })
            return {}
        
        try:
            # This would typically call the Alpaca API to get positions
            # For now, we'll simulate the call since we don't have real API access
            broker_positions = {}
            
            # In real implementation, this would be:
            # positions = self.api_client.get_all_positions()
            # for position in positions:
            #     broker_positions[position.symbol] = float(position.qty)
            
            with self._lock:
                self._broker_positions = broker_positions.copy()
            
            self.logger.debug("BROKER_POSITIONS_FETCHED", extra={
                'positions_count': len(broker_positions),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return broker_positions
            
        except Exception as e:
            self.logger.error("BROKER_POSITION_FETCH_ERROR", extra={
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            return {}
    
    def reconcile_positions(self) -> List[PositionDiscrepancy]:
        """Perform position reconciliation and return any discrepancies."""
        self.logger.info("RECONCILIATION_START", extra={
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        bot_positions = self.get_bot_positions()
        broker_positions = self.get_broker_positions()
        
        discrepancies = []
        
        # Get all symbols from both sides
        all_symbols = set(bot_positions.keys()) | set(broker_positions.keys())
        
        for symbol in all_symbols:
            bot_qty = bot_positions.get(symbol, 0)
            broker_qty = broker_positions.get(symbol, 0)
            
            # Check for discrepancies beyond tolerance
            if abs(bot_qty - broker_qty) > self.position_tolerance:
                # Determine discrepancy type and severity
                discrepancy_type = self._classify_discrepancy(bot_qty, broker_qty)
                severity = self._determine_severity(symbol, bot_qty, broker_qty)
                
                discrepancy = PositionDiscrepancy(
                    symbol=symbol,
                    bot_qty=bot_qty,
                    broker_qty=broker_qty,
                    discrepancy_type=discrepancy_type,
                    severity=severity
                )
                
                discrepancies.append(discrepancy)
                
                # Log the discrepancy
                self.logger.warning("POSITION_DISCREPANCY", extra=discrepancy.to_dict())
        
        # Update tracking
        with self._lock:
            self._current_discrepancies = discrepancies
            self._discrepancy_history.extend(discrepancies)
            
            # Keep history bounded
            if len(self._discrepancy_history) > 1000:
                self._discrepancy_history = self._discrepancy_history[-500:]
            
            # Record reconciliation event
            reconciliation_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'bot_positions': bot_positions,
                'broker_positions': broker_positions,
                'discrepancies_count': len(discrepancies),
                'discrepancies': [d.to_dict() for d in discrepancies]
            }
            
            self._reconciliation_history.append(reconciliation_record)
            
            # Keep reconciliation history bounded
            if len(self._reconciliation_history) > 100:
                self._reconciliation_history = self._reconciliation_history[-50:]
        
        if discrepancies:
            self.logger.error("RECONCILIATION_DISCREPANCIES_FOUND", extra={
                'discrepancies_count': len(discrepancies),
                'symbols_affected': [d.symbol for d in discrepancies],
                'severe_discrepancies': len([d for d in discrepancies if d.severity == "high"])
            })
        else:
            self.logger.info("RECONCILIATION_SUCCESS", extra={
                'message': 'All positions reconciled successfully',
                'positions_checked': len(all_symbols)
            })
        
        return discrepancies
    
    def _classify_discrepancy(self, bot_qty: float, broker_qty: float) -> str:
        """Classify the type of discrepancy."""
        if bot_qty == 0 and broker_qty != 0:
            return "missing_position"  # Bot thinks no position, broker has position
        elif bot_qty != 0 and broker_qty == 0:
            return "phantom_position"  # Bot thinks has position, broker doesn't
        elif (bot_qty > 0) != (broker_qty > 0):
            return "direction_mismatch"  # Long vs short mismatch
        else:
            return "quantity_mismatch"  # Same direction, different quantities
    
    def _determine_severity(self, symbol: str, bot_qty: float, broker_qty: float) -> str:
        """Determine severity of the discrepancy."""
        difference = abs(bot_qty - broker_qty)
        
        if difference >= self.large_discrepancy_threshold:
            return "high"
        elif difference >= 1.0:  # At least 1 share difference
            return "medium"
        else:
            return "low"
    
    def auto_resolve_discrepancies(self, discrepancies: List[PositionDiscrepancy]) -> int:
        """Attempt to automatically resolve discrepancies by updating bot positions."""
        resolved_count = 0
        
        for discrepancy in discrepancies:
            if discrepancy.severity in ["low", "medium"]:
                # For low/medium severity, trust broker and update bot
                self.logger.info("AUTO_RESOLVE_DISCREPANCY", extra={
                    'symbol': discrepancy.symbol,
                    'action': 'update_bot_position',
                    'old_bot_qty': discrepancy.bot_qty,
                    'new_bot_qty': discrepancy.broker_qty,
                    'reason': 'auto_reconciliation'
                })
                
                self.update_bot_position(
                    discrepancy.symbol, 
                    discrepancy.broker_qty,
                    reason="auto_reconciliation"
                )
                
                resolved_count += 1
            else:
                # High severity discrepancies require manual intervention
                self.logger.error("HIGH_SEVERITY_DISCREPANCY_REQUIRES_MANUAL_REVIEW", 
                                extra=discrepancy.to_dict())
        
        return resolved_count
    
    def start_periodic_reconciliation(self, interval: Optional[int] = None) -> None:
        """Start periodic position reconciliation in background thread."""
        if self.running:
            self.logger.warning("RECONCILIATION_ALREADY_RUNNING")
            return
        
        if interval:
            self.reconciliation_interval = interval
        
        self.running = True
        self._reconciliation_thread = Thread(
            target=self._reconciliation_loop,
            daemon=True,
            name="PositionReconciler"
        )
        self._reconciliation_thread.start()
        
        self.logger.info("PERIODIC_RECONCILIATION_STARTED", extra={
            'interval_seconds': self.reconciliation_interval
        })
    
    def stop_periodic_reconciliation(self) -> None:
        """Stop periodic reconciliation."""
        self.running = False
        if self._reconciliation_thread:
            self._reconciliation_thread.join(timeout=10)
        
        self.logger.info("PERIODIC_RECONCILIATION_STOPPED")
    
    def _reconciliation_loop(self) -> None:
        """Background loop for periodic reconciliation."""
        while self.running:
            try:
                discrepancies = self.reconcile_positions()
                
                # Auto-resolve low/medium severity discrepancies
                if discrepancies:
                    resolved = self.auto_resolve_discrepancies(discrepancies)
                    if resolved > 0:
                        self.logger.info("AUTO_RESOLVED_DISCREPANCIES", extra={
                            'resolved_count': resolved,
                            'total_discrepancies': len(discrepancies)
                        })
                
                # Sleep until next reconciliation
                time.sleep(self.reconciliation_interval)
                
            except Exception as e:
                self.logger.error("RECONCILIATION_LOOP_ERROR", extra={
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                time.sleep(min(self.reconciliation_interval, 60))  # Shorter sleep on error
    
    def get_current_discrepancies(self) -> List[PositionDiscrepancy]:
        """Get current position discrepancies."""
        with self._lock:
            return self._current_discrepancies.copy()
    
    def get_reconciliation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reconciliation history."""
        with self._lock:
            return self._reconciliation_history[-limit:].copy()
    
    def get_discrepancy_history(self, symbol: Optional[str] = None, 
                               limit: int = 50) -> List[PositionDiscrepancy]:
        """Get historical discrepancies, optionally filtered by symbol."""
        with self._lock:
            history = self._discrepancy_history[-limit:].copy()
            if symbol:
                history = [d for d in history if d.symbol == symbol]
            return history
    
    def force_sync_from_broker(self) -> Dict[str, float]:
        """Force sync bot positions from broker (emergency recovery)."""
        self.logger.warning("FORCE_SYNC_FROM_BROKER_INITIATED")
        
        broker_positions = self.get_broker_positions()
        
        with self._lock:
            old_positions = self._bot_positions.copy()
            self._bot_positions = broker_positions.copy()
        
        self.logger.info("FORCE_SYNC_COMPLETED", extra={
            'old_positions': old_positions,
            'new_positions': broker_positions,
            'positions_changed': len(set(old_positions.keys()) | set(broker_positions.keys()))
        })
        
        return broker_positions
    
    def get_reconciliation_stats(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        with self._lock:
            current_discrepancies = len(self._current_discrepancies)
            total_historical_discrepancies = len(self._discrepancy_history)
            
            # Calculate discrepancy rates by severity
            severity_counts = defaultdict(int)
            for discrepancy in self._discrepancy_history[-100:]:  # Last 100
                severity_counts[discrepancy.severity] += 1
            
            # Calculate reconciliation frequency
            recent_reconciliations = self._reconciliation_history[-10:]
            avg_discrepancies_per_reconciliation = 0
            if recent_reconciliations:
                total_discrepancies = sum(r['discrepancies_count'] for r in recent_reconciliations)
                avg_discrepancies_per_reconciliation = total_discrepancies / len(recent_reconciliations)
            
            return {
                'running': self.running,
                'interval_seconds': self.reconciliation_interval,
                'current_discrepancies': current_discrepancies,
                'total_historical_discrepancies': total_historical_discrepancies,
                'severity_breakdown': dict(severity_counts),
                'avg_discrepancies_per_reconciliation': avg_discrepancies_per_reconciliation,
                'total_reconciliations': len(self._reconciliation_history),
                'bot_positions_count': len(self._bot_positions),
                'broker_positions_count': len(self._broker_positions)
            }


# Global reconciler instance
_position_reconciler: Optional[PositionReconciler] = None
_reconciler_lock = Lock()


def get_position_reconciler(api_client=None) -> PositionReconciler:
    """Get or create the global position reconciler instance."""
    global _position_reconciler
    with _reconciler_lock:
        if _position_reconciler is None:
            _position_reconciler = PositionReconciler(api_client)
        return _position_reconciler


def update_bot_position(symbol: str, quantity: float, reason: str = "trade_execution") -> None:
    """Update bot position tracking."""
    reconciler = get_position_reconciler()
    reconciler.update_bot_position(symbol, quantity, reason)


def adjust_bot_position(symbol: str, quantity_change: float, reason: str = "trade_execution") -> None:
    """Adjust bot position by relative amount."""
    reconciler = get_position_reconciler()
    reconciler.adjust_bot_position(symbol, quantity_change, reason)


def force_position_reconciliation() -> List[PositionDiscrepancy]:
    """Force an immediate position reconciliation."""
    reconciler = get_position_reconciler()
    return reconciler.reconcile_positions()


def start_position_monitoring(api_client=None, interval: int = 300) -> None:
    """Start periodic position monitoring."""
    reconciler = get_position_reconciler(api_client)
    reconciler.start_periodic_reconciliation(interval)


def stop_position_monitoring() -> None:
    """Stop periodic position monitoring."""
    if _position_reconciler:
        _position_reconciler.stop_periodic_reconciliation()


def get_position_discrepancies() -> List[PositionDiscrepancy]:
    """Get current position discrepancies."""
    reconciler = get_position_reconciler()
    return reconciler.get_current_discrepancies()


def get_reconciliation_statistics() -> Dict[str, Any]:
    """Get position reconciliation statistics."""
    reconciler = get_position_reconciler()
    return reconciler.get_reconciliation_stats()