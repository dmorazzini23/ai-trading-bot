"""
Advanced execution algorithms for institutional trading.

Provides VWAP, TWAP, Implementation Shortfall, and other
sophisticated execution algorithms.
"""

import time
import math
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import threading
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.enums import OrderSide, OrderType
from .engine import Order, OrderManager


class VWAPExecutor:
    """
    Volume Weighted Average Price execution algorithm.
    
    Executes orders to match the market's volume profile
    and achieve volume-weighted average price.
    """
    
    def __init__(self, order_manager: OrderManager):
        """Initialize VWAP executor."""
        # AI-AGENT-REF: VWAP execution algorithm with optimized participation rate
        self.order_manager = order_manager
        self.participation_rate = 0.15  # Optimized 15% of market volume (increased from 10%)
        self.min_slice_size = 100
        logger.info("VWAPExecutor initialized with optimized participation rate")
    
    def execute_vwap_order(self, symbol: str, side: OrderSide, total_quantity: int, 
                          duration_minutes: int = 60, **kwargs) -> List[str]:
        """
        Execute order using VWAP algorithm.
        
        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            duration_minutes: Execution duration in minutes
            
        Returns:
            List of child order IDs
        """
        try:
            # Calculate number of slices based on duration - optimized to 8 slices instead of 10
            slice_interval_minutes = max(5, duration_minutes // 8)  # Optimize to 8 slices for better execution
            num_slices = max(1, duration_minutes // slice_interval_minutes)
            slice_quantity = max(self.min_slice_size, total_quantity // num_slices)
            
            child_orders = []
            remaining_quantity = total_quantity
            
            for i in range(num_slices):
                if remaining_quantity <= 0:
                    break
                
                # Calculate slice size (larger slices during high volume periods)
                current_slice = min(slice_quantity, remaining_quantity)
                
                # Create child order
                child_order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=current_slice,
                    order_type=OrderType.LIMIT,
                    parent_order_id=kwargs.get('parent_order_id'),
                    strategy_id=kwargs.get('strategy_id'),
                    execution_algorithm="vwap"
                )
                
                # Submit order
                if self.order_manager.submit_order(child_order):
                    child_orders.append(child_order.id)
                    remaining_quantity -= current_slice
                    
                    logger.debug(f"VWAP slice {i+1}/{num_slices}: {current_slice} shares")
                
                # Wait between slices
                time.sleep(300)  # 5 minutes
            
            logger.info(f"VWAP execution completed: {len(child_orders)} orders, "
                       f"{total_quantity - remaining_quantity} shares executed")
            
            return child_orders
            
        except Exception as e:
            logger.error(f"Error in VWAP execution: {e}")
            return []


class TWAPExecutor:
    """
    Time Weighted Average Price execution algorithm.
    
    Executes orders evenly over time to minimize market impact
    and achieve time-weighted average price.
    """
    
    def __init__(self, order_manager: OrderManager):
        """Initialize TWAP executor."""
        # AI-AGENT-REF: TWAP execution algorithm
        self.order_manager = order_manager
        self.min_slice_size = 100
        logger.info("TWAPExecutor initialized")
    
    def execute_twap_order(self, symbol: str, side: OrderSide, total_quantity: int,
                          duration_minutes: int = 60, **kwargs) -> List[str]:
        """
        Execute order using TWAP algorithm.
        
        Args:
            symbol: Trading symbol
            side: Order side  
            total_quantity: Total quantity to execute
            duration_minutes: Execution duration in minutes
            
        Returns:
            List of child order IDs
        """
        try:
            # Calculate equal time slices - optimized to 8 slices for better execution
            slice_interval = max(1, duration_minutes // 8)  # Optimize to 8 slices instead of 10
            num_slices = duration_minutes // slice_interval
            slice_quantity = max(self.min_slice_size, total_quantity // num_slices)
            
            child_orders = []
            remaining_quantity = total_quantity
            
            for i in range(num_slices):
                if remaining_quantity <= 0:
                    break
                
                # Equal slice sizes for TWAP
                current_slice = min(slice_quantity, remaining_quantity)
                
                # Create child order
                child_order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=current_slice,
                    order_type=OrderType.LIMIT,
                    parent_order_id=kwargs.get('parent_order_id'),
                    strategy_id=kwargs.get('strategy_id'),
                    execution_algorithm="twap"
                )
                
                # Submit order
                if self.order_manager.submit_order(child_order):
                    child_orders.append(child_order.id)
                    remaining_quantity -= current_slice
                    
                    logger.debug(f"TWAP slice {i+1}/{num_slices}: {current_slice} shares")
                
                # Equal time intervals
                time.sleep(slice_interval * 60)  # Convert to seconds
            
            logger.info(f"TWAP execution completed: {len(child_orders)} orders")
            return child_orders
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {e}")
            return []


class ImplementationShortfall:
    """
    Implementation Shortfall execution algorithm.
    
    Optimizes execution to minimize implementation shortfall
    by balancing market impact and timing risk.
    """
    
    def __init__(self, order_manager: OrderManager):
        """Initialize Implementation Shortfall executor."""
        # AI-AGENT-REF: Implementation Shortfall algorithm
        self.order_manager = order_manager
        self.urgency_factor = 0.5  # 0 = patient, 1 = aggressive
        logger.info("ImplementationShortfall initialized")
    
    def execute_is_order(self, symbol: str, side: OrderSide, total_quantity: int,
                        benchmark_price: float, urgency: float = 0.5, **kwargs) -> List[str]:
        """
        Execute order using Implementation Shortfall algorithm.
        
        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            benchmark_price: Benchmark price for calculation
            urgency: Urgency factor (0-1)
            
        Returns:
            List of child order IDs
        """
        try:
            self.urgency_factor = urgency
            
            # Calculate optimal participation rate based on urgency
            participation_rate = 0.05 + (urgency * 0.15)  # 5-20% based on urgency
            
            # Calculate execution schedule
            execution_schedule = self._calculate_execution_schedule(
                total_quantity, participation_rate
            )
            
            child_orders = []
            
            for i, (slice_quantity, slice_urgency) in enumerate(execution_schedule):
                # Adjust order type based on urgency
                order_type = OrderType.MARKET if slice_urgency > 0.8 else OrderType.LIMIT
                
                child_order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=slice_quantity,
                    order_type=order_type,
                    parent_order_id=kwargs.get('parent_order_id'),
                    strategy_id=kwargs.get('strategy_id'),
                    execution_algorithm="implementation_shortfall"
                )
                
                if self.order_manager.submit_order(child_order):
                    child_orders.append(child_order.id)
                    
                    logger.debug(f"IS slice {i+1}: {slice_quantity} shares, "
                               f"urgency={slice_urgency:.2f}, type={order_type}")
                
                # Dynamic wait based on market conditions
                wait_time = self._calculate_wait_time(slice_urgency)
                time.sleep(wait_time)
            
            logger.info(f"Implementation Shortfall execution completed: {len(child_orders)} orders")
            return child_orders
            
        except Exception as e:
            logger.error(f"Error in Implementation Shortfall execution: {e}")
            return []
    
    def _calculate_execution_schedule(self, total_quantity: int, 
                                    participation_rate: float) -> List[tuple]:
        """Calculate optimal execution schedule."""
        try:
            # Simple schedule: front-load based on urgency
            num_slices = max(3, min(10, total_quantity // 100))
            schedule = []
            
            remaining = total_quantity
            for i in range(num_slices):
                if remaining <= 0:
                    break
                
                # Front-loading factor based on urgency
                front_load_factor = 1 + (self.urgency_factor * (num_slices - i) / num_slices)
                slice_qty = min(remaining, int(total_quantity / num_slices * front_load_factor))
                
                # Urgency increases over time if not filled
                slice_urgency = self.urgency_factor + (i / num_slices) * 0.3
                slice_urgency = min(1.0, slice_urgency)
                
                schedule.append((slice_qty, slice_urgency))
                remaining -= slice_qty
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating execution schedule: {e}")
            return [(total_quantity, self.urgency_factor)]
    
    def _calculate_wait_time(self, urgency: float) -> int:
        """Calculate wait time between slices based on urgency."""
        # High urgency = shorter wait time
        base_wait = 180  # 3 minutes
        urgency_factor = 1 - urgency
        wait_time = int(base_wait * urgency_factor)
        return max(30, wait_time)  # Minimum 30 seconds