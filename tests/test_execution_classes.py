"""
Tests for ExecutionResult and OrderRequest classes.

Validates the newly added classes that fix the ImportError in the execution module.
"""


# Import our new classes
from ai_trading.execution import ExecutionResult, OrderRequest
from ai_trading.core.enums import OrderSide, OrderType


class TestExecutionResult:
    """Test cases for ExecutionResult class."""
    
    def test_execution_result_creation(self):
        """Test basic ExecutionResult instantiation."""
        result = ExecutionResult(
            status="success",
            order_id="test_123",
            symbol="AAPL", 
            side="buy",
            quantity=100,
            fill_price=150.0,
            message="Order filled successfully"
        )
        
        assert result.status == "success"
        assert result.order_id == "test_123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 100
        assert result.fill_price == 150.0
        assert result.message == "Order filled successfully"
        assert result.is_successful is True
        assert result.is_failed is False
        assert result.is_partial is False
    
    def test_execution_result_failed_status(self):
        """Test ExecutionResult with failed status."""
        result = ExecutionResult(
            status="failed",
            order_id="test_456",
            symbol="MSFT",
            message="Insufficient funds"
        )
        
        assert result.status == "failed"
        assert result.is_successful is False
        assert result.is_failed is True
        assert result.is_partial is False
    
    def test_execution_result_partial_status(self):
        """Test ExecutionResult with partial status."""
        result = ExecutionResult(
            status="partial",
            order_id="test_789",
            symbol="GOOGL",
            side="sell",
            quantity=50,
            fill_price=2800.0
        )
        
        assert result.status == "partial"
        assert result.is_successful is False
        assert result.is_failed is False
        assert result.is_partial is True
    
    def test_execution_result_to_dict(self):
        """Test ExecutionResult to_dict conversion."""
        result = ExecutionResult(
            status="success",
            order_id="test_dict",
            symbol="TSLA",
            side="buy",
            quantity=25,
            fill_price=800.0,
            actual_slippage_bps=5.2,
            notional_value=20000.0
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["status"] == "success"
        assert result_dict["order_id"] == "test_dict"
        assert result_dict["symbol"] == "TSLA"
        assert result_dict["side"] == "buy"
        assert result_dict["quantity"] == 25
        assert result_dict["fill_price"] == 800.0
        assert result_dict["actual_slippage_bps"] == 5.2
        assert result_dict["notional_value"] == 20000.0
        assert result_dict["is_successful"] is True
        assert result_dict["is_failed"] is False
        assert result_dict["is_partial"] is False
        assert "timestamp" in result_dict
    
    def test_execution_result_string_representation(self):
        """Test ExecutionResult string representations."""
        result = ExecutionResult("success", "order_1", "AAPL")
        
        str_repr = str(result)
        assert "ExecutionResult" in str_repr
        assert "success" in str_repr
        assert "order_1" in str_repr
        assert "AAPL" in str_repr
        
        repr_str = repr(result)
        assert "ExecutionResult" in repr_str
        assert "status='success'" in repr_str
        assert "order_id='order_1'" in repr_str


class TestOrderRequest:
    """Test cases for OrderRequest class."""
    
    def test_order_request_creation_valid(self):
        """Test valid OrderRequest creation."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            strategy="test_strategy"
        )
        
        assert request.symbol == "AAPL"
        assert request.side == OrderSide.BUY
        assert request.quantity == 100
        assert request.order_type == OrderType.MARKET
        assert request.strategy == "test_strategy"
        assert request.is_valid is True
        assert len(request.validation_errors) == 0
    
    def test_order_request_limit_order(self):
        """Test limit order creation."""
        request = OrderRequest(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            price=300.0
        )
        
        assert request.order_type == OrderType.LIMIT
        assert request.price == 300.0
        assert request.is_valid is True
    
    def test_order_request_validation_empty_symbol(self):
        """Test validation with empty symbol."""
        request = OrderRequest(
            symbol="",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert request.is_valid is False
        assert "Symbol is required" in str(request.validation_errors)
    
    def test_order_request_validation_negative_quantity(self):
        """Test validation with negative quantity."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=-10,
            order_type=OrderType.MARKET
        )
        
        assert request.is_valid is False
        assert "Quantity must be positive" in str(request.validation_errors)
    
    def test_order_request_validation_limit_without_price(self):
        """Test validation of limit order without price."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT
            # No price provided
        )
        
        assert request.is_valid is False
        assert "Limit orders require a valid price" in str(request.validation_errors)
    
    def test_order_request_validation_excessive_quantity(self):
        """Test validation with excessive quantity."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=2000000,  # Exceeds limit
            order_type=OrderType.MARKET
        )
        
        assert request.is_valid is False
        assert "exceeds maximum limit" in str(request.validation_errors)
    
    def test_order_request_notional_value(self):
        """Test notional value calculation."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=150.0
        )
        
        assert request.notional_value == 15000.0  # 100 * 150.0
    
    def test_order_request_to_dict(self):
        """Test OrderRequest to_dict conversion."""
        request = OrderRequest(
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=25,
            order_type=OrderType.LIMIT,
            price=2800.0,
            strategy="momentum",
            time_in_force="GTC"
        )
        
        request_dict = request.to_dict()
        
        assert isinstance(request_dict, dict)
        assert request_dict["symbol"] == "GOOGL"
        assert request_dict["side"] == "sell"
        assert request_dict["quantity"] == 25
        assert request_dict["order_type"] == "limit"
        assert request_dict["price"] == 2800.0
        assert request_dict["strategy"] == "momentum"
        assert request_dict["time_in_force"] == "GTC"
        assert request_dict["is_valid"] is True
        assert "created_at" in request_dict
        assert "request_id" in request_dict
    
    def test_order_request_to_api_request_alpaca(self):
        """Test OrderRequest to_api_request for Alpaca format."""
        request = OrderRequest(
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        api_request = request.to_api_request("alpaca")
        
        assert isinstance(api_request, dict)
        assert api_request["symbol"] == "TSLA"
        assert api_request["side"] == "buy"
        assert api_request["type"] == "market"
        assert api_request["qty"] == "10"
        assert "client_order_id" in api_request
    
    def test_order_request_copy(self):
        """Test OrderRequest copy functionality."""
        original = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        copy = original.copy(quantity=200, strategy="new_strategy")
        
        assert copy.symbol == original.symbol
        assert copy.side == original.side
        assert copy.order_type == original.order_type
        assert copy.quantity == 200  # Updated
        assert copy.strategy == "new_strategy"  # Updated
        assert copy.client_order_id != original.client_order_id  # Should be different
    
    def test_order_request_string_representation(self):
        """Test OrderRequest string representations."""
        request = OrderRequest("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        
        str_repr = str(request)
        assert "OrderRequest" in str_repr
        assert "buy" in str_repr
        assert "100" in str_repr
        assert "AAPL" in str_repr
        assert "market" in str_repr
        
        repr_str = repr(request)
        assert "OrderRequest" in repr_str
        assert "symbol='AAPL'" in repr_str
        assert "valid=True" in repr_str


class TestExecutionIntegration:
    """Test integration between ExecutionResult and OrderRequest."""
    
    def test_order_request_to_execution_result_flow(self):
        """Test the flow from OrderRequest to ExecutionResult."""
        # Create a valid order request
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=150.0,
            strategy="test"
        )
        
        assert request.is_valid
        
        # Simulate successful execution
        result = ExecutionResult(
            status="success",
            order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side.value,
            quantity=request.quantity,
            fill_price=request.price,
            message="Order executed successfully"
        )
        
        assert result.is_successful
        assert result.symbol == request.symbol
        assert result.quantity == request.quantity
        assert result.order_id == request.client_order_id
    
    def test_invalid_request_to_rejected_result(self):
        """Test flow from invalid request to rejected result."""
        # Create invalid order request
        request = OrderRequest(
            symbol="",  # Invalid empty symbol
            side=OrderSide.BUY,
            quantity=-10,  # Invalid negative quantity
            order_type=OrderType.MARKET
        )
        
        assert not request.is_valid
        assert len(request.validation_errors) > 0
        
        # Simulate rejection
        result = ExecutionResult(
            status="rejected",
            order_id=request.client_order_id,
            symbol=request.symbol,
            message=f"Validation failed: {'; '.join(request.validation_errors)}"
        )
        
        assert result.is_failed
        assert "Validation failed" in result.message