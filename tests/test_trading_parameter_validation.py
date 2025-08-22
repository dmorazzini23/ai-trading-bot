"""Test trading parameter validation in bot_engine.py"""
import ast
from pathlib import Path


def test_validate_trading_parameters_no_name_error():
    """Test that validate_trading_parameters function references only defined parameters.
    
    This test parses the bot_engine.py source code and validates that all parameters
    referenced in validate_trading_parameters() are defined before the function call.
    """

    # Read the bot_engine.py source code
    src_path = Path(__file__).resolve().parents[1] / 'bot_engine.py'
    source = src_path.read_text()

    # Parse the AST
    tree = ast.parse(source)

    # Find the validate_trading_parameters function definition
    validate_func = None
    validate_call_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'validate_trading_parameters':
            validate_func = node
        elif isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and
                node.func.id == 'validate_trading_parameters'):
                validate_call_line = node.lineno

    assert validate_func is not None, "validate_trading_parameters function should exist"
    assert validate_call_line is not None, "validate_trading_parameters should be called"

    # Extract parameters referenced in the function
    referenced_params = set()
    for node in ast.walk(validate_func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Check if it looks like a parameter (all caps)
            if node.id.isupper() and '_' in node.id:
                referenced_params.add(node.id)

    # Find where each parameter is defined
    param_definitions = {}
    for i, line in enumerate(source.split('\n'), 1):
        line = line.strip()
        for param in referenced_params:
            if line.startswith(f'{param} =') and i < validate_call_line:
                param_definitions[param] = i

    # Check that all referenced parameters are defined before the call
    undefined_params = []
    for param in referenced_params:
        if param not in param_definitions:
            undefined_params.append(param)

    if undefined_params:
        print(f"Parameters referenced in validate_trading_parameters but not defined before call: {undefined_params}")
        print(f"validate_trading_parameters called at line: {validate_call_line}")
        for param in referenced_params:
            if param in param_definitions:
                print(f"  {param} defined at line {param_definitions[param]}")
            else:
                print(f"  {param} NOT DEFINED before call")

    assert not undefined_params, f"Parameters {undefined_params} are referenced in validate_trading_parameters but not defined before the function call"


def test_buy_threshold_definition_order():
    """Specific test to ensure BUY_THRESHOLD is defined before validate_trading_parameters call."""

    src_path = Path(__file__).resolve().parents[1] / 'bot_engine.py'
    source = src_path.read_text()
    lines = source.split('\n')

    buy_threshold_line = None
    validate_call_line = None

    for i, line in enumerate(lines, 1):
        if line.strip().startswith('BUY_THRESHOLD ='):
            buy_threshold_line = i
        elif 'validate_trading_parameters()' in line and not line.strip().startswith('#'):
            validate_call_line = i

    assert buy_threshold_line is not None, "BUY_THRESHOLD should be defined"
    assert validate_call_line is not None, "validate_trading_parameters should be called"
    assert buy_threshold_line < validate_call_line, \
        f"BUY_THRESHOLD (line {buy_threshold_line}) should be defined before validate_trading_parameters call (line {validate_call_line})"


if __name__ == "__main__":
    test_validate_trading_parameters_no_name_error()
    test_buy_threshold_definition_order()
    print("All tests passed!")
