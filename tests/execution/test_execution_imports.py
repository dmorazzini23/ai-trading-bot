from importlib import import_module


def test_execution_algorithms_and_result_importable():
    """Ensure algorithms submodule and ExecutionResult can be imported."""
    algos = import_module("ai_trading.execution.algorithms")
    from ai_trading.execution import ExecutionResult

    assert algos is not None
    assert ExecutionResult is not None

