import json
import os
import tempfile

from ai_trading.backtesting.grid_runner import grid_search, persist_artifacts


def test_grid_search_basic():
    """Test basic grid search functionality."""
    def evaluator(params):
        return {"sharpe": params.get("kelly", 0.5) * 2, "result": "ok"}

    grid = {"kelly": [0.3, 0.6], "lookback": [50, 100]}
    result = grid_search(evaluator, grid, n_jobs=1)

    assert result["count"] == 4  # 2 * 2 combinations
    assert len(result["results"]) == 4

    # Check that all combinations were tested
    kellys = [r["params"]["kelly"] for r in result["results"]]
    lookbacks = [r["params"]["lookback"] for r in result["results"]]
    assert set(kellys) == {0.3, 0.6}
    assert set(lookbacks) == {50, 100}

def test_persist_artifacts():
    """Test artifact persistence functionality."""
    def evaluator(params):
        return {"sharpe": 1.23, "calmar": 0.8}

    grid = {"kelly": [0.5]}
    run = grid_search(evaluator, grid, n_jobs=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = persist_artifacts(run, tmp_dir)

        # Check that directory was created
        assert os.path.exists(out_dir)
        assert "run_" in os.path.basename(out_dir)

        # Check that results file exists and is valid JSON
        results_file = os.path.join(out_dir, "results.json")
        assert os.path.exists(results_file)

        with open(results_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["count"] == 1
        assert len(saved_data["results"]) == 1
        assert saved_data["results"][0]["metrics"]["sharpe"] == 1.23

def test_grid_search_empty_grid():
    """Test grid search with empty parameter grid."""
    def evaluator(params):
        return {"result": "empty"}

    grid = {}
    result = grid_search(evaluator, grid, n_jobs=1)

    assert result["count"] == 1  # Should still run once with empty params
    assert len(result["results"]) == 1
    assert result["results"][0]["params"] == {}

def test_grid_search_single_param():
    """Test grid search with single parameter."""
    def evaluator(params):
        return {"value": params.get("x", 0) * 2}

    grid = {"x": [1, 2, 3]}
    result = grid_search(evaluator, grid, n_jobs=1)

    assert result["count"] == 3
    values = [r["metrics"]["value"] for r in result["results"]]
    assert sorted(values) == [2, 4, 6]
