import json

def test_best_hyperparams_sensible():
    with open("best_hyperparams.json") as f:
        params = json.load(f)
    assert 1 <= params.get("fast_period", 0) < params.get("slow_period", 100), \
        "Fast period should be < slow period"
    assert params.get("signal_period", 0) > 0, "Signal period must be positive"
