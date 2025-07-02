import json
import os
import pytest
import bot_engine

@pytest.mark.smoke
def test_save_and_load_snapshot(tmp_path):
    fpath = tmp_path / "portfolio_snapshot.json"
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)

    portfolio = {"AAPL": 10, "GOOGL": 5}
    bot_engine.save_portfolio_snapshot(portfolio)

    assert os.path.exists("portfolio_snapshot.json")
    data = json.load(open("portfolio_snapshot.json"))
    assert data["positions"] == portfolio

    loaded = bot_engine.load_portfolio_snapshot()
    assert loaded == portfolio

    os.chdir(orig_cwd)
