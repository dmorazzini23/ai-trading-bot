import builtins
import importlib
import logging
import runpy
import signal
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import alerts
import config
import meta_learning
import ml_model
import risk_engine
import ai_trading.__main__ as main
import utils
import validate_env
from strategies.mean_reversion import MeanReversionStrategy


def test_alert_no_webhook(monkeypatch):
    """send_slack_alert returns early when webhook unset."""
    monkeypatch.setattr(alerts, "SLACK_WEBHOOK", "")
    called = []
    monkeypatch.setattr(alerts.requests, "post", lambda *a, **k: called.append(1))
    alerts.send_slack_alert("msg")
    assert not called


def test_config_missing_vars(monkeypatch):
    """_require_env_vars raises when variables missing."""
    with pytest.raises(RuntimeError):
        config._require_env_vars("MISSING_VAR")


def test_get_env_reload(monkeypatch):
    """get_env reloads environment when requested."""
    called = []
    monkeypatch.setattr(config, "reload_env", lambda: called.append(True))
    with pytest.raises(RuntimeError):
        config.get_env("MISSING", required=True, reload=True)
    assert called == [True]


def test_create_flask_routes():
    """Health endpoints respond correctly."""
    flask_mod = types.ModuleType("flask")
    class Flask:
        def __init__(self, *a, **k):
            pass
        def route(self, *a, **k):
            def deco(f):
                return f
            return deco
        def run(self, *a, **k):
            pass
        def test_client(self):
            return types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(status_code=200, json=lambda: {"status": "ok"}))
    flask_mod.Flask = Flask
    flask_mod.jsonify = lambda **kw: kw
    class DummyClient:
        def __init__(self, *a, **k):
            pass
    flask_mod.testing = types.SimpleNamespace(FlaskClient=DummyClient)
    sys.modules['flask'] = flask_mod
    sys.modules['flask.testing'] = types.ModuleType('flask.testing')
    sys.modules['flask.testing'].FlaskClient = DummyClient
    sys.modules.pop('ai_trading.__main__', None)
    import importlib
    main_mod = importlib.import_module('ai_trading.__main__')
    app = main_mod.create_flask_app()
    client = app.test_client()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/healthz").status_code == 200


def test_main_serve_api(monkeypatch):
    """main launches Flask thread and bot when --serve-api used."""
    monkeypatch.setattr(sys, "argv", ["ai_trading", "--serve-api"])
    monkeypatch.setattr(main, "run_bot", lambda v, s: 0)
    called = {}

    class DummyThread:
        def __init__(self, target, args=(), daemon=None):
            self.target = target
            self.args = args
            called["created"] = True

        def start(self):
            called["started"] = True
            self.target(*self.args)

    monkeypatch.setattr(main, "run_flask_app", lambda port: called.setdefault("port", port))
    monkeypatch.setattr(main.threading, "Event", lambda: types.SimpleNamespace(set=lambda: called.setdefault("set", True)))
    monkeypatch.setattr(main.threading, "Thread", DummyThread)
    monkeypatch.setattr(main, "setup_logging", lambda *a, **k: logging.getLogger("t"))
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(main, "validate_environment", lambda: None)
    exit_codes = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exit_codes.append(code))
    handlers = {}
    monkeypatch.setattr(main.signal, "signal", lambda sig, func: handlers.setdefault(sig, func))

    main.main()
    assert called["started"] and called["port"] == 9000
    handlers[signal.SIGINT](signal.SIGINT, None)
    assert called.get("set")
    assert exit_codes == [0]


def test_meta_update_signal_weights():
    """Signal weights are normalized by performance."""
    w = {"a": 1.0, "b": 1.0}
    perf = {"a": 1.0, "b": 3.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert round(res["b"], 2) == 0.75


def test_meta_load_checkpoint_missing(tmp_path, caplog):
    """Missing checkpoint returns None with warning."""
    caplog.set_level("WARNING")
    path = tmp_path / "m.pkl"
    assert meta_learning.load_model_checkpoint(str(path)) is None
    assert "Checkpoint file missing" in caplog.text


def test_meta_retrain_missing_file(tmp_path):
    """retrain_meta_learner returns False when data file missing."""
    assert not meta_learning.retrain_meta_learner(str(tmp_path/"no.csv"))


def test_meta_retrain_insufficient(tmp_path):
    """retrain_meta_learner aborts when not enough rows."""
    data = tmp_path / "t.csv"
    pd.DataFrame({"entry_price":[1],"exit_price":[2],"signal_tags":["a"],"side":["buy"]}).to_csv(data, index=False)
    assert not meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"h.pkl"), min_samples=5)


def test_meta_update_weights_error(monkeypatch, tmp_path):
    """Errors saving weights are handled."""
    path = tmp_path / "w.csv"
    hist = tmp_path / "h.json"
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: False)
    monkeypatch.setattr(meta_learning.np, "savetxt", lambda *a, **k: (_ for _ in ()).throw(IOError("fail")))
    assert not meta_learning.update_weights(str(path), np.array([1.0]), {}, str(hist))


def test_mlmodel_validation_errors():
    """Validation checks raise appropriate errors."""
    model = ml_model.MLModel(types.SimpleNamespace(predict=lambda X: X, fit=lambda X,y: None))
    with pytest.raises(TypeError):
        model.predict([1])
    df = pd.DataFrame({"a":[np.nan]})
    with pytest.raises(ValueError):
        model.predict(df)
    df = pd.DataFrame({"a":["x"]})
    with pytest.raises(TypeError):
        model.predict(df)


def test_mlmodel_fit_predict_exceptions(monkeypatch):
    """Exceptions during fit and predict are propagated."""
    class Pipe:
        def fit(self, X, y):
            raise RuntimeError("boom")
        def predict(self, X):
            raise RuntimeError("boom")
    m = ml_model.MLModel(Pipe())
    df = pd.DataFrame({"a":[1.0]})
    with pytest.raises(RuntimeError):
        m.fit(df, [1])
    m.pipeline = Pipe()
    with pytest.raises(RuntimeError):
        m.predict(df)


def test_mlmodel_save_load_fail(monkeypatch, tmp_path):
    """Errors in save and load surface as exceptions."""
    m = ml_model.MLModel(types.SimpleNamespace())
    monkeypatch.setattr(ml_model.joblib, "dump", lambda *a, **k: (_ for _ in ()).throw(IOError("fail")))
    with pytest.raises(IOError):
        m.save(str(tmp_path/"m.pkl"))
    def bad_open(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(builtins, "open", bad_open)
    with pytest.raises(FileNotFoundError):
        ml_model.MLModel.load(str(tmp_path/"m.pkl"))


def test_train_and_predict_helpers():
    """train_model and predict_model basic paths."""
    model = ml_model.train_model([1,2,3], [1,2,3])
    preds = ml_model.predict_model(model, [[1],[2],[3]])
    assert len(preds) == 3
    with pytest.raises(ValueError):
        ml_model.predict_model(None, [1])
    with pytest.raises(ValueError):
        ml_model.predict_model(model, None)


def test_risk_engine_branches(monkeypatch):
    """Branches in position sizing and trading limits."""
    eng = risk_engine.RiskEngine()
    sig = risk_engine.TradeSignal(symbol="A", side="buy", confidence=1.0, strategy="s", weight=1.0, asset_class="equity")
    eng.strategy_limits["s"] = 0.5
    assert not eng.can_trade(sig)
    monkeypatch.setattr(eng, "check_max_drawdown", lambda api: False)
    assert eng.position_size(sig, 100, 10, api=object()) == 0
    monkeypatch.setattr(eng, "can_trade", lambda s: False)
    assert eng.position_size(sig, 100, 10) == 0
    eng.strategy_limits["s"] = 0.4
    eng.exposure["equity"] = 0.0
    assert eng._apply_weight_limits(sig) == 0.4
    res = risk_engine.calculate_position_size(sig, 100, 10)
    assert res == 10


def test_runner_main_loop(monkeypatch):
    """Runner exits on SystemExit 0 from bot.main."""
    bot_mod = types.ModuleType('bot')
    bot_mod.main = lambda: (_ for _ in ()).throw(SystemExit(0))
    sys.modules['bot'] = bot_mod
    module = runpy.run_module("runner", run_name="__main__")


def test_mean_reversion_nan_and_short(monkeypatch):
    """NaN close and negative z triggers branches."""
    strat = MeanReversionStrategy(lookback=1, z=1.0)
    class Fetcher:
        def get_daily_df(self, ctx, sym):
            return pd.DataFrame({"close":[np.nan]})
    ctx = types.SimpleNamespace(tickers=["A"], data_fetcher=Fetcher())
    assert strat.generate(ctx) == []
    class Fetcher2:
        def get_daily_df(self, ctx, sym):
            return pd.DataFrame({"close":[1.0, 1.0, 0.0]})
    strat2 = MeanReversionStrategy(lookback=3, z=1.0)
    ctx2 = types.SimpleNamespace(tickers=["A"], data_fetcher=Fetcher2())
    signals = strat2.generate(ctx2)
    assert signals and signals[0].side == "buy"


def test_utils_edge_cases(tmp_path):
    """Cover utility helper edge cases."""
    assert utils.get_latest_close(pd.DataFrame()) == 0.0
    df = pd.DataFrame({"close":[np.nan]})
    assert utils.get_latest_close(df) == 0.0
    mod = types.ModuleType("pandas_market_calendars")
    setattr(mod, "get_calendar", None)
    sys.modules["pandas_market_calendars"] = mod
    assert not utils.is_market_open(datetime(2024,1,6,10, tzinfo=utils.EASTERN_TZ))
    with pytest.raises(AssertionError):
        utils.ensure_utc(1)
    port = utils.get_free_port(utils.get_free_port())
    assert isinstance(port, int)
    assert utils.get_ohlcv_columns(pd.DataFrame({"x":[1]})) == []


def test_validate_env_main(monkeypatch):
    """Running validate_env as script calls _main."""
    runpy.run_module("validate_env", run_name="__main__")
