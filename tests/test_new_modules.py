import importlib
import runpy
import types
import sys
from datetime import datetime, date, timezone
import logging

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
import requests
import pickle
import socket
import os

import logger
import utils
import meta_learning


def reload_module(mod):
    return importlib.reload(mod)


@pytest.fixture(autouse=True)
def reload_utils_module():
    import importlib as _imp
    _imp.reload(utils)
    yield


def load_runner(monkeypatch):
    bot_mod = types.ModuleType("bot")
    bot_mod.main = lambda: None
    monkeypatch.setitem(sys.modules, "bot", bot_mod)
    import runner as r
    return reload_module(r)


# ------------------ logger tests ------------------

def test_get_rotating_handler_fallback(monkeypatch, tmp_path, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(logger, "RotatingFileHandler", lambda *a, **k: (_ for _ in ()).throw(OSError("fail")))
    h = logger.get_rotating_handler(str(tmp_path / "x.log"))
    assert isinstance(h, logging.StreamHandler)
    assert "Cannot open log file" in caplog.text


def test_setup_logging_idempotent(monkeypatch, tmp_path):
    mod = reload_module(logger)
    created = []

    def fake_get_rotating(path, **_):
        created.append(path)
        return logging.StreamHandler()

    monkeypatch.setattr(mod, "get_rotating_handler", fake_get_rotating)
    lg = mod.setup_logging(debug=True, log_file=str(tmp_path / "f.log"))
    assert lg.level == logging.DEBUG
    assert created
    created.clear()
    lg2 = mod.setup_logging(debug=False)
    assert lg2 is lg
    assert created == []


def test_get_logger():
    mod = reload_module(logger)
    root = mod.setup_logging(debug=True)
    lg = mod.get_logger("test")
    assert lg is mod._loggers["test"]
    assert len(lg.handlers) == len(root.handlers)


# ------------------ runner tests ------------------

def test_handle_signal_sets_shutdown(monkeypatch):
    mod = load_runner(monkeypatch)
    mod._shutdown = False
    mod._handle_signal(15, None)
    assert mod._shutdown


def test_run_forever_exit(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    mod._run_forever()


def test_run_forever_exception(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    with pytest.raises(ValueError):
        mod._run_forever()


# ------------------ main alias tests ------------------

def test_main_aliases(monkeypatch):
    run_mod = types.ModuleType("run")
    run_mod.create_flask_app = lambda: "app"
    run_mod.run_flask_app = lambda port: port
    run_mod.run_bot = lambda v, s: 0
    run_mod.validate_environment = lambda: None
    run_mod.main = lambda: "main"
    run_mod.__spec__ = importlib.util.spec_from_loader("run", loader=None)
    monkeypatch.setitem(sys.modules, "run", run_mod)
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)
    import main as main_mod
    main_mod = reload_module(main_mod)
    assert main_mod.create_flask_app() == "app"
    assert main_mod.main() == "main"


def test_main_executes_run(monkeypatch):
    run_mod = types.ModuleType("run")
    called = []
    run_mod.main = lambda: called.append(True)
    run_mod.create_flask_app = lambda: None
    run_mod.run_flask_app = lambda port: None
    run_mod.run_bot = lambda v, s: 0
    run_mod.validate_environment = lambda: None
    run_mod.__spec__ = importlib.util.spec_from_loader("run", loader=None)
    monkeypatch.setitem(sys.modules, "run", run_mod)
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)
    runpy.run_module("main", run_name="__main__")
    assert called == [True]


# ------------------ utils tests ------------------

def test_warn_limited(caplog):
    caplog.set_level("WARNING")
    utils._WARN_COUNTS.clear()
    for i in range(4):
        utils._warn_limited("k", "warn %d", i, limit=2)
    msgs = [rec.message for rec in caplog.records]
    assert "warn 0" in msgs[0]
    assert "warn 1" in msgs[1]
    assert "suppressed" in msgs[2]
    assert len(msgs) == 3


def test_to_serializable_mappingproxy():
    from types import MappingProxyType

    mp = MappingProxyType({"a": 1})
    res = utils.to_serializable({"x": mp, "l": [1, 2]})
    assert res == {"x": {"a": 1}, "l": [1, 2]}


def test_get_free_port():
    port = utils.get_free_port(start=9260, end=9260)
    assert port == 9260


def test_ensure_utc_variants():
    naive = datetime(2024, 1, 1, 12, 0)
    assert utils.ensure_utc(naive).tzinfo == timezone.utc
    aware = datetime(2024, 1, 1, 12, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.ensure_utc(aware).tzinfo == timezone.utc
    d = date(2024, 1, 1)
    assert utils.ensure_utc(d).tzinfo == timezone.utc


def test_is_market_open_with_calendar(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")

    class Cal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame({
                "market_open": [pd.Timestamp("2024-01-02 09:30", tz="US/Eastern")],
                "market_close": [pd.Timestamp("2024-01-02 16:00", tz="US/Eastern")],
            })

    mod.get_calendar = lambda name: Cal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024, 1, 2, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.is_market_open(now)


def test_is_market_open_fallback(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")
    mod.get_calendar = lambda name: (_ for _ in ()).throw(Exception("fail"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    weekend = datetime(2024, 1, 6, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(weekend)


def test_safe_to_datetime_various(caplog):
    caplog.set_level("WARNING")
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    assert utils.safe_to_datetime(("SYM", ts)) == ts
    arr = [("A", ts), ("B", ts)]
    idx = utils.safe_to_datetime(arr)
    assert list(idx) == [ts, ts]
    res = utils.safe_to_datetime(["bad"])
    assert res.isna().all()
    assert "coercing" in caplog.text


def test_get_latest_close_cases():
    assert utils.get_latest_close(None) == 1.0
    df = pd.DataFrame()
    assert utils.get_latest_close(df) == 1.0
    df = pd.DataFrame({"close": [0, np.nan]})
    assert utils.get_latest_close(df) == 1.0
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert utils.get_latest_close(df) == 2.0


def test_health_check_paths(monkeypatch):
    assert not utils.health_check(None, "m")
    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setenv("HEALTH_MIN_ROWS", "5")
    assert not utils.health_check(df, "m")
    df = pd.DataFrame({"a": range(10)})
    assert utils.health_check(df, "m")


def test_safe_get_column_warning(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"other": [1]})
    assert utils.get_open_column(df) is None
    assert "open price" in caplog.text


# ------------------ meta_learning tests ------------------

def test_load_weights_creates_default(tmp_path):
    p = tmp_path / "w.csv"
    arr = meta_learning.load_weights(str(p), default=np.array([1.0]))
    assert p.exists()
    assert np.allclose(arr, [1.0])


def test_update_weights_no_change(tmp_path):
    p = tmp_path / "w.csv"
    np.savetxt(p, np.array([0.1, 0.2]), delimiter=",")
    ok = meta_learning.update_weights(str(p), np.array([0.1, 0.2]), {"m": 1})
    assert ok is False


def test_update_signal_weights_normal():
    w = {"a": 1.0, "b": 1.0}
    perf = {"a": 1.0, "b": 3.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert round(res["b"], 2) == 0.75


def test_save_and_load_checkpoint(tmp_path):
    path = tmp_path / "m.pkl"
    meta_learning.save_model_checkpoint({"x": 1}, str(path))
    obj = meta_learning.load_model_checkpoint(str(path))
    assert obj == {"x": 1}


def test_optimize_signals(monkeypatch):
    m = types.SimpleNamespace(predict=lambda X: [0] * len(X))
    data = [1, 2]
    cfg = types.SimpleNamespace(MODEL_PATH="")
    assert meta_learning.optimize_signals(data, cfg, model=m) == [0, 0]
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: None)
    assert meta_learning.optimize_signals(data, cfg, model=None) == data


def test_retrain_meta_missing(tmp_path):
    assert not meta_learning.retrain_meta_learner(str(tmp_path / "no.csv"))

# Additional runner tests for error branches

def test_run_forever_request_exception(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(requests.exceptions.RequestException("boom")))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    with pytest.raises(requests.exceptions.RequestException):
        mod._run_forever()


def test_run_forever_system_exit_nonzero(monkeypatch):
    mod = load_runner(monkeypatch)
    seq = [SystemExit(1), SystemExit(0)]
    def side():
        exc = seq.pop(0)
        raise exc
    monkeypatch.setattr(mod, "main", side)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    mod._run_forever()
    assert not seq

# ------------------ utils additional tests ------------------

def test_log_warning(caplog):
    caplog.set_level("WARNING")
    utils.log_warning("msg", exc=ValueError("boom"), extra={"a":1})
    assert "boom" in caplog.text


def test_get_column_validation_errors():
    df = pd.DataFrame({"a":[1,2]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["b"], "b")
    df = pd.DataFrame({"c":[1,2]})
    with pytest.raises(TypeError):
        utils.get_column(df, ["c"], "c", dtype="datetime64[ns]")


def test_safe_get_column_and_ohlcv():
    df = pd.DataFrame({"Open":[1],"High":[2],"Low":[0],"Close":[1],"Volume":[10]})
    assert utils.get_open_column(df) == "Open"
    assert utils.get_ohlcv_columns(df) == ["Open","High","Low","Close","Volume"]


def test_pre_trade_health_check(monkeypatch):
    called=[]
    monkeypatch.setattr(utils, "check_symbol", lambda s,a: called.append(s) or True)
    res = utils.pre_trade_health_check(["A","B"], api=None)
    assert res == {"A":True,"B":True}
    assert called == ["A","B"]

# ------------------ meta_learning additional tests ------------------

def test_update_weights_history_error(tmp_path):
    w = tmp_path/"w.csv"
    h = tmp_path/"hist.json"
    np.savetxt(w, np.array([0.1]), delimiter=",")
    h.write_text("{bad")
    assert meta_learning.update_weights(str(w), np.array([0.2]), {"m":1}, str(h))


def test_load_weights_corrupted(tmp_path):
    p = tmp_path/"w.csv"
    p.write_text("bad,data")
    arr = meta_learning.load_weights(str(p), default=np.array([0.5]))
    assert arr.tolist() == [0.5]


def test_retrain_meta_learner_success(monkeypatch, tmp_path):
    data = tmp_path/"trades.csv"
    df = pd.DataFrame({
        "entry_price":[1,2,3,4],
        "exit_price":[2,3,4,5],
        "signal_tags":["a","a+b","b","a"],
        "side":["buy","sell","buy","sell"],
    })
    df.to_csv(data, index=False)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda m,p: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: [])
    class DummyModel:
        def fit(self,X,y,sample_weight=None):
            self.fitted=True
        def predict(self,X):
            return [0]*len(X)
    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: DummyModel())
    ok = meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"hist.pkl"), min_samples=1)
    assert ok

def test_runner_as_main(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    runpy.run_module("runner", run_name="__main__")

# Additional utils coverage
def test_column_helpers(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({
        "Datetime": dates,
        "symbol": ["A","B","C"],
        "Return": [0.1,0.2,0.3],
        "indicator": [1,2,3],
        "OrderID": [1,2,3]
    })
    assert utils.get_datetime_column(df) == "Datetime"
    assert utils.get_symbol_column(df) == "symbol"
    assert utils.get_return_column(df) == "Return"
    assert utils.get_indicator_column(df, ["indicator"]) == "indicator"
    assert utils.get_order_column(df, "OrderID") == "OrderID"

# More meta_learning coverage

def test_update_weights_success(tmp_path):
    p = tmp_path/"w.csv"
    np.savetxt(p, np.array([0.1]), delimiter=",")
    h = tmp_path/"hist.json"
    res = meta_learning.update_weights(str(p), np.array([0.2]), {"m":1}, str(h))
    assert res


def test_load_model_checkpoint_missing(tmp_path):
    path = tmp_path/"x.pkl"
    assert meta_learning.load_model_checkpoint(str(path)) is None


def test_update_signal_weights_zero(caplog):
    caplog.set_level("WARNING")
    w={"a":1.0}
    perf={"a":0.0}
    res=meta_learning.update_signal_weights(w, perf)
    assert res == w
    assert "Total performance sum is zero" in caplog.text


def test_load_weights_existing(tmp_path):
    p = tmp_path/"w.csv"
    np.savetxt(p, np.array([0.5,0.6]), delimiter=",")
    arr = meta_learning.load_weights(str(p), default=np.array([1.0]))
    assert np.allclose(arr, [0.5,0.6])


def test_optimize_signals_failure(monkeypatch):
    class Bad:
        def predict(self, d):
            raise ValueError("x")
    cfg = types.SimpleNamespace(MODEL_PATH="")
    res = meta_learning.optimize_signals([1], cfg, model=Bad())
    assert res == [1]

def test_load_weights_default_zero(tmp_path):
    arr = meta_learning.load_weights(str(tmp_path/"none.csv"))
    assert arr.size == 0


def test_update_weights_failure(monkeypatch, tmp_path):
    p = tmp_path/"w.csv"
    monkeypatch.setattr(meta_learning.np, "savetxt", lambda *a, **k: (_ for _ in ()).throw(OSError("bad")))
    ok = meta_learning.update_weights(str(p), np.array([1.0]), {}, str(tmp_path/"h.json"))
    assert not ok


def test_update_signal_weights_empty(caplog):
    caplog.set_level("ERROR")
    assert meta_learning.update_signal_weights({}, {}) is None
    assert "Empty weights" in caplog.text


def test_update_signal_weights_norm_zero(caplog):
    caplog.set_level("WARNING")
    w = {"a": 0.0}
    perf = {"a": 1.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert res == w
    assert "Normalization factor zero" in caplog.text


def test_retrain_meta_insufficient(monkeypatch, tmp_path):
    data = tmp_path/"t.csv"
    pd.DataFrame({"entry_price":[1],"exit_price":[2],"signal_tags":["a"],"side":["buy"]}).to_csv(data, index=False)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    assert not meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"hist.pkl"), min_samples=5)


def test_retrain_meta_training_fail(monkeypatch, tmp_path):
    data = tmp_path/"t.csv"
    pd.DataFrame({
        "entry_price":[1,2],
        "exit_price":[2,3],
        "signal_tags":["a","b"],
        "side":["buy","sell"],
    }).to_csv(data, index=False)
    class Bad:
        def fit(self,X,y,sample_weight=None):
            raise RuntimeError("boom")
    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: Bad())
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    assert not meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"hist.pkl"), min_samples=1)


def test_retrain_meta_load_history(monkeypatch, tmp_path):
    data = tmp_path/"t.csv"
    pd.DataFrame({
        "entry_price":[1,2],"exit_price":[2,3],"signal_tags":["a","b"],"side":["buy","buy"]
    }).to_csv(data, index=False)
    hist = tmp_path/"hist.pkl"
    pickle.dump([], open(hist, "wb"))
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: types.SimpleNamespace(fit=lambda X,y,sample_weight=None: None, predict=lambda X:[0]*len(X)))
    ok = meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(hist), min_samples=1)
    assert ok

def test_runner_import_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "bot", None)
    bot_engine_mod = types.ModuleType("bot_engine")
    bot_engine_mod.main = lambda: None
    monkeypatch.setitem(sys.modules, "bot_engine", bot_engine_mod)
    import importlib
    r = importlib.reload(importlib.import_module("runner"))
    assert r.main is bot_engine_mod.main

# More utils edge cases

def test_log_warning_no_exc(caplog):
    caplog.set_level("WARNING")
    utils.log_warning("plain")
    assert "plain" in caplog.text


def test_callable_lock_methods():
    lock = utils._CallableLock()
    assert not lock.locked()
    lock.acquire()
    assert lock.locked()
    lock.release()
    with lock():
        assert lock.locked()
    assert not lock.locked()


def test_get_latest_close_no_column():
    df = pd.DataFrame({"open":[1]})
    assert utils.get_latest_close(df) == 1.0


def test_is_market_open_holiday(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")
    class Cal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame()
    mod.get_calendar=lambda name: Cal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024,1,1,10,0,tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(now)


def test_ensure_utc_type_error():
    with pytest.raises(AssertionError):
        utils.ensure_utc(1)


def test_get_free_port_none(monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    # occupy port so function cannot bind
    monkeypatch.setattr(socket, "socket", lambda *a, **k: socket.socket(*a, **k))
    try:
        res = utils.get_free_port(start=port, end=port-1)
        assert res is None
    finally:
        sock.close()


def test_safe_to_datetime_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    class Bad:
        pass
    monkeypatch.setattr(pd, "to_datetime", lambda *a, **k: (_ for _ in ()).throw(TypeError("bad")))
    res = utils.safe_to_datetime([Bad()])
    assert res.isna().all()
    assert "failed" in caplog.text


def test_health_check_empty(caplog, monkeypatch):
    caplog.set_level("CRITICAL")
    monkeypatch.setenv("HEALTH_MIN_ROWS", "1")
    df = pd.DataFrame()
    assert not utils.health_check(df, "d")
    assert "empty dataset" in caplog.text


def test_get_column_errors():
    dates = pd.to_datetime(["2024-01-02","2024-01-01","2024-01-03"])
    df = pd.DataFrame({"d": dates})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_monotonic=True)
    df = pd.DataFrame({"d": [pd.NaT, pd.NaT, pd.NaT]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_non_null=True)
    df = pd.DataFrame({"d": [1,1,2]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_unique=True)
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=3, tz="UTC")})
    df["d"] = df["d"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_timezone_aware=True)


def test_safe_get_column_non_df():
    assert utils._safe_get_column([], ["x"], "lbl") is None


def test_get_ohlcv_columns_missing():
    df = pd.DataFrame({"Open":[1],"High":[2]})
    assert utils.get_ohlcv_columns(df) == []


def test_check_symbol_failure(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: (_ for _ in ()).throw(IOError("bad")))
    assert not utils.check_symbol("A", api=None)

def test_is_market_open_attr_error(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024,1,6,10,0,tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(now)


def test_check_symbol_success(monkeypatch, tmp_path):
    file = tmp_path/"A.csv"
    pd.DataFrame({"close":[1,2,3]}).to_csv(file, index=False)
    monkeypatch.setattr(os.path, "join", lambda *a: str(file))
    monkeypatch.setenv("HEALTH_MIN_ROWS", "1")
    assert utils.check_symbol("A", api=None)
