
from ai_trading.utils import workers


def teardown_module(module):
    workers.shutdown_all()


def test_get_executor_singleton():
    ex1 = workers.get_executor("alpha")
    ex2 = workers.get_executor("alpha")
    assert ex1 is ex2


def test_submit_and_map_background():
    fut = workers.submit_background("beta", lambda x: x + 1, 1)
    assert fut.result() == 2
    res = workers.map_background("gamma", lambda x: x * 2, [1, 2, 3])
    assert res == [2, 4, 6]
