import pytest

from ai_trading.main import _update_interval_autotune_state as update


def step(state, utilization, **overrides):
    settings = dict(enabled=True, only_when_open=True, min_interval_s=10,
                    max_interval_s=20, over_streak_cycles=2, under_streak_cycles=2,
                    step_up_s=7, step_down_s=7, cooldown_cycles=1)
    settings.update(overrides.pop('settings', {}))
    return update(settings=settings, state=state, closed=overrides.pop('closed', False),
                  elapsed_ms=100 * utilization, budget_ms=100, over_budget=False,
                  cycle_index=1, **overrides)


def test_sustained_load_changes_interval_with_cooldown_and_bounds():
    state = {}
    assert step(state, 1.) == (10, None)
    interval, event = step(state, 1.)
    assert interval == 17 and event['action'] == 'raise'
    assert step(state, 1.) == (17, None)
    interval, event = step(state, 1.)
    assert interval == 20 and event['action'] == 'raise'
    assert step(state, 1.) == (20, None)
    assert step(state, 1.) == (20, None)
    assert step(state, .1) == (20, None)
    interval, event = step(state, .1)
    assert interval == 13 and event['action'] == 'lower'
    assert step(state, .1) == (13, None)
    interval, event = step(state, .1)
    assert interval == 10 and event['action'] == 'lower'
    assert step(state, .1) == (10, None)


@pytest.mark.parametrize('mode', ['disabled', 'closed', 'missing_elapsed', 'missing_budget', 'zero_budget'])
def test_unusable_measurements_never_adapt(mode):
    state = {'active_interval_s': 15, 'over_streak': 10}
    interval, event = update(settings={'enabled': mode != 'disabled', 'min_interval_s': 10,
        'max_interval_s': 20}, state=state, closed=mode == 'closed',
        elapsed_ms=None if mode == 'missing_elapsed' else 200,
        budget_ms=None if mode == 'missing_budget' else 0 if mode == 'zero_budget' else 100,
        over_budget=True, cycle_index=2)
    assert (interval, event) == (15, None)
    assert state['over_streak'] == 10


def test_middle_utilization_resets_streak_and_invalid_interval_clamps():
    state = {'active_interval_s': 'invalid', 'over_streak': 1, 'under_streak': 9}
    assert step(state, .8) == (10, None)
    assert state['over_streak'] == state['under_streak'] == 0
