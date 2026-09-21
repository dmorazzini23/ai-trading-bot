from datetime import UTC, datetime

import pytest

from ai_trading.execution import live_trading as lt


@pytest.mark.parametrize('hour,minute,stage', [(13, 35, 'opening'), (19, 45, 'closing'), (17, 0, 'regular')])
@pytest.mark.parametrize('existing', [True, False])
def test_session_overrides_only_tighten_thresholds(monkeypatch, hour, minute, stage, existing):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 21, hour, minute, tzinfo=UTC)
    monkeypatch.setattr(lt, 'datetime', Clock)
    monkeypatch.setattr(lt, '_market_is_open_now', lambda: True)
    monkeypatch.setattr(lt, '_resolve_bool_env', lambda key: True)
    def floats(key, default):
        if 'WINDOW_SEC' in key:
            return 1800.
        if key.endswith('MAX_SLIPPAGE_DRAG_BPS'):
            return 3.
        return 2.
    monkeypatch.setattr(lt, '_config_float', floats)
    monkeypatch.setattr(lt, '_config_int', lambda *a: 5)
    original = {'min_profit_factor': 3., 'min_win_rate': 1., 'max_slippage_drag_bps': 8., 'min_closed_trades': 8} if existing else {}
    before = dict(original)
    result, detail = lt.ExecutionEngine.__new__(lt.ExecutionEngine)._apply_session_runtime_gonogo_overrides(original)
    assert original == before
    assert detail['stage'] == stage
    if stage == 'regular':
        assert result == original and detail['applied'] == {}
    else:
        assert result['min_profit_factor'] == (3. if existing else 2.)
        assert result['min_closed_trades'] == (8 if existing else 5)
        assert result['max_slippage_drag_bps'] == 3.


@pytest.mark.parametrize('enabled,opened,reason', [(False, True, 'disabled'), (True, False, 'market_closed')])
def test_session_gates_preserve_base_thresholds_when_inactive(monkeypatch, enabled, opened, reason):
    monkeypatch.setattr(lt, '_resolve_bool_env', lambda key: enabled)
    monkeypatch.setattr(lt, '_market_is_open_now', lambda: opened)
    result, detail = lt.ExecutionEngine.__new__(lt.ExecutionEngine)._apply_session_runtime_gonogo_overrides({'min_profit_factor': 2.})
    assert result == {'min_profit_factor': 2.}
    assert detail['reason'] == reason
