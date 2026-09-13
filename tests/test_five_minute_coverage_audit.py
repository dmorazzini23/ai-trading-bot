import pandas as pd

from ai_trading.tools.five_minute_coverage import count_session


def test_exact_window_and_session_boundary():
    opening = pd.Timestamp('2024-07-03T13:30:00Z')
    closing = opening + pd.Timedelta(minutes=20)
    index = pd.date_range(opening, closing, freq='min', inclusive='left')
    result = count_session(index, opening, closing)
    assert result == dict(candidate_slots=3, session_boundary_excluded=1,
                          missing_slots=0, duplicate_slots=0, complete_slots=2)
    missing = count_session(index.delete(0), opening, closing)
    assert missing['missing_slots'] == 1
    assert missing['complete_slots'] == 1
    duplicate = count_session(index.append(index[:1]), opening, closing)
    assert duplicate['duplicate_slots'] == 1
    assert duplicate['complete_slots'] == 1


def test_no_observation_carry_forward():
    opening = pd.Timestamp('2024-01-02T14:30:00Z')
    closing = opening + pd.Timedelta(minutes=15)
    index = pd.date_range(opening, closing, freq='min', inclusive='left')
    # Remove the exact exit minute; later observations cannot replace it.
    result = count_session(index.delete(11), opening, closing)
    assert result['complete_slots'] == 0
    assert result['missing_slots'] == 1
    assert result['session_boundary_excluded'] == 1
