from __future__ import annotations

import pandas as pd

from src.signals import generate_positions_with_reasons, rolling_zscore


def test_stop_exit_and_cooldown_are_deterministic() -> None:
    index = pd.bdate_range("2024-01-01", periods=8)
    zscore = pd.Series([0.0, 2.1, 3.2, 2.2, 2.1, 2.1, 0.2, 0.0], index=index)
    position, reasons = generate_positions_with_reasons(
        zscore,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_threshold=3.0,
        cooldown_days=2,
    )
    assert position.iloc[1] == -1
    assert position.iloc[2] == 0
    assert reasons.iloc[2] == "stop_loss"
    assert position.iloc[3] == 0
    assert position.iloc[4] == 0


def test_rolling_zscore_does_not_change_when_future_changes() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    series = pd.Series(range(100), index=index, dtype=float)
    changed = series.copy()
    changed.iloc[80:] += 1000.0
    pd.testing.assert_series_equal(rolling_zscore(series, 20).iloc[:80], rolling_zscore(changed, 20).iloc[:80])
