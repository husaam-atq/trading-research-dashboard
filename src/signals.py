from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0.0, np.nan)
    zscore = (series - mean) / std
    zscore.name = "zscore"
    return zscore


def fixed_zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index, name="zscore")
    zscore = (series - mean) / std
    zscore.name = "zscore"
    return zscore


def generate_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.0,
    can_enter: pd.Series | None = None,
    max_holding_period: int = 20,
    cooldown_days: int = 0,
) -> pd.Series:
    """Generate end-of-day target positions from spread z-scores."""
    positions, _ = generate_positions_with_reasons(
        zscore=zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
        can_enter=can_enter,
        max_holding_period=max_holding_period,
        cooldown_days=cooldown_days,
    )
    return positions


def generate_positions_with_reasons(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.0,
    can_enter: pd.Series | None = None,
    max_holding_period: int = 20,
    cooldown_days: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Generate target positions and exit reasons from end-of-day z-scores."""
    if exit_threshold >= entry_threshold:
        raise ValueError("Exit threshold must be lower than entry threshold.")
    if stop_threshold <= entry_threshold:
        raise ValueError("Stop threshold must be higher than entry threshold.")

    if can_enter is None:
        can_enter = pd.Series(True, index=zscore.index)
    can_enter = can_enter.reindex(zscore.index).fillna(False)

    current_position = 0
    holding_period = 0
    cooldown_remaining = 0
    positions: list[int] = []
    exit_reasons: list[str] = []

    for date, z in zscore.items():
        exit_reason = ""
        if pd.isna(z):
            positions.append(current_position)
            exit_reasons.append(exit_reason)
            if current_position != 0:
                holding_period += 1
            continue

        if current_position == 0:
            holding_period = 0
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif bool(can_enter.loc[date]) and z <= -entry_threshold and abs(z) < stop_threshold:
                current_position = 1
                holding_period = 1
            elif bool(can_enter.loc[date]) and z >= entry_threshold and abs(z) < stop_threshold:
                current_position = -1
                holding_period = 1
        elif current_position == 1:
            holding_period += 1
            if abs(z) >= stop_threshold:
                current_position = 0
                exit_reason = "stop_loss"
                holding_period = 0
                cooldown_remaining = cooldown_days
            elif abs(z) <= exit_threshold:
                current_position = 0
                exit_reason = "mean_reversion"
                holding_period = 0
            elif holding_period >= max_holding_period:
                current_position = 0
                exit_reason = "time_stop"
                holding_period = 0
        elif current_position == -1:
            holding_period += 1
            if abs(z) >= stop_threshold:
                current_position = 0
                exit_reason = "stop_loss"
                holding_period = 0
                cooldown_remaining = cooldown_days
            elif abs(z) <= exit_threshold:
                current_position = 0
                exit_reason = "mean_reversion"
                holding_period = 0
            elif holding_period >= max_holding_period:
                current_position = 0
                exit_reason = "time_stop"
                holding_period = 0

        positions.append(current_position)
        exit_reasons.append(exit_reason)

    return (
        pd.Series(positions, index=zscore.index, name="position"),
        pd.Series(exit_reasons, index=zscore.index, name="exit_reason"),
    )


def count_threshold_crossings(zscore: pd.Series, entry_threshold: float) -> int:
    clean = zscore.dropna()
    if clean.empty:
        return 0
    long_crosses = (clean.shift(1) > -entry_threshold) & (clean <= -entry_threshold)
    short_crosses = (clean.shift(1) < entry_threshold) & (clean >= entry_threshold)
    return int((long_crosses | short_crosses).sum())
