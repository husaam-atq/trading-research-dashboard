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
) -> pd.Series:
    """Generate end-of-day target positions from spread z-scores."""
    if exit_threshold >= entry_threshold:
        raise ValueError("Exit threshold must be lower than entry threshold.")
    if stop_threshold <= entry_threshold:
        raise ValueError("Stop threshold must be higher than entry threshold.")

    current_position = 0
    positions: list[int] = []

    for z in zscore:
        if pd.isna(z):
            positions.append(current_position)
            continue

        if current_position == 0:
            if abs(z) >= stop_threshold:
                current_position = 0
            elif z <= -entry_threshold:
                current_position = 1
            elif z >= entry_threshold:
                current_position = -1
        elif current_position == 1:
            if abs(z) <= exit_threshold or abs(z) >= stop_threshold:
                current_position = 0
        elif current_position == -1:
            if abs(z) <= exit_threshold or abs(z) >= stop_threshold:
                current_position = 0

        positions.append(current_position)

    return pd.Series(positions, index=zscore.index, name="position")
