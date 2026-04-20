# pair_selection.py

from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

from data_loader import get_pair_frame
from core import analyse_pair


def rank_pairs(
    prices: pd.DataFrame,
    tickers: list[str],
    train_ratio: float,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    min_observations: int
) -> pd.DataFrame:
    """
    Rank all valid pairs using only the training segment.
    """
    records: list[dict] = []

    for t1, t2 in combinations(tickers, 2):
        try:
            df = get_pair_frame(prices, t1, t2)
            if len(df) < min_observations:
                continue

            split_idx = int(len(df) * train_ratio)
            train_df = df.iloc[:split_idx].copy()

            if len(train_df) < min_observations:
                continue

            stats = analyse_pair(train_df["y"], train_df["x"])

            valid = (
                stats.coint_pvalue <= max_coint_pvalue
                and stats.adf_pvalue <= max_adf_pvalue
                and not np.isnan(stats.half_life)
                and min_half_life <= stats.half_life <= max_half_life
            )

            records.append({
                "ticker_y": t1,
                "ticker_x": t2,
                "n_obs_train": len(train_df),
                "hedge_ratio": stats.hedge_ratio,
                "intercept": stats.intercept,
                "coint_pvalue": stats.coint_pvalue,
                "adf_pvalue": stats.adf_pvalue,
                "half_life": stats.half_life,
                "valid": valid
            })
        except Exception:
            continue

    ranking = pd.DataFrame(records)
    if ranking.empty:
        raise ValueError("No pairs could be evaluated.")

    ranking = ranking.sort_values(
        by=["valid", "coint_pvalue", "adf_pvalue", "half_life"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    return ranking


def choose_top_pairs(ranking: pd.DataFrame, top_n_pairs: int) -> pd.DataFrame:
    valid = ranking[ranking["valid"]].copy()
    if valid.empty:
        raise ValueError("No valid pairs passed the filters. Loosen thresholds or change the universe.")
    return valid.head(top_n_pairs).reset_index(drop=True)