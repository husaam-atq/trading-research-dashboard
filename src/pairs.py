from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from .signals import count_threshold_crossings, rolling_zscore


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Estimate y = alpha + beta * x with OLS."""
    frame = pd.concat([y, x], axis=1).dropna()
    if frame.shape[0] < 30:
        raise ValueError("Not enough observations to estimate hedge ratio.")
    model = sm.OLS(frame.iloc[:, 0], sm.add_constant(frame.iloc[:, 1])).fit()
    intercept = float(model.params.iloc[0])
    hedge_ratio = float(model.params.iloc[1])
    return hedge_ratio, intercept


def calculate_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float,
) -> pd.Series:
    spread = y - (intercept + hedge_ratio * x)
    spread.name = "spread"
    return spread


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from delta_spread = a + b * lag_spread."""
    clean = spread.dropna()
    lagged = clean.shift(1)
    delta = clean.diff()
    frame = pd.concat([delta, lagged], axis=1).dropna()
    if len(frame) < 30:
        return np.nan
    model = sm.OLS(frame.iloc[:, 0], sm.add_constant(frame.iloc[:, 1])).fit()
    beta = float(model.params.iloc[1])
    if beta >= 0:
        return np.nan
    half_life = -np.log(2.0) / beta
    if not np.isfinite(half_life) or half_life <= 0:
        return np.nan
    return float(half_life)


def analyse_pair(y: pd.Series, x: pd.Series, zscore_window: int = 60, entry_threshold: float = 2.0) -> dict[str, float]:
    """Calculate correlation, Engle-Granger p-value, hedge ratio, and spread stats."""
    frame = pd.concat([y, x], axis=1).dropna()
    log_frame = np.log(frame)
    log_y = log_frame.iloc[:, 0]
    log_x = log_frame.iloc[:, 1]
    corr = float(log_y.corr(log_x))
    hedge_ratio, intercept = estimate_hedge_ratio(log_y, log_x)
    spread = calculate_spread(log_y, log_x, hedge_ratio, intercept)
    zscore = rolling_zscore(spread, zscore_window)

    try:
        _, pvalue, _ = coint(log_y, log_x, autolag="AIC")
        coint_pvalue = float(pvalue)
    except Exception:
        coint_pvalue = np.nan

    half_life = estimate_half_life(spread)
    return {
        "correlation": corr,
        "coint_pvalue": coint_pvalue,
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "half_life": half_life,
        "threshold_crossings": float(count_threshold_crossings(zscore, entry_threshold)),
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std(ddof=0)),
        "observations": int(len(spread)),
    }


def screen_pairs(
    prices: pd.DataFrame,
    peer_group: str = "default",
    min_abs_correlation: float = 0.80,
    max_coint_pvalue: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    min_threshold_crossings: int = 4,
    zscore_window: int = 60,
    entry_threshold: float = 2.0,
) -> pd.DataFrame:
    """Screen all pair combinations using log-price correlation and cointegration."""
    records: list[dict[str, float | str | bool]] = []

    for ticker_y, ticker_x in combinations(prices.columns, 2):
        pair_prices = prices[[ticker_y, ticker_x]].dropna()
        if len(pair_prices) < 252:
            continue
        try:
            stats = analyse_pair(pair_prices[ticker_y], pair_prices[ticker_x], zscore_window, entry_threshold)
        except Exception:
            continue

        passed_correlation = abs(float(stats["correlation"])) >= min_abs_correlation
        pvalue = float(stats["coint_pvalue"]) if pd.notna(stats["coint_pvalue"]) else np.nan
        passed_cointegration = pd.notna(pvalue) and pvalue <= max_coint_pvalue
        half_life = float(stats["half_life"]) if pd.notna(stats["half_life"]) else np.nan
        passed_half_life = pd.notna(half_life) and min_half_life <= half_life <= max_half_life
        crossings = int(stats["threshold_crossings"])
        passed_crossings = crossings >= min_threshold_crossings
        records.append(
            {
                "peer_group": peer_group,
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                **stats,
                "passed_correlation": bool(passed_correlation),
                "passed_cointegration": bool(passed_cointegration),
                "passed_half_life": bool(passed_half_life),
                "passed_crossings": bool(passed_crossings),
                "selected_candidate": bool(
                    passed_correlation and passed_cointegration and passed_half_life and passed_crossings
                ),
            }
        )

    if not records:
        raise ValueError("No pairs could be screened.")

    results = pd.DataFrame(records)
    results["abs_correlation"] = results["correlation"].abs()
    results = results.sort_values(
        ["selected_candidate", "coint_pvalue", "half_life", "abs_correlation"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return results


def screen_peer_groups(
    prices: pd.DataFrame,
    peer_groups: dict[str, list[str]],
    **kwargs: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for peer_group, tickers in peer_groups.items():
        available = [ticker for ticker in tickers if ticker in prices.columns]
        if len(available) < 2:
            continue
        group_prices = prices[available].dropna()
        try:
            frames.append(screen_pairs(group_prices, peer_group=peer_group, **kwargs))
        except ValueError:
            continue
    if not frames:
        raise ValueError("No peer groups produced screenable pairs.")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["selected_candidate", "coint_pvalue", "half_life", "abs_correlation"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def choose_pairs(screening_results: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Choose top pairs, falling back to best diagnostics if strict filters are too restrictive."""
    passed = screening_results[screening_results["selected_candidate"]].copy()
    if not passed.empty:
        selected = passed.head(top_n).copy()
        selected["selection_method"] = "strict_peer_group_filters"
        return selected

    fallback = screening_results.sort_values(
        ["passed_correlation", "passed_cointegration", "passed_half_life", "passed_crossings", "coint_pvalue"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).head(top_n).copy()
    fallback["selection_method"] = "diagnostic_fallback_no_strict_survivors"
    return fallback
