from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


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


def analyse_pair(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """Calculate correlation, Engle-Granger p-value, hedge ratio, and spread stats."""
    frame = pd.concat([y, x], axis=1).dropna()
    log_frame = np.log(frame)
    log_y = log_frame.iloc[:, 0]
    log_x = log_frame.iloc[:, 1]
    corr = float(log_y.corr(log_x))
    hedge_ratio, intercept = estimate_hedge_ratio(log_y, log_x)
    spread = calculate_spread(log_y, log_x, hedge_ratio, intercept)

    try:
        _, pvalue, _ = coint(log_y, log_x, autolag="AIC")
        coint_pvalue = float(pvalue)
    except Exception:
        coint_pvalue = np.nan

    return {
        "correlation": corr,
        "coint_pvalue": coint_pvalue,
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std(ddof=0)),
        "observations": int(len(spread)),
    }


def screen_pairs(
    prices: pd.DataFrame,
    min_abs_correlation: float = 0.75,
    max_coint_pvalue: float = 0.10,
) -> pd.DataFrame:
    """Screen all pair combinations using log-price correlation and cointegration."""
    records: list[dict[str, float | str | bool]] = []

    for ticker_y, ticker_x in combinations(prices.columns, 2):
        pair_prices = prices[[ticker_y, ticker_x]].dropna()
        if len(pair_prices) < 252:
            continue
        try:
            stats = analyse_pair(pair_prices[ticker_y], pair_prices[ticker_x])
        except Exception:
            continue

        passed_correlation = abs(float(stats["correlation"])) >= min_abs_correlation
        pvalue = float(stats["coint_pvalue"]) if pd.notna(stats["coint_pvalue"]) else np.nan
        passed_cointegration = pd.notna(pvalue) and pvalue <= max_coint_pvalue
        records.append(
            {
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                **stats,
                "passed_correlation": bool(passed_correlation),
                "passed_cointegration": bool(passed_cointegration),
                "selected_candidate": bool(passed_correlation and passed_cointegration),
            }
        )

    if not records:
        raise ValueError("No pairs could be screened.")

    results = pd.DataFrame(records)
    results["abs_correlation"] = results["correlation"].abs()
    results = results.sort_values(
        ["selected_candidate", "coint_pvalue", "abs_correlation"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return results


def choose_pairs(screening_results: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Choose top pairs, falling back to correlation if no pair passes cointegration."""
    passed = screening_results[screening_results["selected_candidate"]].copy()
    if not passed.empty:
        selected = passed.head(top_n).copy()
        selected["selection_method"] = "correlation_and_cointegration"
        return selected

    fallback = screening_results.sort_values("abs_correlation", ascending=False).head(top_n).copy()
    fallback["selection_method"] = "correlation_fallback_no_cointegrated_pairs"
    return fallback
