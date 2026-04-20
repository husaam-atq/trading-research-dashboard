# core.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairStats:
    hedge_ratio: float
    intercept: float
    coint_pvalue: float
    adf_pvalue: float
    half_life: float
    spread: pd.Series


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """
    OLS: y_t = alpha + beta * x_t + e_t
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    intercept = float(model.params.iloc[0])
    hedge_ratio = float(model.params.iloc[1])
    return hedge_ratio, intercept


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float
) -> pd.Series:
    spread = y - (intercept + hedge_ratio * x)
    spread.name = "spread"
    return spread


def estimate_half_life(spread: pd.Series) -> float:
    """
    OU-style half-life approximation from:
    Δs_t = a + b*s_{t-1} + e_t
    """
    s = spread.dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()

    common_index = lagged.index.intersection(delta.index)
    lagged = lagged.loc[common_index]
    delta = delta.loc[common_index]

    if len(common_index) < 20:
        return np.nan

    x = sm.add_constant(lagged)
    model = sm.OLS(delta, x).fit()
    b = float(model.params.iloc[1])

    if b >= 0:
        return np.nan

    half_life = -np.log(2) / b
    if np.isnan(half_life) or np.isinf(half_life):
        return np.nan

    return float(half_life)


def analyse_pair(y: pd.Series, x: pd.Series) -> PairStats:
    """
    Compute cointegration, hedge ratio, spread, ADF, half-life.
    """
    coint_stat, coint_pvalue, _ = coint(y, x, autolag="AIC")
    hedge_ratio, intercept = estimate_hedge_ratio(y, x)
    spread = compute_spread(y, x, hedge_ratio, intercept)
    adf_pvalue = float(adfuller(spread.dropna(), autolag="AIC")[1])
    half_life = estimate_half_life(spread)

    return PairStats(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        coint_pvalue=float(coint_pvalue),
        adf_pvalue=adf_pvalue,
        half_life=half_life,
        spread=spread
    )


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean_ = series.rolling(window=window).mean()
    std_ = series.rolling(window=window).std(ddof=0)
    z = (series - mean_) / std_.replace(0.0, np.nan)
    z.name = "zscore"
    return z


def annualised_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return 0.0
    total = (1.0 + r).prod()
    n = len(r)
    return float(total ** (252 / n) - 1.0)


def annualised_volatility(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(252))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    vol = annualised_volatility(daily_returns)
    if vol == 0:
        return 0.0
    ret = annualised_return(daily_returns)
    return float((ret - risk_free_rate) / vol)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(annualised_return(daily_returns) / mdd)


def cap_weights(weights: pd.Series, max_pair_weight: float) -> pd.Series:
    """
    Cap individual weights then renormalise.
    """
    if weights.empty:
        return weights

    w = weights.clip(lower=0.0, upper=max_pair_weight)
    total = w.sum()
    if total <= 0:
        return pd.Series(0.0, index=weights.index)
    return w / total


def realised_volatility(daily_returns: pd.Series, min_vol: float = 1e-6) -> float:
    vol = annualised_volatility(daily_returns)
    return max(vol, min_vol)