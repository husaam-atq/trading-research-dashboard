from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown_duration(returns: pd.Series) -> int:
    equity = equity_curve(returns)
    dd = drawdown(equity)
    longest = 0
    current = 0
    for value in dd:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _cagr(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    total = float((1.0 + clean).prod())
    years = len(clean) / 252.0
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {key: 0.0 for key in METRIC_ORDER}

    equity = equity_curve(clean)
    dd = drawdown(equity)
    cagr = _cagr(clean)
    ann_vol = float(clean.std(ddof=0) * np.sqrt(252))
    downside = clean[clean < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(252)) if len(downside) > 1 else 0.0
    max_dd = float(dd.min())
    monthly = clean.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    var_95 = float(clean.quantile(0.05))
    tail = clean[clean <= var_95]

    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": float((cagr - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0,
        "sortino_ratio": float((cagr - risk_free_rate) / downside_vol) if downside_vol > 0 else 0.0,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
        "hit_rate": float((clean > 0).mean()),
        "monthly_win_rate": float((monthly > 0).mean()) if not monthly.empty else 0.0,
        "best_month": float(monthly.max()) if not monthly.empty else 0.0,
        "worst_month": float(monthly.min()) if not monthly.empty else 0.0,
        "skewness": float(clean.skew()) if len(clean) > 2 else 0.0,
        "kurtosis": float(clean.kurtosis()) if len(clean) > 3 else 0.0,
        "var_95": var_95,
        "expected_shortfall_95": float(tail.mean()) if not tail.empty else var_95,
        "max_drawdown_duration": float(max_drawdown_duration(clean)),
    }


METRIC_ORDER = [
    "total_return",
    "cagr",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "hit_rate",
    "monthly_win_rate",
    "best_month",
    "worst_month",
    "skewness",
    "kurtosis",
    "var_95",
    "expected_shortfall_95",
    "max_drawdown_duration",
]


def benchmark_relative_metrics(returns: pd.Series, spy_returns: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    metrics = performance_metrics(clean)
    out = {
        "total_return": metrics["total_return"],
        "cagr": metrics["cagr"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "best_month": metrics["best_month"],
        "worst_month": metrics["worst_month"],
        "monthly_win_rate": metrics["monthly_win_rate"],
        "max_drawdown_duration": metrics["max_drawdown_duration"],
        "alpha_vs_cash": metrics["cagr"],
        "information_ratio_vs_cash": metrics["sharpe_ratio"],
        "beta_to_spy": 0.0,
        "correlation_to_spy": 0.0,
        "alpha_vs_spy": metrics["cagr"],
        "information_ratio_vs_spy": 0.0,
    }
    if spy_returns is not None:
        aligned = pd.concat([clean, spy_returns], axis=1).dropna()
        if len(aligned) > 2:
            aligned.columns = ["strategy", "spy"]
            spy_var = float(aligned["spy"].var(ddof=0))
            beta = float(aligned["strategy"].cov(aligned["spy"]) / spy_var) if spy_var > 0 else 0.0
            correlation = float(aligned["strategy"].corr(aligned["spy"]))
            spy_cagr = _cagr(aligned["spy"])
            excess = aligned["strategy"] - aligned["spy"]
            tracking = float(excess.std(ddof=0) * np.sqrt(252))
            out.update(
                {
                    "beta_to_spy": beta,
                    "correlation_to_spy": correlation,
                    "alpha_vs_spy": metrics["cagr"] - beta * spy_cagr,
                    "information_ratio_vs_spy": float(_cagr(excess) / tracking) if tracking > 0 else 0.0,
                }
            )
    return out


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"
