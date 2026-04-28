from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import pair_leg_returns, summarize_trades, trade_metrics
from .metrics import performance_metrics
from .pairs import calculate_spread, estimate_hedge_ratio
from .signals import fixed_zscore, generate_positions


def walk_forward_pair(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.0,
    transaction_cost_bps: float = 5.0,
    train_window: int = 252,
    test_window: int = 63,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run rolling train/test walk-forward validation with no future parameters."""
    pair = prices[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    segments: list[pd.DataFrame] = []
    segment_id = 0

    start = 0
    while start + train_window + 5 < len(pair):
        train = pair.iloc[start : start + train_window]
        test = pair.iloc[start + train_window : start + train_window + test_window]
        if len(test) < 5:
            break

        log_train = np.log(train)
        log_test = np.log(test)
        hedge_ratio, intercept = estimate_hedge_ratio(log_train["y"], log_train["x"])
        train_spread = calculate_spread(log_train["y"], log_train["x"], hedge_ratio, intercept)
        test_spread = calculate_spread(log_test["y"], log_test["x"], hedge_ratio, intercept)
        zscore = fixed_zscore(test_spread, float(train_spread.mean()), float(train_spread.std(ddof=0)))
        position = generate_positions(zscore, entry_threshold, exit_threshold, stop_threshold)

        daily = pair_leg_returns(test, hedge_ratio, position, transaction_cost_bps)
        daily["spread"] = test_spread
        daily["zscore"] = zscore
        daily["segment"] = segment_id
        daily["train_start"] = train.index[0]
        daily["train_end"] = train.index[-1]
        daily["test_start"] = test.index[0]
        daily["test_end"] = test.index[-1]
        daily["hedge_ratio"] = hedge_ratio
        daily["intercept"] = intercept
        segments.append(daily)

        segment_id += 1
        start += test_window

    if not segments:
        raise ValueError(f"Not enough data for walk-forward validation: {ticker_y}/{ticker_x}")

    result = pd.concat(segments).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    trades = summarize_trades(result)
    metrics = performance_metrics(result["strategy_return"])
    metrics.update(trade_metrics(result, trades))
    metrics["segments"] = float(segment_id)
    return result, metrics


def walk_forward_many(
    prices: pd.DataFrame,
    pairs: pd.DataFrame,
    **kwargs: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, float | str]] = []

    for row in pairs.itertuples(index=False):
        ticker_y = row.ticker_y
        ticker_x = row.ticker_x
        daily, metrics = walk_forward_pair(prices, ticker_y, ticker_x, **kwargs)
        pair_name = f"{ticker_y}/{ticker_x}"
        pair_daily = daily[["strategy_return"]].rename(columns={"strategy_return": pair_name})
        daily_frames.append(pair_daily)
        metric_records.append({"pair": pair_name, "ticker_y": ticker_y, "ticker_x": ticker_x, **metrics})

    pair_returns = pd.concat(daily_frames, axis=1).fillna(0.0)
    pair_returns["portfolio"] = pair_returns.mean(axis=1)
    metrics = pd.DataFrame(metric_records)
    portfolio_metrics = performance_metrics(pair_returns["portfolio"])
    portfolio_metrics["number_of_trades"] = float(metrics["number_of_trades"].sum()) if not metrics.empty else 0.0
    portfolio_row = {"pair": "Equal-weight selected pairs", "ticker_y": "PORTFOLIO", "ticker_x": "", **portfolio_metrics}
    metrics = pd.concat([metrics, pd.DataFrame([portfolio_row])], ignore_index=True)
    return pair_returns, metrics
