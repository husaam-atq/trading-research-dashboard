from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .backtest import (
    pair_leg_returns,
    summarize_trades,
    trade_metrics,
    training_volatility_limit,
    volatility_entry_filter,
)
from .metrics import performance_metrics
from .pairs import calculate_spread, estimate_hedge_ratio
from .signals import fixed_zscore, generate_positions_with_reasons, rolling_zscore


def _evaluate_training_thresholds(
    train: pd.DataFrame,
    hedge_ratio: float,
    intercept: float,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    transaction_cost_bps: float,
    max_holding_period: int,
    zscore_window: int,
    min_trades: int,
) -> dict[str, float]:
    log_train = np.log(train)
    spread = calculate_spread(log_train["y"], log_train["x"], hedge_ratio, intercept)
    zscore = rolling_zscore(spread, zscore_window)
    vol_limit = training_volatility_limit(spread, zscore_window)
    can_enter = volatility_entry_filter(spread, zscore_window, vol_limit)
    position, exit_reasons = generate_positions_with_reasons(
        zscore,
        entry_threshold,
        exit_threshold,
        stop_threshold,
        can_enter=can_enter,
        max_holding_period=max_holding_period,
    )
    daily = pair_leg_returns(train, pd.Series(hedge_ratio, index=train.index), position, transaction_cost_bps)
    daily["zscore"] = zscore
    daily["signal_zscore"] = daily["zscore"].shift(1)
    daily["target_exit_reason"] = exit_reasons
    daily["realized_exit_reason"] = daily["target_exit_reason"].shift(1).fillna("")
    trades = summarize_trades(daily, "training")
    metrics = performance_metrics(daily["strategy_return"])
    metrics.update(trade_metrics(daily, trades))
    if metrics["number_of_trades"] < min_trades:
        metrics["sharpe_ratio"] = -np.inf
    return metrics


def select_thresholds_on_training(
    train: pd.DataFrame,
    hedge_ratio: float,
    intercept: float,
    entry_grid: list[float],
    exit_grid: list[float],
    stop_grid: list[float],
    transaction_cost_bps: float,
    max_holding_period: int,
    zscore_window: int,
    min_trades: int = 3,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    for entry, exit_, stop in product(entry_grid, exit_grid, stop_grid):
        if exit_ >= entry or stop <= entry:
            continue
        metrics = _evaluate_training_thresholds(
            train=train,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            entry_threshold=entry,
            exit_threshold=exit_,
            stop_threshold=stop,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_period=max_holding_period,
            zscore_window=zscore_window,
            min_trades=min_trades,
        )
        records.append(
            {
                "entry_threshold": entry,
                "exit_threshold": exit_,
                "stop_threshold": stop,
                "train_sharpe": float(metrics["sharpe_ratio"]),
                "train_total_return": float(metrics["total_return"]),
                "train_number_of_trades": float(metrics["number_of_trades"]),
            }
        )

    ranked = pd.DataFrame(records).sort_values(
        ["train_sharpe", "train_total_return", "train_number_of_trades"],
        ascending=[False, False, False],
    )
    if ranked.empty or not np.isfinite(float(ranked.iloc[0]["train_sharpe"])):
        fallback = pd.DataFrame(records).sort_values("train_number_of_trades", ascending=False)
        if fallback.empty:
            return {"entry_threshold": 2.0, "exit_threshold": 0.5, "stop_threshold": 3.0}
        return fallback.iloc[0].to_dict()
    return ranked.iloc[0].to_dict()


def walk_forward_pair(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    entry_grid: list[float],
    exit_grid: list[float],
    stop_grid: list[float],
    transaction_cost_bps: float = 5.0,
    max_holding_period: int = 20,
    zscore_window: int = 60,
    train_window: int = 252,
    test_window: int = 63,
    min_trades: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run nested walk-forward validation with training-only threshold selection."""
    pair_name = f"{ticker_y}/{ticker_x}"
    pair = prices[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    segments: list[pd.DataFrame] = []
    threshold_records: list[dict[str, float | str | pd.Timestamp]] = []
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
        selected = select_thresholds_on_training(
            train=train,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            stop_grid=stop_grid,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_period=max_holding_period,
            zscore_window=zscore_window,
            min_trades=min_trades,
        )
        vol_limit = training_volatility_limit(train_spread, zscore_window)
        zscore = fixed_zscore(test_spread, float(train_spread.mean()), float(train_spread.std(ddof=0)))
        can_enter = volatility_entry_filter(test_spread, zscore_window, vol_limit)
        position, exit_reasons = generate_positions_with_reasons(
            zscore=zscore,
            entry_threshold=float(selected["entry_threshold"]),
            exit_threshold=float(selected["exit_threshold"]),
            stop_threshold=float(selected["stop_threshold"]),
            can_enter=can_enter,
            max_holding_period=max_holding_period,
        )

        daily = pair_leg_returns(test, pd.Series(hedge_ratio, index=test.index), position, transaction_cost_bps)
        daily["spread"] = test_spread
        daily["zscore"] = zscore
        daily["signal_zscore"] = daily["zscore"].shift(1)
        daily["target_exit_reason"] = exit_reasons
        daily["realized_exit_reason"] = daily["target_exit_reason"].shift(1).fillna("")
        daily["can_enter"] = can_enter
        daily["segment"] = segment_id
        daily["train_start"] = train.index[0]
        daily["train_end"] = train.index[-1]
        daily["test_start"] = test.index[0]
        daily["test_end"] = test.index[-1]
        daily["hedge_ratio"] = hedge_ratio
        daily["intercept"] = intercept
        daily["entry_threshold"] = float(selected["entry_threshold"])
        daily["exit_threshold"] = float(selected["exit_threshold"])
        daily["stop_threshold"] = float(selected["stop_threshold"])
        segments.append(daily)

        threshold_records.append(
            {
                "pair": pair_name,
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "segment": segment_id,
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                **selected,
            }
        )

        segment_id += 1
        start += test_window

    if not segments:
        raise ValueError(f"Not enough data for walk-forward validation: {pair_name}")

    result = pd.concat(segments).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    thresholds = pd.DataFrame(threshold_records)
    trades = summarize_trades(result, pair_name)
    metrics = performance_metrics(result["strategy_return"])
    metrics.update(trade_metrics(result, trades))
    metrics["segments"] = float(segment_id)
    return result, thresholds, metrics


def walk_forward_many(
    prices: pd.DataFrame,
    pairs: pd.DataFrame,
    **kwargs: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, float | str]] = []

    for row in pairs.itertuples(index=False):
        ticker_y = row.ticker_y
        ticker_x = row.ticker_x
        daily, thresholds, metrics = walk_forward_pair(prices, ticker_y, ticker_x, **kwargs)
        pair_name = f"{ticker_y}/{ticker_x}"
        pair_daily = daily[["strategy_return"]].rename(columns={"strategy_return": pair_name})
        daily_frames.append(pair_daily)
        threshold_frames.append(thresholds)
        peer_group = getattr(row, "peer_group", "")
        metric_records.append(
            {"pair": pair_name, "peer_group": peer_group, "ticker_y": ticker_y, "ticker_x": ticker_x, **metrics}
        )

    pair_returns = pd.concat(daily_frames, axis=1).fillna(0.0)
    pair_returns["portfolio"] = pair_returns.mean(axis=1)
    metrics = pd.DataFrame(metric_records)
    portfolio_metrics = performance_metrics(pair_returns["portfolio"])
    portfolio_metrics["number_of_trades"] = float(metrics["number_of_trades"].sum()) if not metrics.empty else 0.0
    portfolio_row = {"pair": "Equal-weight selected pairs", "peer_group": "portfolio", "ticker_y": "PORTFOLIO", "ticker_x": "", **portfolio_metrics}
    metrics = pd.concat([metrics, pd.DataFrame([portfolio_row])], ignore_index=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    return pair_returns, metrics, thresholds
