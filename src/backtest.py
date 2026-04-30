from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import drawdown, equity_curve, performance_metrics
from .pairs import calculate_spread, estimate_hedge_ratio
from .signals import generate_positions_with_reasons, rolling_zscore


@dataclass
class BacktestResult:
    pair_name: str
    daily: pd.DataFrame
    metrics: dict[str, float | str]
    trades: pd.DataFrame


def rolling_hedge_parameters(log_pair: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Estimate OLS hedge parameters using only observations before each date."""
    records: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    for i in range(len(log_pair)):
        if i < window:
            records.append({"hedge_ratio": np.nan, "intercept": np.nan})
            index.append(log_pair.index[i])
            continue
        train = log_pair.iloc[i - window : i]
        hedge_ratio, intercept = estimate_hedge_ratio(train["y"], train["x"])
        records.append({"hedge_ratio": hedge_ratio, "intercept": intercept})
        index.append(log_pair.index[i])
    return pd.DataFrame(records, index=index)


def kalman_hedge_parameters(
    log_pair: pd.DataFrame,
    process_variance: float = 1e-5,
    observation_variance: float = 1e-3,
) -> pd.DataFrame:
    """Recursive two-state Kalman update for intercept and hedge ratio."""
    theta = np.array([0.0, 1.0], dtype=float)
    covariance = np.eye(2) * 1.0
    process = np.eye(2) * process_variance
    records: list[dict[str, float]] = []

    for _, row in log_pair.iterrows():
        x_vec = np.array([1.0, float(row["x"])], dtype=float)
        covariance = covariance + process
        prediction = float(x_vec @ theta)
        innovation = float(row["y"]) - prediction
        innovation_var = float(x_vec @ covariance @ x_vec.T + observation_variance)
        gain = covariance @ x_vec.T / innovation_var
        theta = theta + gain * innovation
        covariance = covariance - np.outer(gain, x_vec) @ covariance
        records.append({"intercept": float(theta[0]), "hedge_ratio": float(theta[1])})

    return pd.DataFrame(records, index=log_pair.index)


def static_hedge_parameters(
    log_pair: pd.DataFrame,
    hedge_ratio: float | None = None,
    intercept: float | None = None,
    fit_window: int | None = None,
) -> pd.DataFrame:
    if hedge_ratio is None or intercept is None:
        fit_data = log_pair.iloc[:fit_window] if fit_window else log_pair
        hedge_ratio, intercept = estimate_hedge_ratio(fit_data["y"], fit_data["x"])
    return pd.DataFrame(
        {"hedge_ratio": float(hedge_ratio), "intercept": float(intercept)},
        index=log_pair.index,
    )


def calculate_dynamic_spread(log_pair: pd.DataFrame, hedge_params: pd.DataFrame) -> pd.Series:
    spread = log_pair["y"] - (hedge_params["intercept"] + hedge_params["hedge_ratio"] * log_pair["x"])
    spread.name = "spread"
    return spread


def volatility_entry_filter(spread: pd.Series, lookback: int, volatility_limit: float) -> pd.Series:
    rolling_vol = spread.diff().rolling(lookback).std(ddof=0)
    return (rolling_vol <= volatility_limit).fillna(False)


def combined_entry_filter(
    spread: pd.Series,
    zscore: pd.Series,
    log_pair: pd.DataFrame,
    hedge_ratio: pd.Series,
    lookback: int,
    volatility_limit: float,
    transaction_cost_bps: float,
    edge_threshold: float = 0.0,
    use_volatility_filter: bool = True,
    use_correlation_filter: bool = False,
    use_trend_filter: bool = False,
    min_recent_correlation: float = 0.70,
) -> pd.Series:
    can_enter = pd.Series(True, index=spread.index)
    recent_vol = spread.diff().rolling(lookback).std(ddof=0)
    if use_volatility_filter:
        can_enter &= recent_vol <= volatility_limit

    if edge_threshold > 0:
        edge = zscore.abs() / recent_vol.replace(0.0, np.nan)
        cost_hurdle = (transaction_cost_bps / 10000.0) * (1.0 + hedge_ratio.abs()) * 100.0
        can_enter &= edge >= (edge_threshold + cost_hurdle)

    if use_correlation_filter:
        recent_corr = log_pair["y"].rolling(63).corr(log_pair["x"])
        can_enter &= recent_corr >= min_recent_correlation

    if use_trend_filter:
        trend_std = spread.rolling(60).std(ddof=0).replace(0.0, np.nan)
        trend_score = (spread.rolling(60).mean() - spread.rolling(60).mean().shift(20)).abs() / trend_std
        can_enter &= trend_score <= 0.75

    return can_enter.fillna(False)


def training_volatility_limit(spread: pd.Series, lookback: int, percentile: float = 0.90) -> float:
    vol = spread.diff().rolling(lookback).std(ddof=0).dropna()
    if vol.empty:
        return np.inf
    return float(vol.quantile(percentile))


def pair_leg_returns(
    prices: pd.DataFrame,
    hedge_ratio: pd.Series,
    position: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Calculate lagged, cost-adjusted pair returns."""
    frame = prices.copy()
    frame["ret_y"] = frame["y"].pct_change()
    frame["ret_x"] = frame["x"].pct_change()
    frame["hedge_ratio"] = hedge_ratio.reindex(frame.index)
    frame["position"] = position.reindex(frame.index).fillna(0.0)
    frame["executed_position"] = frame["position"].shift(1).fillna(0.0)
    frame["executed_hedge_ratio"] = frame["hedge_ratio"].shift(1)
    frame["raw_pair_return"] = (
        frame["ret_y"] - frame["executed_hedge_ratio"] * frame["ret_x"]
    ) / (1.0 + frame["executed_hedge_ratio"].abs())
    frame["position_change"] = frame["executed_position"].diff().abs().fillna(frame["executed_position"].abs())
    frame["trading_cost"] = frame["position_change"] * 2.0 * transaction_cost_bps / 10000.0
    frame["strategy_return_gross"] = frame["executed_position"] * frame["raw_pair_return"]
    frame["strategy_return"] = frame["strategy_return_gross"] - frame["trading_cost"]
    frame["equity"] = equity_curve(frame["strategy_return"].fillna(0.0))
    frame["drawdown"] = drawdown(frame["equity"])
    return frame.fillna(0.0)


def apply_pair_drawdown_stop(
    prices: pd.DataFrame,
    hedge_ratio: pd.Series,
    position: pd.Series,
    transaction_cost_bps: float,
    drawdown_stop: float | None,
) -> pd.DataFrame:
    daily = pair_leg_returns(prices, hedge_ratio, position, transaction_cost_bps)
    if drawdown_stop is None or drawdown_stop >= 0:
        return daily

    adjusted = position.copy()
    stopped = False
    for date, row in daily.iterrows():
        if stopped:
            adjusted.loc[date] = 0
        elif row["drawdown"] <= drawdown_stop:
            stopped = True
            adjusted.loc[date] = 0
    if stopped:
        daily = pair_leg_returns(prices, hedge_ratio, adjusted, transaction_cost_bps)
        daily["position"] = adjusted.reindex(daily.index).fillna(0.0)
    return daily


def summarize_trades(daily: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    """Build a trade log from executed positions."""
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    previous_position = 0

    for date, row in daily.iterrows():
        position = int(row["executed_position"])
        zscore_source = row.get("signal_zscore", row["zscore"])
        zscore = float(zscore_source) if pd.notna(zscore_source) else np.nan

        if current is None and position != 0:
            current = {
                "pair": pair_name,
                "entry_date": date,
                "direction": "long_spread" if position > 0 else "short_spread",
                "entry_zscore": zscore,
                "gross_returns": [],
                "net_returns": [],
                "costs": [],
            }

        if current is not None:
            current["gross_returns"].append(float(row["strategy_return_gross"]))
            current["net_returns"].append(float(row["strategy_return"]))
            current["costs"].append(float(row["trading_cost"]))

        exit_reason = str(row.get("realized_exit_reason", ""))
        should_close = current is not None and previous_position != 0 and position == 0
        flipped = current is not None and previous_position != 0 and position != 0 and np.sign(position) != np.sign(previous_position)
        if current is not None and (should_close or flipped):
            gross_returns = pd.Series(current["gross_returns"], dtype=float)
            net_returns = pd.Series(current["net_returns"], dtype=float)
            records.append(
                {
                    "pair": pair_name,
                    "entry_date": current["entry_date"],
                    "exit_date": date,
                    "direction": current["direction"],
                    "entry_zscore": current["entry_zscore"],
                    "exit_zscore": zscore,
                    "holding_period": int(len(net_returns)),
                    "gross_return": float((1.0 + gross_returns).prod() - 1.0),
                    "net_return": float((1.0 + net_returns).prod() - 1.0),
                    "exit_reason": exit_reason or "mean_reversion",
                    "transaction_cost": float(pd.Series(current["costs"], dtype=float).sum()),
                }
            )
            current = None

        if flipped and position != 0:
            current = {
                "pair": pair_name,
                "entry_date": date,
                "direction": "long_spread" if position > 0 else "short_spread",
                "entry_zscore": zscore,
                "gross_returns": [float(row["strategy_return_gross"])],
                "net_returns": [float(row["strategy_return"])],
                "costs": [float(row["trading_cost"])],
            }

        previous_position = position

    if current is not None:
        gross_returns = pd.Series(current["gross_returns"], dtype=float)
        net_returns = pd.Series(current["net_returns"], dtype=float)
        records.append(
            {
                "pair": pair_name,
                "entry_date": current["entry_date"],
                "exit_date": daily.index[-1],
                "direction": current["direction"],
                "entry_zscore": current["entry_zscore"],
                "exit_zscore": float(daily["zscore"].iloc[-1]),
                "holding_period": int(len(net_returns)),
                "gross_return": float((1.0 + gross_returns).prod() - 1.0),
                "net_return": float((1.0 + net_returns).prod() - 1.0),
                "exit_reason": "time_stop",
                "transaction_cost": float(pd.Series(current["costs"], dtype=float).sum()),
            }
        )

    return pd.DataFrame(records)


def trade_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    base = {
        "number_of_trades": 0.0,
        "win_rate": 0.0,
        "average_trade_return": 0.0,
        "median_trade_return": 0.0,
        "average_holding_period": 0.0,
        "profit_factor": 0.0,
        "stop_loss_exit_pct": 0.0,
        "time_stop_exit_pct": 0.0,
        "mean_reversion_exit_pct": 0.0,
        "turnover": float(daily["position_change"].sum()),
        "transaction_cost_impact": float(daily["trading_cost"].sum()),
    }
    if trades.empty:
        return base

    winners = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losers = trades.loc[trades["net_return"] < 0, "net_return"].sum()
    base.update(
        {
            "number_of_trades": float(len(trades)),
            "win_rate": float((trades["net_return"] > 0).mean()),
            "average_trade_return": float(trades["net_return"].mean()),
            "median_trade_return": float(trades["net_return"].median()),
            "average_holding_period": float(trades["holding_period"].mean()),
            "profit_factor": float(winners / abs(losers)) if losers < 0 else float("inf") if winners > 0 else 0.0,
            "stop_loss_exit_pct": float((trades["exit_reason"] == "stop_loss").mean()),
            "time_stop_exit_pct": float((trades["exit_reason"] == "time_stop").mean()),
            "mean_reversion_exit_pct": float((trades["exit_reason"] == "mean_reversion").mean()),
        }
    )
    return base


def run_pair_backtest(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.0,
    zscore_window: int = 60,
    transaction_cost_bps: float = 5.0,
    max_holding_period: int = 20,
    hedge_mode: str = "static",
    hedge_training_window: int = 252,
    hedge_ratio: float | None = None,
    intercept: float | None = None,
    volatility_limit: float | None = None,
    volatility_percentile: float = 0.90,
    use_volatility_filter: bool = True,
    use_correlation_filter: bool = False,
    use_trend_filter: bool = False,
    edge_threshold: float = 0.0,
    cooldown_days: int = 0,
    fit_window: int | None = None,
    pair_drawdown_stop: float | None = None,
) -> BacktestResult:
    pair_name = f"{ticker_y}/{ticker_x}"
    pair = prices[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    log_pair = np.log(pair)

    if hedge_mode == "rolling":
        hedge_params = rolling_hedge_parameters(log_pair, hedge_training_window)
    elif hedge_mode == "kalman":
        hedge_params = kalman_hedge_parameters(log_pair)
    elif hedge_mode == "static":
        hedge_params = static_hedge_parameters(log_pair, hedge_ratio, intercept, fit_window)
    else:
        raise ValueError("hedge_mode must be 'static', 'rolling', or 'kalman'.")

    spread = calculate_dynamic_spread(log_pair, hedge_params)
    zscore = rolling_zscore(spread, zscore_window)
    if volatility_limit is None:
        training_spread = spread.iloc[:fit_window] if fit_window else spread
        volatility_limit = training_volatility_limit(training_spread, zscore_window, volatility_percentile)
    can_enter = combined_entry_filter(
        spread=spread,
        zscore=zscore,
        log_pair=log_pair,
        hedge_ratio=hedge_params["hedge_ratio"],
        lookback=zscore_window,
        volatility_limit=volatility_limit,
        transaction_cost_bps=transaction_cost_bps,
        edge_threshold=edge_threshold,
        use_volatility_filter=use_volatility_filter,
        use_correlation_filter=use_correlation_filter,
        use_trend_filter=use_trend_filter,
    )
    position, exit_reasons = generate_positions_with_reasons(
        zscore=zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
        can_enter=can_enter,
        max_holding_period=max_holding_period,
        cooldown_days=cooldown_days,
    )

    daily = apply_pair_drawdown_stop(pair, hedge_params["hedge_ratio"], position, transaction_cost_bps, pair_drawdown_stop)
    daily["spread"] = spread.reindex(daily.index)
    daily["zscore"] = zscore.reindex(daily.index)
    daily["signal_zscore"] = daily["zscore"].shift(1)
    daily["target_exit_reason"] = exit_reasons.reindex(daily.index)
    daily["realized_exit_reason"] = daily["target_exit_reason"].shift(1).fillna("")
    daily["can_enter"] = can_enter.reindex(daily.index).fillna(False)
    daily["intercept"] = hedge_params["intercept"].reindex(daily.index)
    trades = summarize_trades(daily, pair_name)
    metrics = performance_metrics(daily["strategy_return"])
    metrics.update(trade_metrics(daily, trades))
    metrics.update(
        {
            "hedge_mode": hedge_mode,
            "hedge_ratio": float(hedge_params["hedge_ratio"].dropna().iloc[-1])
            if not hedge_params["hedge_ratio"].dropna().empty
            else np.nan,
            "intercept": float(hedge_params["intercept"].dropna().iloc[-1])
            if not hedge_params["intercept"].dropna().empty
            else np.nan,
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
            "stop_threshold": float(stop_threshold),
            "max_holding_period": float(max_holding_period),
            "transaction_cost_bps": float(transaction_cost_bps),
            "volatility_limit": float(volatility_limit),
        }
    )

    return BacktestResult(pair_name, daily, metrics, trades)
