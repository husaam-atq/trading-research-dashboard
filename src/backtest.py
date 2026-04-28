from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import drawdown, equity_curve, performance_metrics
from .pairs import calculate_spread, estimate_hedge_ratio
from .signals import generate_positions, rolling_zscore


@dataclass
class BacktestResult:
    pair_name: str
    daily: pd.DataFrame
    metrics: dict[str, float]
    trades: pd.DataFrame


def pair_leg_returns(
    prices: pd.DataFrame,
    hedge_ratio: float,
    position: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Calculate lagged, cost-adjusted pair returns."""
    frame = prices.copy()
    frame["ret_y"] = frame["y"].pct_change()
    frame["ret_x"] = frame["x"].pct_change()
    frame["raw_pair_return"] = (frame["ret_y"] - hedge_ratio * frame["ret_x"]) / (1.0 + abs(hedge_ratio))

    target_position = position.reindex(frame.index).fillna(0.0)
    frame["position"] = target_position
    frame["executed_position"] = target_position.shift(1).fillna(0.0)
    frame["position_change"] = frame["executed_position"].diff().abs().fillna(frame["executed_position"].abs())
    frame["trading_cost"] = frame["position_change"] * 2.0 * transaction_cost_bps / 10000.0
    frame["strategy_return_gross"] = frame["executed_position"] * frame["raw_pair_return"]
    frame["strategy_return"] = frame["strategy_return_gross"] - frame["trading_cost"]
    frame["equity"] = equity_curve(frame["strategy_return"])
    frame["drawdown"] = drawdown(frame["equity"])
    return frame.fillna(0.0)


def summarize_trades(daily: pd.DataFrame) -> pd.DataFrame:
    """Build a compact trade log from executed positions."""
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for date, row in daily.iterrows():
        position = int(row["executed_position"])
        previous = int(daily["executed_position"].shift(1).fillna(0.0).loc[date])

        if current is None and position != 0:
            current = {
                "entry_date": date,
                "side": "long_spread" if position > 0 else "short_spread",
                "returns": [],
            }

        if current is not None:
            current["returns"].append(float(row["strategy_return"]))

        if current is not None and previous != 0 and position == 0:
            trade_returns = pd.Series(current["returns"], dtype=float)
            records.append(
                {
                    "entry_date": current["entry_date"],
                    "exit_date": date,
                    "side": current["side"],
                    "holding_period": int(len(trade_returns)),
                    "trade_return": float((1.0 + trade_returns).prod() - 1.0),
                }
            )
            current = None

        if current is not None and previous != 0 and position != 0 and np.sign(position) != np.sign(previous):
            trade_returns = pd.Series(current["returns"], dtype=float)
            records.append(
                {
                    "entry_date": current["entry_date"],
                    "exit_date": date,
                    "side": current["side"],
                    "holding_period": int(len(trade_returns)),
                    "trade_return": float((1.0 + trade_returns).prod() - 1.0),
                }
            )
            current = {
                "entry_date": date,
                "side": "long_spread" if position > 0 else "short_spread",
                "returns": [float(row["strategy_return"])],
            }

    if current is not None:
        trade_returns = pd.Series(current["returns"], dtype=float)
        records.append(
            {
                "entry_date": current["entry_date"],
                "exit_date": daily.index[-1],
                "side": current["side"],
                "holding_period": int(len(trade_returns)),
                "trade_return": float((1.0 + trade_returns).prod() - 1.0),
            }
        )

    return pd.DataFrame(records)


def trade_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "number_of_trades": 0.0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "average_holding_period": 0.0,
            "turnover": float(daily["position_change"].sum()),
            "transaction_cost_impact": float(daily["trading_cost"].sum()),
        }

    return {
        "number_of_trades": float(len(trades)),
        "win_rate": float((trades["trade_return"] > 0).mean()),
        "average_trade_return": float(trades["trade_return"].mean()),
        "average_holding_period": float(trades["holding_period"].mean()),
        "turnover": float(daily["position_change"].sum()),
        "transaction_cost_impact": float(daily["trading_cost"].sum()),
    }


def run_pair_backtest(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.0,
    zscore_window: int = 60,
    transaction_cost_bps: float = 5.0,
    hedge_ratio: float | None = None,
    intercept: float | None = None,
) -> BacktestResult:
    pair = prices[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    log_pair = np.log(pair)

    if hedge_ratio is None or intercept is None:
        hedge_ratio, intercept = estimate_hedge_ratio(log_pair["y"], log_pair["x"])

    spread = calculate_spread(log_pair["y"], log_pair["x"], hedge_ratio, intercept)
    zscore = rolling_zscore(spread, zscore_window)
    position = generate_positions(zscore, entry_threshold, exit_threshold, stop_threshold)

    daily = pair_leg_returns(pair, hedge_ratio, position, transaction_cost_bps)
    daily["spread"] = spread.reindex(daily.index)
    daily["zscore"] = zscore.reindex(daily.index)
    trades = summarize_trades(daily)
    metrics = performance_metrics(daily["strategy_return"])
    metrics.update(trade_metrics(daily, trades))
    metrics.update(
        {
            "hedge_ratio": float(hedge_ratio),
            "intercept": float(intercept),
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
            "stop_threshold": float(stop_threshold),
            "transaction_cost_bps": float(transaction_cost_bps),
        }
    )

    return BacktestResult(f"{ticker_y}/{ticker_x}", daily, metrics, trades)
