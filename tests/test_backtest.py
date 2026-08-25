from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import pair_leg_returns, rolling_hedge_parameters, run_pair_backtest


def test_zero_position_has_zero_strategy_pnl() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"y": np.linspace(100, 110, 10), "x": np.linspace(50, 53, 10)}, index=index)
    result = pair_leg_returns(prices, pd.Series(1.0, index=index), pd.Series(0, index=index), 5.0)
    assert result["strategy_return"].eq(0.0).all()


def test_position_is_lagged_and_costs_reconcile() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"y": [100, 100, 101, 102, 102], "x": [100, 100, 100, 100, 100]}, index=index)
    target = pd.Series([0, 1, 1, 0, 0], index=index)
    result = pair_leg_returns(prices, pd.Series(1.0, index=index), target, 5.0)
    assert result["executed_position"].tolist() == [0, 0, 1, 1, 0]
    assert np.isclose(result["trading_cost"].sum(), 0.002)
    assert np.allclose(
        result["strategy_return"],
        result["strategy_return_gross"] - result["trading_cost"],
    )


def test_long_spread_direction_is_correct() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"y": [100, 100, 102], "x": [100, 100, 100]}, index=index)
    target = pd.Series([1, 1, 1], index=index)
    result = pair_leg_returns(prices, pd.Series(1.0, index=index), target, 0.0)
    assert result["strategy_return"].iloc[2] > 0


def test_rolling_hedge_uses_only_prior_prices(synthetic_prices: pd.DataFrame) -> None:
    log_pair = np.log(synthetic_prices[["A", "B"]].rename(columns={"A": "y", "B": "x"}))
    changed = log_pair.copy()
    changed.iloc[500:, 0] += 5.0
    original_params = rolling_hedge_parameters(log_pair, 126)
    changed_params = rolling_hedge_parameters(changed, 126)
    pd.testing.assert_frame_equal(original_params.iloc[:501], changed_params.iloc[:501])


def test_trade_start_forces_flat_boundary(synthetic_prices: pd.DataFrame) -> None:
    trade_start = synthetic_prices.index[600]
    result = run_pair_backtest(
        synthetic_prices,
        "A",
        "B",
        hedge_mode="static",
        fit_window=504,
        trade_start=trade_start,
        use_volatility_filter=False,
    )
    assert result.daily.loc[:trade_start, "executed_position"].eq(0).all()
