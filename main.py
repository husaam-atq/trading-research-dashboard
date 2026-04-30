from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import run_pair_backtest, trade_metrics
from src.data import DEFAULT_END, DEFAULT_START, flatten_universes, research_universes, download_adjusted_close
from src.metrics import performance_metrics
from src.plots import (
    plot_cost_sensitivity,
    plot_drawdowns,
    plot_equity_curves,
    plot_hedge_mode_comparison,
    plot_nested_selected_portfolio,
    plot_pair_comparison,
    plot_spread_zscore,
    plot_trade_return_distribution,
    plot_walk_forward_performance,
)
from src.robust import benchmark_metrics_table, cost_sensitivity_from_selection, nested_pair_selection_portfolio, screen_all_modes

OUTPUT_DIR = Path("outputs")

ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
STOP_THRESHOLD = 3.0
ENTRY_GRID = [1.25, 1.5, 1.75, 2.0, 2.25]
EXIT_GRID = [0.0, 0.25, 0.5, 0.75]
STOP_GRID = [2.75, 3.0, 3.5, 4.0]
MAX_HOLD_GRID = [10, 20, 30]
HEDGE_MODES = ["static", "rolling", "kalman"]

TRANSACTION_COST_BPS = 5.0
TRAINING_WINDOW = 504
VALIDATION_WINDOW = 126
TEST_WINDOW = 63
ROBUST_TOP_N = 3
MAX_VALIDATION_CANDIDATES = 1
WALK_FORWARD_STEP = 126

CORRELATION_THRESHOLD = 0.85
COINTEGRATION_PVALUE_THRESHOLD = 0.05
ADF_PVALUE_THRESHOLD = 0.05
MIN_HALF_LIFE = 3.0
MAX_HALF_LIFE = 45.0
MIN_THRESHOLD_CROSSINGS = 8
MIN_TRAINING_TRADES = 5
MIN_TRAINING_SHARPE = 0.25
MIN_TRAINING_MAX_DRAWDOWN = -0.20


def _benchmark_returns(prices: pd.DataFrame, universe_tickers: list[str]) -> pd.DataFrame:
    returns = pd.DataFrame(index=prices.index)
    returns["cash"] = 0.0
    if "SPY" in prices:
        returns["SPY_buy_hold"] = prices["SPY"].pct_change().fillna(0.0)
    available = [ticker for ticker in universe_tickers if ticker in prices.columns and ticker != "SPY"]
    if available:
        returns["equal_weight_long_only_universe"] = prices[available].pct_change().mean(axis=1).fillna(0.0)
    return returns


def _aggregate_trade_analytics(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["universe_mode", "peer_group", "pair", "hedge_mode"]
    for key, trades in trade_log.groupby(group_cols):
        winners = trades.loc[trades["net_return"] > 0, "net_return"].sum()
        losers = trades.loc[trades["net_return"] < 0, "net_return"].sum()
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "number_of_trades": float(len(trades)),
                "average_trade_return": float(trades["net_return"].mean()),
                "median_trade_return": float(trades["net_return"].median()),
                "profit_factor": float(winners / abs(losers)) if losers < 0 else float("inf") if winners > 0 else 0.0,
                "win_rate": float((trades["net_return"] > 0).mean()),
                "average_holding_period": float(trades["holding_period"].mean()),
                "stop_loss_exit_rate": float((trades["exit_reason"] == "stop_loss").mean()),
                "time_stop_exit_rate": float((trades["exit_reason"] == "time_stop").mean()),
                "mean_reversion_exit_rate": float((trades["exit_reason"] == "mean_reversion").mean()),
            }
        )
    return pd.DataFrame(rows)


def _run_hedge_mode_comparison(prices: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    returns: list[pd.Series] = []
    trade_logs: list[pd.DataFrame] = []
    for row in candidates.itertuples(index=False):
        for hedge_mode in HEDGE_MODES:
            result = run_pair_backtest(
                prices,
                row.ticker_y,
                row.ticker_x,
                entry_threshold=ENTRY_THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
                stop_threshold=STOP_THRESHOLD,
                zscore_window=int(row.zscore_window),
                transaction_cost_bps=TRANSACTION_COST_BPS,
                max_holding_period=20,
                hedge_mode=hedge_mode,
                hedge_ratio=float(row.hedge_ratio) if hedge_mode == "static" else None,
                intercept=float(row.intercept) if hedge_mode == "static" else None,
                fit_window=TRAINING_WINDOW,
                pair_drawdown_stop=-0.20,
            )
            rows.append(
                {
                    "universe_mode": row.universe_mode,
                    "peer_group": row.peer_group,
                    "pair": row.pair,
                    "hedge_mode": hedge_mode,
                    **result.metrics,
                }
            )
            returns.append(result.daily["strategy_return"].rename(f"{row.pair}_{hedge_mode}"))
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["universe_mode"] = row.universe_mode
                trades["peer_group"] = row.peer_group
                trades["hedge_mode"] = hedge_mode
                trades["entry_threshold"] = ENTRY_THRESHOLD
                trades["exit_threshold"] = EXIT_THRESHOLD
                trades["stop_threshold"] = STOP_THRESHOLD
                trades["max_holding_period"] = 20
                trade_logs.append(trades)

    comparison = pd.DataFrame(rows)
    returns_df = pd.concat(returns, axis=1).fillna(0.0) if returns else pd.DataFrame(index=prices.index)
    for hedge_mode in HEDGE_MODES:
        cols = [col for col in returns_df.columns if col.endswith(f"_{hedge_mode}")]
        if cols:
            returns_df[f"all_screened_{hedge_mode}_portfolio"] = returns_df[cols].mean(axis=1)
    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()
    return comparison, returns_df, trade_log


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    universes = research_universes()
    tickers = flatten_universes(universes)

    print("Downloading adjusted close data...")
    prices = download_adjusted_close(tickers, DEFAULT_START, DEFAULT_END)
    training_prices = prices.iloc[:TRAINING_WINDOW]

    print("Running stock and ETF pair stability diagnostics...")
    screening, diagnostics = screen_all_modes(
        training_prices,
        min_abs_correlation=CORRELATION_THRESHOLD,
        max_coint_pvalue=COINTEGRATION_PVALUE_THRESHOLD,
        max_adf_pvalue=ADF_PVALUE_THRESHOLD,
        min_half_life=MIN_HALF_LIFE,
        max_half_life=MAX_HALF_LIFE,
        min_threshold_crossings=MIN_THRESHOLD_CROSSINGS,
        min_training_trades=MIN_TRAINING_TRADES,
        min_training_sharpe=MIN_TRAINING_SHARPE,
        min_training_max_drawdown=MIN_TRAINING_MAX_DRAWDOWN,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        max_holding_period=20,
    )
    screening.to_csv(OUTPUT_DIR / "pair_screening_results.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "pair_stability_diagnostics.csv", index=False)

    hedge_candidates = diagnostics[diagnostics["stable_candidate"]].copy()
    if hedge_candidates.empty:
        hedge_candidates = diagnostics.head(12).copy()
    else:
        hedge_candidates = hedge_candidates.head(20).copy()
    robust_candidate_pool = pd.concat(
        [
            diagnostics[diagnostics["stable_candidate"]].head(12),
            diagnostics.sort_values(["coint_pvalue", "training_sharpe"], ascending=[True, False]).head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(["universe_mode", "peer_group", "ticker_y", "ticker_x"])

    print("Comparing static, rolling, and Kalman hedge modes...")
    hedge_comparison, all_pair_returns, hedge_trade_log = _run_hedge_mode_comparison(prices, hedge_candidates)
    hedge_comparison.to_csv(OUTPUT_DIR / "hedge_mode_comparison.csv", index=False)

    print("Running nested walk-forward pair and parameter selection...")
    nested_selection, nested_results, nested_daily, nested_pair_returns, nested_trade_log = nested_pair_selection_portfolio(
        prices,
        train_window=TRAINING_WINDOW,
        validation_window=VALIDATION_WINDOW,
        test_window=TEST_WINDOW,
        top_n_pairs=ROBUST_TOP_N,
        cost_bps=TRANSACTION_COST_BPS,
        entry_grid=ENTRY_GRID,
        exit_grid=EXIT_GRID,
        stop_grid=STOP_GRID,
        max_hold_grid=MAX_HOLD_GRID,
        hedge_modes=HEDGE_MODES,
        max_validation_candidates=MAX_VALIDATION_CANDIDATES,
        use_vol_filter=True,
        vol_percentiles=(0.90,),
        candidate_pool=robust_candidate_pool,
        step_size=WALK_FORWARD_STEP,
    )
    nested_selection.to_csv(OUTPUT_DIR / "nested_pair_selection.csv", index=False)
    nested_results.to_csv(OUTPUT_DIR / "nested_walk_forward_results.csv", index=False)
    nested_daily.to_csv(OUTPUT_DIR / "nested_walk_forward_daily_returns.csv", index_label="date")
    nested_results.to_csv(OUTPUT_DIR / "pair_portfolio_comparison.csv", index=False)

    print("Running transaction cost sensitivity...")
    costs = cost_sensitivity_from_selection(
        prices,
        nested_selection,
        costs=[0.0, 1.0, 2.0, 5.0, 10.0],
    )
    costs.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)

    benchmark_returns = _benchmark_returns(prices, tickers)
    daily_returns = pd.concat([all_pair_returns, nested_daily.add_prefix("robust_"), benchmark_returns], axis=1).fillna(0.0)
    daily_returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")

    trade_log = pd.concat([hedge_trade_log, nested_trade_log], ignore_index=True) if not nested_trade_log.empty else hedge_trade_log
    trade_log.to_csv(OUTPUT_DIR / "trade_log.csv", index=False)
    trade_analytics = _aggregate_trade_analytics(trade_log)
    trade_analytics.to_csv(OUTPUT_DIR / "trade_analytics.csv", index=False)

    benchmark_table = benchmark_metrics_table(
        daily_returns[
            [
                *[f"all_screened_{mode}_portfolio" for mode in HEDGE_MODES if f"all_screened_{mode}_portfolio" in daily_returns],
                *[f"robust_{col}" for col in nested_daily.columns],
                "cash",
                "SPY_buy_hold",
                "equal_weight_long_only_universe",
            ]
        ],
        daily_returns["SPY_buy_hold"] if "SPY_buy_hold" in daily_returns else None,
    )
    benchmark_table.to_csv(OUTPUT_DIR / "benchmark_relative_metrics.csv", index=False)

    print("Generating charts...")
    chart_columns = [
        *[f"all_screened_{mode}_portfolio" for mode in HEDGE_MODES if f"all_screened_{mode}_portfolio" in daily_returns],
        *[f"robust_{col}" for col in nested_daily.columns],
        "cash",
        "SPY_buy_hold",
        "equal_weight_long_only_universe",
    ]
    plot_equity_curves(daily_returns[chart_columns], OUTPUT_DIR / "equity_curves.png")
    plot_drawdowns(daily_returns[chart_columns], OUTPUT_DIR / "drawdowns.png")
    if not nested_daily.empty:
        wf_plot = nested_daily.rename(columns={nested_daily.columns[0]: "portfolio"})
        plot_walk_forward_performance(wf_plot, OUTPUT_DIR / "walk_forward_performance.png")
        plot_nested_selected_portfolio(nested_daily, OUTPUT_DIR / "nested_selected_portfolio.png")
    plot_hedge_mode_comparison(hedge_comparison, OUTPUT_DIR / "hedge_mode_comparison.png")
    plot_cost_sensitivity(costs, OUTPUT_DIR / "cost_sensitivity.png")
    plot_trade_return_distribution(trade_log, OUTPUT_DIR / "trade_return_distribution.png")
    plot_pair_comparison(hedge_comparison.assign(label=hedge_comparison["pair"] + " " + hedge_comparison["hedge_mode"]), OUTPUT_DIR / "pair_comparison.png")
    if not hedge_candidates.empty:
        first = hedge_candidates.iloc[0]
        preview = run_pair_backtest(
            prices,
            first["ticker_y"],
            first["ticker_x"],
            zscore_window=int(first["zscore_window"]),
            hedge_mode="kalman",
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        plot_spread_zscore(preview.daily, OUTPUT_DIR / "spread_zscore.png", ENTRY_THRESHOLD, EXIT_THRESHOLD)

    print("\nHedge mode comparison")
    print(hedge_comparison.groupby("hedge_mode")[["total_return", "sharpe_ratio", "max_drawdown"]].mean().to_string(float_format=lambda x: f"{x:0.4f}"))
    print("\nRobust selected portfolio")
    print(nested_results[["strategy", "total_return", "cagr", "sharpe_ratio", "max_drawdown"]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nStable candidates: {int(diagnostics['stable_candidate'].sum())}")
    print(f"Saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
