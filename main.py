from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_pair_backtest, trade_metrics
from src.data import DEFAULT_END, DEFAULT_START, PEER_GROUP_UNIVERSES, download_adjusted_close
from src.metrics import performance_metrics
from src.pairs import choose_pairs, screen_peer_groups
from src.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_pair_comparison,
    plot_spread_zscore,
    plot_threshold_selection,
    plot_trade_return_distribution,
    plot_walk_forward_performance,
)
from src.walk_forward import walk_forward_many

OUTPUT_DIR = Path("outputs")

ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
STOP_THRESHOLD = 3.0
ENTRY_GRID = [1.5, 2.0, 2.5]
EXIT_GRID = [0.0, 0.5, 1.0]
STOP_GRID = [3.0, 3.5, 4.0]

TRANSACTION_COST_BPS = 5.0
ROLLING_ZSCORE_WINDOW = 60
SCREENING_DAYS = 756
TOP_N_PAIRS = 8
CORRELATION_THRESHOLD = 0.80
COINTEGRATION_PVALUE_THRESHOLD = 0.05
MIN_HALF_LIFE = 5.0
MAX_HALF_LIFE = 60.0
MIN_THRESHOLD_CROSSINGS = 4
MAX_HOLDING_PERIOD = 20
ROLLING_HEDGE_WINDOW = 252
WALK_FORWARD_TRAIN_WINDOW = 252
WALK_FORWARD_TEST_WINDOW = 63
MIN_TRAINING_TRADES = 3


def _all_research_tickers() -> list[str]:
    tickers: list[str] = []
    for group in PEER_GROUP_UNIVERSES.values():
        tickers.extend(group)
    tickers.append("SPY")
    return sorted(dict.fromkeys(tickers))


def _metrics_row(pair: str, peer_group: str, hedge_mode: str, result) -> dict[str, float | str]:
    label = f"{pair} ({hedge_mode})"
    return {"pair": pair, "label": label, "peer_group": peer_group, "hedge_mode": hedge_mode, **result.metrics}


def _benchmark_returns(prices: pd.DataFrame, universe_tickers: list[str]) -> pd.DataFrame:
    returns = pd.DataFrame(index=prices.index)
    returns["cash"] = 0.0
    if "SPY" in prices:
        returns["SPY_buy_hold"] = prices["SPY"].pct_change().fillna(0.0)
    available = [ticker for ticker in universe_tickers if ticker in prices.columns and ticker != "SPY"]
    if available:
        returns["equal_weight_long_only_universe"] = prices[available].pct_change().mean(axis=1).fillna(0.0)
    return returns


def _trade_analytics(trade_log: pd.DataFrame, backtest_daily: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (pair, hedge_mode), daily in backtest_daily.items():
        trades = trade_log[(trade_log["pair"] == pair) & (trade_log["hedge_mode"] == hedge_mode)].copy()
        metrics = trade_metrics(daily, trades)
        rows.append({"pair": pair, "hedge_mode": hedge_mode, **metrics})
    return pd.DataFrame(rows)


def _print_summary(backtest_results: pd.DataFrame, walk_forward_results: pd.DataFrame) -> None:
    print("\nSelected pair backtests")
    print(
        backtest_results[
            ["pair", "peer_group", "hedge_mode", "total_return", "cagr", "sharpe_ratio", "max_drawdown", "number_of_trades"]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )
    print("\nNested walk-forward validation")
    print(
        walk_forward_results[
            ["pair", "peer_group", "total_return", "cagr", "sharpe_ratio", "max_drawdown", "number_of_trades"]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    research_tickers = _all_research_tickers()
    print("Downloading adjusted close data...")
    prices = download_adjusted_close(research_tickers, DEFAULT_START, DEFAULT_END)

    screening_prices = prices.iloc[: min(SCREENING_DAYS, len(prices))]
    print(f"Screening peer-group pairs on first {len(screening_prices)} observations...")
    screening = screen_peer_groups(
        screening_prices,
        PEER_GROUP_UNIVERSES,
        min_abs_correlation=CORRELATION_THRESHOLD,
        max_coint_pvalue=COINTEGRATION_PVALUE_THRESHOLD,
        min_half_life=MIN_HALF_LIFE,
        max_half_life=MAX_HALF_LIFE,
        min_threshold_crossings=MIN_THRESHOLD_CROSSINGS,
        zscore_window=ROLLING_ZSCORE_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
    )
    selected_pairs = choose_pairs(screening, TOP_N_PAIRS)
    screening.to_csv(OUTPUT_DIR / "pair_screening_results.csv", index=False)

    print("Running static and rolling hedge-ratio backtests...")
    backtest_rows: list[dict[str, float | str]] = []
    return_frames: list[pd.Series] = []
    trade_logs: list[pd.DataFrame] = []
    backtest_daily: dict[tuple[str, str], pd.DataFrame] = {}
    first_result = None

    for row in selected_pairs.itertuples(index=False):
        pair_name = f"{row.ticker_y}/{row.ticker_x}"
        for hedge_mode in ["static", "rolling"]:
            result = run_pair_backtest(
                prices=prices,
                ticker_y=row.ticker_y,
                ticker_x=row.ticker_x,
                entry_threshold=ENTRY_THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
                stop_threshold=STOP_THRESHOLD,
                zscore_window=ROLLING_ZSCORE_WINDOW,
                transaction_cost_bps=TRANSACTION_COST_BPS,
                max_holding_period=MAX_HOLDING_PERIOD,
                hedge_mode=hedge_mode,
                hedge_training_window=ROLLING_HEDGE_WINDOW,
                hedge_ratio=float(row.hedge_ratio) if hedge_mode == "static" else None,
                intercept=float(row.intercept) if hedge_mode == "static" else None,
                fit_window=SCREENING_DAYS,
            )
            if first_result is None:
                first_result = result
            backtest_rows.append(_metrics_row(pair_name, row.peer_group, hedge_mode, result))
            return_frames.append(result.daily["strategy_return"].rename(f"{pair_name}_{hedge_mode}"))
            backtest_daily[(pair_name, hedge_mode)] = result.daily
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["peer_group"] = row.peer_group
                trades["hedge_mode"] = hedge_mode
                trade_logs.append(trades)

    backtest_results = pd.DataFrame(backtest_rows)
    strategy_returns = pd.concat(return_frames, axis=1).fillna(0.0)
    static_cols = [col for col in strategy_returns.columns if col.endswith("_static")]
    rolling_cols = [col for col in strategy_returns.columns if col.endswith("_rolling")]
    strategy_returns["static_pairs_portfolio"] = strategy_returns[static_cols].mean(axis=1) if static_cols else 0.0
    strategy_returns["rolling_pairs_portfolio"] = strategy_returns[rolling_cols].mean(axis=1) if rolling_cols else 0.0

    benchmark_returns = _benchmark_returns(prices, research_tickers)
    daily_returns = pd.concat([strategy_returns, benchmark_returns], axis=1).fillna(0.0)
    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()
    trade_analytics = _trade_analytics(trade_log, backtest_daily)

    benchmark_rows = []
    for column in benchmark_returns.columns:
        metrics = performance_metrics(benchmark_returns[column])
        benchmark_rows.append({"pair": column, "label": column, "peer_group": "benchmark", "hedge_mode": "benchmark", **metrics})
    backtest_results = pd.concat([backtest_results, pd.DataFrame(benchmark_rows)], ignore_index=True)

    backtest_results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    daily_returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    trade_log.to_csv(OUTPUT_DIR / "trade_log.csv", index=False)
    trade_analytics.to_csv(OUTPUT_DIR / "trade_analytics.csv", index=False)

    print("Running nested walk-forward validation...")
    walk_forward_returns, walk_forward_results, threshold_results = walk_forward_many(
        prices=prices,
        pairs=selected_pairs,
        entry_grid=ENTRY_GRID,
        exit_grid=EXIT_GRID,
        stop_grid=STOP_GRID,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        max_holding_period=MAX_HOLDING_PERIOD,
        zscore_window=ROLLING_ZSCORE_WINDOW,
        train_window=WALK_FORWARD_TRAIN_WINDOW,
        test_window=WALK_FORWARD_TEST_WINDOW,
        min_trades=MIN_TRAINING_TRADES,
    )
    walk_forward_results.to_csv(OUTPUT_DIR / "walk_forward_results.csv", index=False)
    threshold_results.to_csv(OUTPUT_DIR / "walk_forward_thresholds.csv", index=False)

    print("Generating charts...")
    plot_equity_curves(
        daily_returns[["static_pairs_portfolio", "rolling_pairs_portfolio", "cash", "SPY_buy_hold", "equal_weight_long_only_universe"]],
        OUTPUT_DIR / "equity_curves.png",
    )
    if first_result is not None:
        plot_spread_zscore(first_result.daily, OUTPUT_DIR / "spread_zscore.png", ENTRY_THRESHOLD, EXIT_THRESHOLD)
    plot_drawdowns(
        daily_returns[["static_pairs_portfolio", "rolling_pairs_portfolio", "cash", "SPY_buy_hold", "equal_weight_long_only_universe"]],
        OUTPUT_DIR / "drawdowns.png",
    )
    plot_walk_forward_performance(walk_forward_returns, OUTPUT_DIR / "walk_forward_performance.png")
    plot_pair_comparison(backtest_results[backtest_results["hedge_mode"].isin(["static", "rolling"])], OUTPUT_DIR / "pair_comparison.png")
    plot_threshold_selection(threshold_results, OUTPUT_DIR / "threshold_selection.png")
    plot_trade_return_distribution(trade_log, OUTPUT_DIR / "trade_return_distribution.png")

    _print_summary(
        backtest_results[backtest_results["hedge_mode"].isin(["static", "rolling"])],
        walk_forward_results,
    )
    strict_count = int(screening["selected_candidate"].sum())
    print(f"\nStrict filter survivors: {strict_count}")
    print(f"Saved CSV and PNG outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
