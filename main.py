from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import run_pair_backtest
from src.data import DEFAULT_END, DEFAULT_START, DEFAULT_UNIVERSE, download_adjusted_close
from src.pairs import choose_pairs, screen_pairs
from src.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_pair_comparison,
    plot_spread_zscore,
    plot_walk_forward_performance,
)
from src.walk_forward import walk_forward_many

OUTPUT_DIR = Path("outputs")
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
STOP_THRESHOLD = 3.0
TRANSACTION_COST_BPS = 5.0
ROLLING_ZSCORE_WINDOW = 60
SCREENING_DAYS = 756
TOP_N_PAIRS = 5


def _metrics_row(pair: str, result) -> dict[str, float | str]:
    return {"pair": pair, **result.metrics}


def _print_summary(backtest_results: pd.DataFrame, walk_forward_results: pd.DataFrame) -> None:
    print("\nSelected pair backtests")
    print(
        backtest_results[
            ["pair", "total_return", "cagr", "sharpe_ratio", "max_drawdown", "number_of_trades", "win_rate"]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )
    print("\nWalk-forward validation")
    print(
        walk_forward_results[
            ["pair", "total_return", "cagr", "sharpe_ratio", "max_drawdown", "number_of_trades"]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Downloading adjusted close data...")
    prices = download_adjusted_close(DEFAULT_UNIVERSE, DEFAULT_START, DEFAULT_END)

    screening_prices = prices.iloc[: min(SCREENING_DAYS, len(prices))]
    print(f"Screening pairs on first {len(screening_prices)} observations...")
    screening = screen_pairs(screening_prices)
    selected_pairs = choose_pairs(screening, TOP_N_PAIRS)
    screening.to_csv(OUTPUT_DIR / "pair_screening_results.csv", index=False)

    print("Running selected-pair backtests...")
    backtest_rows: list[dict[str, float | str]] = []
    daily_returns: list[pd.Series] = []
    first_result = None

    for row in selected_pairs.itertuples(index=False):
        pair_name = f"{row.ticker_y}/{row.ticker_x}"
        result = run_pair_backtest(
            prices=prices,
            ticker_y=row.ticker_y,
            ticker_x=row.ticker_x,
            entry_threshold=ENTRY_THRESHOLD,
            exit_threshold=EXIT_THRESHOLD,
            stop_threshold=STOP_THRESHOLD,
            zscore_window=ROLLING_ZSCORE_WINDOW,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            hedge_ratio=float(row.hedge_ratio),
            intercept=float(row.intercept),
        )
        if first_result is None:
            first_result = result
        backtest_rows.append(_metrics_row(pair_name, result))
        daily_returns.append(result.daily["strategy_return"].rename(pair_name))

    backtest_results = pd.DataFrame(backtest_rows)
    pair_returns = pd.concat(daily_returns, axis=1).fillna(0.0)
    pair_returns["portfolio"] = pair_returns.mean(axis=1)

    backtest_results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    pair_returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")

    print("Running walk-forward validation...")
    walk_forward_returns, walk_forward_results = walk_forward_many(
        prices=prices,
        pairs=selected_pairs,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        train_window=252,
        test_window=63,
    )
    walk_forward_results.to_csv(OUTPUT_DIR / "walk_forward_results.csv", index=False)

    print("Generating charts...")
    plot_equity_curves(pair_returns, OUTPUT_DIR / "equity_curves.png")
    if first_result is not None:
        plot_spread_zscore(first_result.daily, OUTPUT_DIR / "spread_zscore.png", ENTRY_THRESHOLD, EXIT_THRESHOLD)
    plot_drawdowns(pair_returns, OUTPUT_DIR / "drawdowns.png")
    plot_walk_forward_performance(walk_forward_returns, OUTPUT_DIR / "walk_forward_performance.png")
    plot_pair_comparison(backtest_results, OUTPUT_DIR / "pair_comparison.png")

    _print_summary(backtest_results, walk_forward_results)
    print(f"\nSaved CSV and PNG outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
