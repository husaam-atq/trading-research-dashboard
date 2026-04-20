# reporting.py

from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_metrics(metrics: dict[str, float]) -> None:
    print_header("PORTFOLIO METRICS")
    for k, v in metrics.items():
        if "Ratio" in k or "Leverage" in k or "Pairs" in k or "Trades" in k or "Turnover" in k:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2%}")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_outputs(
    results_dir: str,
    pair_ranking: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    pair_params: pd.DataFrame,
    portfolio_results: pd.DataFrame,
    pair_results: dict[str, pd.DataFrame],
    pair_trade_logs: dict[str, pd.DataFrame],
    pair_recalibration_logs: dict[str, pd.DataFrame],
) -> None:
    ensure_dir(results_dir)

    pair_ranking.to_csv(os.path.join(results_dir, "pair_ranking.csv"), index=False)
    selected_pairs.to_csv(os.path.join(results_dir, "selected_pairs.csv"), index=False)
    pair_params.to_csv(os.path.join(results_dir, "pair_parameters.csv"), index=False)
    portfolio_results.to_csv(os.path.join(results_dir, "portfolio_results.csv"))

    for pair_name, df in pair_results.items():
        safe_name = pair_name.replace("/", "_")
        df.to_csv(os.path.join(results_dir, f"pair_results__{safe_name}.csv"))

    for pair_name, df in pair_trade_logs.items():
        safe_name = pair_name.replace("/", "_")
        df.to_csv(os.path.join(results_dir, f"trade_log__{safe_name}.csv"), index=False)

    for pair_name, df in pair_recalibration_logs.items():
        safe_name = pair_name.replace("/", "_")
        df.to_csv(os.path.join(results_dir, f"recalibration_log__{safe_name}.csv"), index=False)


def plot_portfolio_results(portfolio_results: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(portfolio_results.index, portfolio_results["equity_curve"], label="Strategy")
    plt.plot(portfolio_results.index, portfolio_results["benchmark_equity_curve"], label="Benchmark")
    plt.title("Portfolio Equity Curve vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(portfolio_results.index, portfolio_results["portfolio_return"])
    plt.title("Daily Portfolio Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.show()


def print_selected_pairs(selected_pairs: pd.DataFrame) -> None:
    print_header("SELECTED PAIRS")
    print(selected_pairs.to_string(index=False))


def print_pair_params(pair_params: pd.DataFrame) -> None:
    print_header("PAIR PARAMETERS")
    print(pair_params.to_string(index=False))