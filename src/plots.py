from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import drawdown, equity_curve


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(daily_returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for column in daily_returns.columns:
        equity_curve(daily_returns[column]).plot(ax=ax, label=column)
    ax.set_title("Pairs Trading Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, output_path)


def plot_spread_zscore(daily: pd.DataFrame, output_path: Path, entry: float, exit_: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    daily["spread"].plot(ax=axes[0], color="#284b63")
    axes[0].set_title("Spread")
    axes[0].grid(True, alpha=0.25)

    daily["zscore"].plot(ax=axes[1], color="#0b6e4f")
    axes[1].axhline(entry, color="#b23a48", linestyle="--", linewidth=1)
    axes[1].axhline(-entry, color="#b23a48", linestyle="--", linewidth=1)
    axes[1].axhline(exit_, color="#5f6c7b", linestyle=":", linewidth=1)
    axes[1].axhline(-exit_, color="#5f6c7b", linestyle=":", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Rolling Spread Z-Score")
    axes[1].grid(True, alpha=0.25)
    _finish(fig, output_path)


def plot_drawdowns(daily_returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for column in daily_returns.columns:
        drawdown(equity_curve(daily_returns[column])).plot(ax=ax, label=column)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, output_path)


def plot_walk_forward_performance(walk_forward_returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    equity_curve(walk_forward_returns["portfolio"]).plot(ax=ax, color="#1f7a8c")
    ax.set_title("Walk-Forward Equal-Weight Portfolio")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    _finish(fig, output_path)


def plot_pair_comparison(backtest_results: pd.DataFrame, output_path: Path) -> None:
    label_col = "label" if "label" in backtest_results.columns else "pair"
    plot_df = backtest_results.set_index(label_col)[["cagr", "sharpe_ratio", "max_drawdown"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_df["cagr"].plot(kind="bar", ax=axes[0], color="#1f7a8c", title="CAGR")
    plot_df["sharpe_ratio"].plot(kind="bar", ax=axes[1], color="#6a994e", title="Sharpe")
    plot_df["max_drawdown"].plot(kind="bar", ax=axes[2], color="#b23a48", title="Max Drawdown")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
    _finish(fig, output_path)


def plot_threshold_selection(thresholds: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column, title in zip(
        axes,
        ["entry_threshold", "exit_threshold", "stop_threshold"],
        ["Selected Entry", "Selected Exit", "Selected Stop"],
    ):
        if thresholds.empty or column not in thresholds:
            ax.text(0.5, 0.5, "No threshold data", ha="center", va="center")
        else:
            thresholds[column].value_counts().sort_index().plot(kind="bar", ax=ax, color="#1f7a8c")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    _finish(fig, output_path)


def plot_trade_return_distribution(trade_log: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if trade_log.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
    else:
        trade_log["net_return"].plot(kind="hist", bins=30, ax=ax, color="#6a994e", alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Trade Return Distribution")
    ax.set_xlabel("Net trade return")
    ax.grid(True, axis="y", alpha=0.25)
    _finish(fig, output_path)


def plot_hedge_mode_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if comparison.empty:
        ax.text(0.5, 0.5, "No hedge mode data", ha="center", va="center")
    else:
        pivot = comparison.pivot_table(index="hedge_mode", values="sharpe_ratio", aggfunc="mean")
        pivot["sharpe_ratio"].plot(kind="bar", ax=ax, color="#1f7a8c")
    ax.set_title("Average Sharpe by Hedge Mode")
    ax.grid(True, axis="y", alpha=0.25)
    _finish(fig, output_path)


def plot_nested_selected_portfolio(daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for column in daily.columns:
        equity_curve(daily[column]).plot(ax=ax, label=column)
    ax.set_title("Robust Selected Portfolio")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, output_path)


def plot_cost_sensitivity(costs: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if costs.empty:
        ax.text(0.5, 0.5, "No cost data", ha="center", va="center")
    else:
        for strategy, frame in costs.groupby("strategy"):
            frame = frame.sort_values("transaction_cost_bps")
            ax.plot(frame["transaction_cost_bps"], frame["sharpe_ratio"], marker="o", label=strategy)
    ax.set_title("Transaction Cost Sensitivity")
    ax.set_xlabel("Transaction cost, bps per one-way leg")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, output_path)


def plot_metric_bars(frame: pd.DataFrame, label_col: str, metric_col: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        plot_frame = frame.set_index(label_col)[metric_col]
        plot_frame.plot(kind="bar", ax=ax, color="#1f7a8c")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    _finish(fig, output_path)


def plot_vol_target_comparison(pair_results: pd.DataFrame, portfolio_results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    frames = []
    if not pair_results.empty:
        frames.append(pair_results.assign(group="pair"))
    if not portfolio_results.empty:
        frames.append(portfolio_results.assign(group="portfolio"))
    if not frames:
        ax.text(0.5, 0.5, "No vol target data", ha="center", va="center")
    else:
        data = pd.concat(frames, ignore_index=True)
        for group, frame in data.groupby("group"):
            ax.plot(frame["target_vol"] * 100, frame["sharpe_ratio"], marker="o", label=group)
    ax.set_title("Volatility Target Comparison")
    ax.set_xlabel("Target annualised volatility (%)")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _finish(fig, output_path)
