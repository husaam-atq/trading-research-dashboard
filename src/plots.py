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
    plot_df = backtest_results.set_index("pair")[["cagr", "sharpe_ratio", "max_drawdown"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_df["cagr"].plot(kind="bar", ax=axes[0], color="#1f7a8c", title="CAGR")
    plot_df["sharpe_ratio"].plot(kind="bar", ax=axes[1], color="#6a994e", title="Sharpe")
    plot_df["max_drawdown"].plot(kind="bar", ax=axes[2], color="#b23a48", title="Max Drawdown")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
    _finish(fig, output_path)
