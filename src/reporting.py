from __future__ import annotations

from pathlib import Path

import pandas as pd

from .experiment import WalkForwardResult
from .plots import plot_cost_curve, plot_pair_contributions, plot_research_equity_drawdown, plot_segment_breadth
from .statistics import (
    assumed_borrow_cost_sensitivity,
    fixed_leverage_cost_sensitivity,
    pair_contribution_table,
    research_summary,
    yearly_metrics,
)


def load_walk_forward_result(output_dir: Path) -> WalkForwardResult:
    portfolio = pd.read_csv(output_dir / "portfolio_daily_returns.csv", parse_dates=["date"]).set_index("date")
    return WalkForwardResult(
        boundaries=pd.read_csv(output_dir / "walk_forward_boundaries.csv", parse_dates=["train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end"]),
        candidate_diagnostics=pd.read_csv(output_dir / "candidate_diagnostics.csv"),
        validation_scores=pd.read_csv(output_dir / "validation_scores.csv"),
        selection_decisions=pd.read_csv(output_dir / "selection_decisions.csv"),
        selected_pairs=pd.read_csv(output_dir / "selected_pairs_by_segment.csv"),
        validation_daily_returns=pd.read_csv(output_dir / "validation_daily_returns.csv", parse_dates=["date"]),
        pair_daily_returns=pd.read_csv(output_dir / "pair_daily_returns.csv", parse_dates=["date"]),
        portfolio_daily_returns=portfolio,
        trade_log=pd.read_csv(output_dir / "trade_log.csv"),
        segment_metrics=pd.read_csv(output_dir / "segment_metrics.csv"),
        concentration=pd.read_csv(output_dir / "concentration.csv"),
    )


def save_walk_forward_result(
    result: WalkForwardResult,
    output_dir: Path,
    spy_returns: pd.Series | None,
    transaction_cost_sensitivity: tuple[float, ...],
    borrow_cost_sensitivity: tuple[float, ...],
    title: str,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "walk_forward_boundaries": result.boundaries,
        "candidate_diagnostics": result.candidate_diagnostics,
        "validation_scores": result.validation_scores,
        "selection_decisions": result.selection_decisions,
        "selected_pairs_by_segment": result.selected_pairs,
        "validation_daily_returns": result.validation_daily_returns,
        "pair_daily_returns": result.pair_daily_returns,
        "portfolio_daily_returns": result.portfolio_daily_returns.reset_index(names="date"),
        "trade_log": result.trade_log,
        "segment_metrics": result.segment_metrics,
        "concentration": result.concentration,
    }
    pair_contributions = pair_contribution_table(result.pair_daily_returns)
    tables["pair_contributions"] = pair_contributions
    summary = research_summary(
        result.portfolio_daily_returns,
        result.selected_pairs,
        result.trade_log,
        result.segment_metrics,
        pair_contributions,
        spy_returns,
    )
    tables["performance_metrics"] = summary
    tables["yearly_metrics"] = yearly_metrics(result.portfolio_daily_returns["strategy_return"])
    tables["cost_sensitivity"] = fixed_leverage_cost_sensitivity(
        result.portfolio_daily_returns,
        transaction_cost_sensitivity,
    )
    tables["borrow_cost_sensitivity"] = assumed_borrow_cost_sensitivity(
        result.portfolio_daily_returns,
        result.trade_log,
        borrow_cost_sensitivity,
    )
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    plot_research_equity_drawdown(
        result.portfolio_daily_returns["strategy_return"],
        output_dir / "equity_drawdown.png",
        title,
    )
    plot_segment_breadth(result.segment_metrics, output_dir / "segment_consistency_breadth.png")
    if not pair_contributions.empty:
        plot_pair_contributions(pair_contributions, output_dir / "pair_contributions.png")
    plot_cost_curve(tables["cost_sensitivity"], output_dir / "cost_sensitivity.png")
    return tables


def format_summary_markdown(name: str, summary: pd.Series) -> str:
    return f"""# {name}

| Metric | Result |
| --- | ---: |
| Total return | {summary['total_return']:.2%} |
| CAGR | {summary['cagr']:.2%} |
| Annualised volatility | {summary['annualised_volatility']:.2%} |
| Sharpe | {summary['sharpe_ratio']:.2f} |
| Sortino | {summary['sortino_ratio']:.2f} |
| Maximum drawdown | {summary['max_drawdown']:.2%} |
| Completed OOS trades | {int(summary['completed_trades'])} |
| Average selected pairs | {summary['average_selected_pairs']:.2f} |
| Unique pairs | {int(summary['unique_pairs'])} |
| Peer groups | {int(summary['peer_groups'])} |
| Positive segment rate | {summary['positive_segment_rate']:.1%} |
| Maximum absolute pair P&L share | {summary['maximum_pair_pnl_share']:.1%} |
| Sharpe bootstrap 95% interval | [{summary['bootstrap_sharpe_lower']:.2f}, {summary['bootstrap_sharpe_upper']:.2f}] |
| Newey-West mean-return p-value | {summary['newey_west_pvalue']:.3f} |
"""
