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
    plot_metric_bars,
    plot_nested_selected_portfolio,
    plot_pair_comparison,
    plot_spread_zscore,
    plot_trade_return_distribution,
    plot_vol_target_comparison,
    plot_walk_forward_performance,
)
from src.robust import (
    _portfolio_weights,
    _risk_scaled_returns,
    _run_window_result,
    benchmark_metrics_table,
    cost_sensitivity_from_selection,
    nested_pair_selection_portfolio,
    screen_all_modes,
)

OUTPUT_DIR = Path("outputs")

ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
STOP_THRESHOLD = 3.0
ENTRY_GRID = [1.25, 1.5, 1.75, 2.0, 2.25]
EXIT_GRID = [0.0, 0.25, 0.5, 0.75]
STOP_GRID = [2.75, 3.0, 3.5, 4.0]
MAX_HOLD_GRID = [10, 20, 30]
SHARPE_ENTRY_GRID = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
SHARPE_MAX_HOLD_GRID = [5, 10, 15, 20, 30]
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


def _daily_profit_factor(returns: pd.Series) -> float:
    winners = returns[returns > 0].sum()
    losers = returns[returns < 0].sum()
    return float(winners / abs(losers)) if losers < 0 else float("inf") if winners > 0 else 0.0


def _selection_daily_returns(
    prices: pd.DataFrame,
    selection: pd.DataFrame,
    method: str = "inverse_volatility",
    edge_threshold: float | None = None,
    use_volatility_filter: bool | None = None,
    use_correlation_filter: bool | None = None,
    use_trend_filter: bool | None = None,
    cooldown_days: int | None = None,
    volatility_percentile: float | None = None,
) -> pd.Series:
    frames: list[pd.Series] = []
    for _, segment_selection in selection.groupby("segment"):
        test_start = pd.to_datetime(segment_selection["test_start"].iloc[0])
        test_end = pd.to_datetime(segment_selection["test_end"].iloc[0])
        test_index = prices.loc[test_start:test_end].index
        pair_returns: list[pd.Series] = []
        for _, selected_row in segment_selection.iterrows():
            result = _run_window_result(
                prices.loc[:test_end],
                str(selected_row["ticker_y"]),
                str(selected_row["ticker_x"]),
                str(selected_row["hedge_mode"]),
                float(selected_row["entry_threshold"]),
                float(selected_row["exit_threshold"]),
                float(selected_row["stop_threshold"]),
                int(selected_row["max_holding_period"]),
                int(selected_row["zscore_window"]),
                TRANSACTION_COST_BPS,
                bool(selected_row.get("use_volatility_filter", True)) if use_volatility_filter is None else use_volatility_filter,
                float(selected_row["volatility_percentile"]) if volatility_percentile is None else volatility_percentile,
                edge_threshold=float(selected_row.get("edge_threshold", 0.0)) if edge_threshold is None else edge_threshold,
                use_correlation_filter=bool(selected_row.get("use_correlation_filter", False)) if use_correlation_filter is None else use_correlation_filter,
                use_trend_filter=bool(selected_row.get("use_trend_filter", False)) if use_trend_filter is None else use_trend_filter,
                cooldown_days=int(selected_row.get("cooldown_days", 0)) if cooldown_days is None else cooldown_days,
            )
            pair_returns.append(result.daily.loc[result.daily.index.intersection(test_index), "strategy_return"].rename(str(selected_row["pair"])))
        if pair_returns:
            frame = pd.concat(pair_returns, axis=1).fillna(0.0)
            weights = _portfolio_weights(segment_selection, method)
            frames.append(frame.mul(weights.reindex(frame.columns).fillna(0.0), axis=1).sum(axis=1))
    return pd.concat(frames).sort_index() if frames else pd.Series(dtype=float)


def _sensitivity_table(name: str, variants: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, returns in variants.items():
        rows.append({name: variant, **performance_metrics(returns), "profit_factor": _daily_profit_factor(returns)})
    return pd.DataFrame(rows)


def _pair_vol_target_table(pair_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame(index=pair_returns.index)
    rows: list[dict[str, object]] = []
    for target in [0.05, 0.08, 0.10]:
        scaled_pairs = pair_returns.apply(lambda s: _risk_scaled_returns(s, target_vol=target, max_leverage=1.5, window=63))
        portfolio = scaled_pairs.mean(axis=1)
        label = f"pair_vol_target_{int(target * 100)}"
        daily[label] = portfolio
        rows.append({"method": label, "target_vol": target, **performance_metrics(portfolio), "profit_factor": _daily_profit_factor(portfolio)})
    return pd.DataFrame(rows), daily


def _portfolio_vol_target_table(base_returns: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame(index=base_returns.index)
    rows: list[dict[str, object]] = []
    for target in [0.05, 0.08, 0.10]:
        scaled = _risk_scaled_returns(base_returns, target_vol=target, max_leverage=1.5, window=63)
        label = f"portfolio_vol_target_{int(target * 100)}"
        daily[label] = scaled
        rows.append({"method": label, "target_vol": target, **performance_metrics(scaled), "profit_factor": _daily_profit_factor(scaled)})
    return pd.DataFrame(rows), daily


def _sharpe_results(
    returns: pd.DataFrame,
    trade_log: pd.DataFrame,
    spy_returns: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    trade_count = float(len(trade_log)) if not trade_log.empty else 0.0
    avg_hold = float(trade_log["holding_period"].mean()) if not trade_log.empty else 0.0
    for method in returns.columns:
        metrics = performance_metrics(returns[method])
        rel = benchmark_metrics_table(returns[[method]], spy_returns).iloc[0].to_dict()
        rows.append(
            {
                "method": method,
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "annualised_volatility": metrics["annualised_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"],
                "monthly_win_rate": metrics["monthly_win_rate"],
                "profit_factor": _daily_profit_factor(returns[method]),
                "number_of_trades": trade_count,
                "average_holding_period": avg_hold,
                "beta_to_spy": rel["beta_to_spy"],
                "correlation_to_spy": rel["correlation_to_spy"],
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
    nested_selection, nested_results, nested_daily, nested_pair_returns, nested_trade_log, robust_scores = nested_pair_selection_portfolio(
        prices,
        train_window=TRAINING_WINDOW,
        validation_window=VALIDATION_WINDOW,
        test_window=TEST_WINDOW,
        top_n_pairs=ROBUST_TOP_N,
        cost_bps=TRANSACTION_COST_BPS,
        entry_grid=SHARPE_ENTRY_GRID,
        exit_grid=EXIT_GRID,
        stop_grid=STOP_GRID,
        max_hold_grid=SHARPE_MAX_HOLD_GRID,
        hedge_modes=HEDGE_MODES,
        max_validation_candidates=MAX_VALIDATION_CANDIDATES,
        use_vol_filter=True,
        vol_percentiles=(0.90,),
        candidate_pool=robust_candidate_pool,
        step_size=WALK_FORWARD_STEP,
    )
    nested_selection.to_csv(OUTPUT_DIR / "nested_pair_selection.csv", index=False)
    robust_scores.to_csv(OUTPUT_DIR / "robust_selection_scores.csv", index=False)
    robust_scores.to_csv(OUTPUT_DIR / "threshold_optimisation_results.csv", index=False)
    nested_results.to_csv(OUTPUT_DIR / "nested_walk_forward_results.csv", index=False)
    nested_daily.to_csv(OUTPUT_DIR / "nested_walk_forward_daily_returns.csv", index_label="date")
    nested_results.to_csv(OUTPUT_DIR / "pair_portfolio_comparison.csv", index=False)

    print("Running Sharpe-focused sensitivity and volatility targeting...")
    edge_variants = {
        str(edge): _selection_daily_returns(prices, nested_selection, "robust_score_weighted", edge_threshold=edge)
        for edge in [0.0, 0.5, 1.0, 1.5]
    }
    edge_table = _sensitivity_table("edge_threshold", edge_variants)
    edge_table.to_csv(OUTPUT_DIR / "edge_filter_sensitivity.csv", index=False)

    filter_variants = {
        "no_filters": _selection_daily_returns(prices, nested_selection, "robust_score_weighted", use_volatility_filter=False, use_correlation_filter=False, use_trend_filter=False, cooldown_days=0),
        "volatility_only": _selection_daily_returns(prices, nested_selection, "robust_score_weighted", use_volatility_filter=True, use_correlation_filter=False, use_trend_filter=False, cooldown_days=0, volatility_percentile=0.80),
        "correlation_only": _selection_daily_returns(prices, nested_selection, "robust_score_weighted", use_volatility_filter=False, use_correlation_filter=True, use_trend_filter=False, cooldown_days=0),
        "volatility_correlation": _selection_daily_returns(prices, nested_selection, "robust_score_weighted", use_volatility_filter=True, use_correlation_filter=True, use_trend_filter=False, cooldown_days=0, volatility_percentile=0.80),
        "full_filter_set": _selection_daily_returns(prices, nested_selection, "robust_score_weighted", use_volatility_filter=True, use_correlation_filter=True, use_trend_filter=True, cooldown_days=5, volatility_percentile=0.80),
    }
    filter_table = _sensitivity_table("filter_set", filter_variants)
    filter_table.to_csv(OUTPUT_DIR / "filter_sensitivity.csv", index=False)

    pair_vol_results, pair_vol_daily = _pair_vol_target_table(nested_pair_returns) if not nested_pair_returns.empty else (pd.DataFrame(), pd.DataFrame())
    pair_vol_results.to_csv(OUTPUT_DIR / "pair_vol_target_results.csv", index=False)

    portfolio_vol_results, portfolio_vol_daily = _portfolio_vol_target_table(nested_daily["inverse_volatility"]) if "inverse_volatility" in nested_daily else (pd.DataFrame(), pd.DataFrame())
    portfolio_vol_results.to_csv(OUTPUT_DIR / "portfolio_vol_target_results.csv", index=False)

    print("Running transaction cost sensitivity...")
    costs = cost_sensitivity_from_selection(
        prices,
        nested_selection,
        costs=[0.0, 1.0, 2.0, 5.0, 10.0],
    )
    costs.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    costs.rename(columns={"transaction_cost_bps": "cost_bps"}).to_csv(OUTPUT_DIR / "sharpe_cost_sensitivity.csv", index=False)

    benchmark_returns = _benchmark_returns(prices, tickers)
    daily_returns = pd.concat([all_pair_returns, nested_daily.add_prefix("robust_"), benchmark_returns], axis=1).fillna(0.0)
    sharpe_method_returns = pd.concat(
        [
            nested_daily[["inverse_volatility"]].rename(columns={"inverse_volatility": "previous_best_inverse_volatility"}) if "inverse_volatility" in nested_daily else pd.DataFrame(),
            nested_daily[["robust_score_weighted"]].rename(columns={"robust_score_weighted": "robust_score_weighted"}) if "robust_score_weighted" in nested_daily else pd.DataFrame(),
            nested_daily[["sharpe_weighted"]].rename(columns={"sharpe_weighted": "validation_sharpe_weighted"}) if "sharpe_weighted" in nested_daily else pd.DataFrame(),
            nested_daily[["risk_capped_inverse_volatility"]].rename(columns={"risk_capped_inverse_volatility": "risk_capped_inverse_volatility"}) if "risk_capped_inverse_volatility" in nested_daily else pd.DataFrame(),
            pair_vol_daily[["pair_vol_target_8"]].rename(columns={"pair_vol_target_8": "pair_vol_target_8"}) if "pair_vol_target_8" in pair_vol_daily else pd.DataFrame(),
            portfolio_vol_daily[["portfolio_vol_target_8"]].rename(columns={"portfolio_vol_target_8": "portfolio_vol_target_8"}) if "portfolio_vol_target_8" in portfolio_vol_daily else pd.DataFrame(),
            filter_variants["full_filter_set"].rename("combined_filtered_robust_portfolio").to_frame(),
        ],
        axis=1,
    ).fillna(0.0)
    sharpe_results = _sharpe_results(sharpe_method_returns, nested_trade_log, benchmark_returns["SPY_buy_hold"])
    sharpe_results.to_csv(OUTPUT_DIR / "sharpe_optimised_results.csv", index=False)
    daily_returns = pd.concat([daily_returns, sharpe_method_returns.add_prefix("sharpe_")], axis=1).fillna(0.0)
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
    plot_metric_bars(edge_table, "edge_threshold", "sharpe_ratio", OUTPUT_DIR / "edge_filter_sensitivity.png", "Edge Filter Sensitivity")
    plot_metric_bars(filter_table, "filter_set", "sharpe_ratio", OUTPUT_DIR / "filter_sensitivity.png", "Filter Sensitivity")
    plot_vol_target_comparison(pair_vol_results, portfolio_vol_results, OUTPUT_DIR / "vol_target_comparison.png")
    plot_equity_curves(sharpe_method_returns, OUTPUT_DIR / "sharpe_optimised_equity_curve.png")
    plot_drawdowns(sharpe_method_returns, OUTPUT_DIR / "sharpe_optimised_drawdown.png")
    plot_metric_bars(sharpe_results, "method", "sharpe_ratio", OUTPUT_DIR / "sharpe_method_comparison.png", "Sharpe-Optimised Method Comparison")
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
