from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ExperimentConfig
from .metrics import benchmark_relative_metrics, performance_metrics
from .portfolio import lagged_volatility_scale, validation_weights


def block_bootstrap_intervals(
    returns: pd.Series,
    simulations: int = 1000,
    block_length: int = 20,
    seed: int = 20260825,
) -> dict[str, float]:
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < block_length * 2:
        return {
            "bootstrap_mean_return_lower": np.nan,
            "bootstrap_mean_return_upper": np.nan,
            "bootstrap_sharpe_lower": np.nan,
            "bootstrap_sharpe_upper": np.nan,
        }
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(clean) - block_length + 1)
    means: list[float] = []
    sharpes: list[float] = []
    blocks_needed = int(np.ceil(len(clean) / block_length))
    for _ in range(simulations):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[start : start + block_length] for start in sampled_starts])[: len(clean)]
        means.append(float(np.mean(sample) * 252))
        volatility = float(np.std(sample, ddof=0) * np.sqrt(252))
        sharpes.append(float(np.mean(sample) * 252 / volatility) if volatility > 0 else 0.0)
    return {
        "bootstrap_mean_return_lower": float(np.quantile(means, 0.025)),
        "bootstrap_mean_return_upper": float(np.quantile(means, 0.975)),
        "bootstrap_sharpe_lower": float(np.quantile(sharpes, 0.025)),
        "bootstrap_sharpe_upper": float(np.quantile(sharpes, 0.975)),
    }


def newey_west_mean_test(returns: pd.Series, max_lags: int = 5) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if len(clean) < max_lags + 5 or clean.std(ddof=0) == 0:
        return {"annualised_mean_return": 0.0, "newey_west_t_stat": 0.0, "newey_west_pvalue": 1.0}
    model = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit()
    robust = model.get_robustcov_results(cov_type="HAC", maxlags=max_lags)
    return {
        "annualised_mean_return": float(model.params[0] * 252),
        "newey_west_t_stat": float(robust.tvalues[0]),
        "newey_west_pvalue": float(robust.pvalues[0]),
    }


def yearly_metrics(returns: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, values in returns.groupby(returns.index.year):
        rows.append({"year": int(year), **performance_metrics(values)})
    return pd.DataFrame(rows)


def pair_contribution_table(pair_daily: pd.DataFrame) -> pd.DataFrame:
    if pair_daily.empty:
        return pd.DataFrame()
    grouped = pair_daily.groupby("pair").agg(
        active_days=("strategy_return", lambda values: int((values != 0).sum())),
        weighted_pnl=("weighted_return", "sum"),
        gross_pnl=("strategy_return_gross", "sum"),
        turnover=("position_change", "sum"),
    )
    total_absolute = float(grouped["weighted_pnl"].abs().sum())
    grouped["absolute_pnl_share"] = grouped["weighted_pnl"].abs() / total_absolute if total_absolute > 0 else 0.0
    return grouped.sort_values("absolute_pnl_share", ascending=False).reset_index()


def research_summary(
    portfolio_daily: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    trade_log: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    pair_contributions: pd.DataFrame,
    spy_returns: pd.Series | None = None,
) -> pd.DataFrame:
    returns = portfolio_daily["strategy_return"].fillna(0.0)
    metrics = performance_metrics(returns)
    relative = benchmark_relative_metrics(returns, spy_returns)
    bootstrap = block_bootstrap_intervals(returns)
    inference = newey_west_mean_test(returns)
    positive_segments = float((segment_metrics["total_return"] > 0).mean()) if not segment_metrics.empty else 0.0
    maximum_pair_share = float(pair_contributions["absolute_pnl_share"].max()) if not pair_contributions.empty else 0.0
    row = {
        **metrics,
        **relative,
        **bootstrap,
        **inference,
        "oos_observations": float(len(returns)),
        "completed_trades": float(len(trade_log)),
        "walk_forward_segments": float(segment_metrics["segment"].nunique()) if not segment_metrics.empty else 0.0,
        "positive_segment_rate": positive_segments,
        "average_selected_pairs": float(selected_pairs.groupby("segment").size().mean()) if not selected_pairs.empty else 0.0,
        "maximum_selected_pairs": float(selected_pairs.groupby("segment").size().max()) if not selected_pairs.empty else 0.0,
        "unique_pairs": float(selected_pairs["pair"].nunique()) if not selected_pairs.empty else 0.0,
        "peer_groups": float(selected_pairs["peer_group"].nunique()) if not selected_pairs.empty else 0.0,
        "maximum_pair_pnl_share": maximum_pair_share,
        "turnover": float(portfolio_daily["pair_turnover"].sum()) if "pair_turnover" in portfolio_daily else 0.0,
    }
    return pd.DataFrame([row])


def fixed_leverage_cost_sensitivity(
    portfolio_daily: pd.DataFrame,
    costs_bps: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for cost in costs_bps:
        returns = portfolio_daily["leverage"] * (
            portfolio_daily["portfolio_gross_return_unscaled"]
            - portfolio_daily["pair_turnover"] * 2.0 * cost / 10000.0
        )
        rows.append({"transaction_cost_bps_per_leg": cost, **performance_metrics(returns)})
    return pd.DataFrame(rows)


def assumed_borrow_cost_sensitivity(
    portfolio_daily: pd.DataFrame,
    trade_log: pd.DataFrame,
    annual_rates: tuple[float, ...],
    assumed_short_gross_fraction: float = 0.50,
) -> pd.DataFrame:
    base_returns = portfolio_daily["strategy_return"].copy()
    daily_weighted_short = pd.Series(0.0, index=base_returns.index)
    if not trade_log.empty:
        for trade in trade_log.itertuples(index=False):
            weight = float(trade.weight)
            if not np.isfinite(weight):
                continue
            active = daily_weighted_short.index.to_series().between(
                pd.Timestamp(trade.entry_date), pd.Timestamp(trade.exit_date)
            )
            daily_weighted_short.loc[active.to_numpy()] += weight * assumed_short_gross_fraction
    rows: list[dict[str, float]] = []
    for rate in annual_rates:
        returns = base_returns - daily_weighted_short * rate / 252.0
        rows.append(
            {
                "assumed_annual_borrow_rate": rate,
                "assumed_short_gross_fraction": assumed_short_gross_fraction,
                "estimated_borrow_cost": float((daily_weighted_short * rate / 252.0).sum()),
                **performance_metrics(returns),
            }
        )
    return pd.DataFrame(rows)


def portfolio_method_comparison(
    selected_pairs: pd.DataFrame,
    validation_daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    config: ExperimentConfig,
    all_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["equal_weight", "constrained_inverse_volatility", "validation_score", "score_risk_budget"]
    daily_by_method: dict[str, list[pd.Series]] = {method: [] for method in methods}
    for segment, selection in selected_pairs.groupby("segment"):
        validation = validation_daily[validation_daily["segment"] == segment].pivot(index="date", columns="pair", values="strategy_return")
        test = pair_daily[pair_daily["segment"] == segment].pivot(index="date", columns="pair", values="strategy_return")
        for method in methods:
            weights = validation_weights(selection, method, config.portfolio.maximum_pair_weight)
            validation_portfolio = validation.reindex(columns=weights.index).fillna(0.0).mul(weights, axis=1).sum(axis=1)
            test_portfolio = test.reindex(columns=weights.index).fillna(0.0).mul(weights, axis=1).sum(axis=1)
            if config.portfolio.target_volatility is not None:
                history = pd.concat([validation_portfolio, test_portfolio])
                scale = lagged_volatility_scale(
                    history,
                    config.portfolio.target_volatility,
                    config.portfolio.volatility_lookback,
                    config.portfolio.maximum_leverage,
                )
                test_portfolio = test_portfolio * scale.reindex(test_portfolio.index).fillna(1.0)
            daily_by_method[method].append(test_portfolio.rename(method))
    daily = pd.concat(
        [pd.concat(parts).sort_index().rename(method) for method, parts in daily_by_method.items() if parts],
        axis=1,
    )
    if all_index is not None:
        daily = daily.reindex(all_index).fillna(0.0)
    comparison = pd.DataFrame(
        [{"method": method, **performance_metrics(daily[method])} for method in daily.columns]
    )
    return comparison, daily


def volatility_target_comparison(
    selected_pairs: pd.DataFrame,
    validation_daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    config: ExperimentConfig,
    targets: tuple[float | None, ...] = (None, 0.05, 0.08, 0.10),
    all_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = config.portfolio.weighting_method
    returns_by_target: dict[str, list[pd.Series]] = {}
    for target in targets:
        label = "unscaled" if target is None else f"target_{int(target * 100)}pct"
        returns_by_target[label] = []
        for segment, selection in selected_pairs.groupby("segment"):
            validation = validation_daily[validation_daily["segment"] == segment].pivot(index="date", columns="pair", values="strategy_return")
            test = pair_daily[pair_daily["segment"] == segment].pivot(index="date", columns="pair", values="strategy_return")
            weights = validation_weights(selection, method, config.portfolio.maximum_pair_weight)
            validation_portfolio = validation.reindex(columns=weights.index).fillna(0.0).mul(weights, axis=1).sum(axis=1)
            test_portfolio = test.reindex(columns=weights.index).fillna(0.0).mul(weights, axis=1).sum(axis=1)
            if target is not None:
                history = pd.concat([validation_portfolio, test_portfolio])
                scale = lagged_volatility_scale(history, target, config.portfolio.volatility_lookback, config.portfolio.maximum_leverage)
                test_portfolio = test_portfolio * scale.reindex(test_portfolio.index).fillna(1.0)
            returns_by_target[label].append(test_portfolio.rename(label))
    daily = pd.concat(
        [pd.concat(parts).sort_index().rename(label) for label, parts in returns_by_target.items() if parts],
        axis=1,
    )
    if all_index is not None:
        daily = daily.reindex(all_index).fillna(0.0)
    comparison = pd.DataFrame(
        [{"target": label, **performance_metrics(daily[label])} for label in daily.columns]
    )
    return comparison, daily
