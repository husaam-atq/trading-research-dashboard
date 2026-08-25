from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import PortfolioConfig, ValidationConfig


def capped_weights(raw_scores: pd.Series, maximum_weight: float) -> pd.Series:
    scores = raw_scores.astype(float).clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scores.empty:
        return scores
    if scores.sum() <= 0:
        scores[:] = 1.0

    effective_cap = max(float(maximum_weight), 1.0 / len(scores))
    weights = pd.Series(0.0, index=scores.index)
    remaining = list(scores.index)
    remaining_mass = 1.0
    while remaining:
        remaining_scores = scores.loc[remaining]
        if remaining_scores.sum() <= 0:
            proposal = pd.Series(remaining_mass / len(remaining), index=remaining)
        else:
            proposal = remaining_scores / remaining_scores.sum() * remaining_mass
        above = proposal[proposal > effective_cap + 1e-12]
        if above.empty:
            weights.loc[remaining] = proposal
            break
        for label in above.index:
            weights.loc[label] = effective_cap
            remaining.remove(label)
            remaining_mass -= effective_cap
        if remaining_mass <= 1e-12:
            break
    return weights / weights.sum()


def select_diversified_pairs(
    validation_summary: pd.DataFrame,
    validation_returns: pd.DataFrame,
    validation_config: ValidationConfig,
    portfolio_config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_summary.empty:
        return validation_summary.copy(), pd.DataFrame()

    eligible = validation_summary[
        (validation_summary["validation_trades"] >= validation_config.minimum_trades)
        & (validation_summary["validation_sharpe"] > validation_config.minimum_sharpe)
        & (validation_summary["validation_profit_factor"] > validation_config.minimum_profit_factor)
        & (validation_summary["validation_max_drawdown"] > validation_config.minimum_max_drawdown)
    ].sort_values(["robust_score", "validation_sharpe", "stability_score"], ascending=False)

    selected_rows: list[pd.Series] = []
    decisions: list[dict[str, object]] = []
    ticker_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    selected_pairs: list[str] = []

    for _, row in eligible.iterrows():
        reason = "selected"
        tickers = (str(row["ticker_y"]), str(row["ticker_x"]))
        if any(ticker_counts[ticker] >= portfolio_config.maximum_pairs_per_ticker for ticker in tickers):
            reason = "ticker_concentration"
        elif group_counts[str(row["peer_group"])] >= portfolio_config.maximum_pairs_per_peer_group:
            reason = "peer_group_concentration"
        elif str(row["pair"]) in selected_pairs:
            reason = "duplicate_pair"
        elif selected_pairs and str(row["pair"]) in validation_returns:
            correlations = validation_returns[selected_pairs].corrwith(validation_returns[str(row["pair"])]).abs()
            if (correlations > portfolio_config.maximum_pair_return_correlation).any():
                reason = "pair_return_correlation"

        decisions.append({"pair": row["pair"], "selected": reason == "selected", "selection_decision": reason})
        if reason != "selected":
            continue
        selected_rows.append(row)
        selected_pairs.append(str(row["pair"]))
        ticker_counts.update(tickers)
        group_counts.update([str(row["peer_group"])])
        if len(selected_rows) >= portfolio_config.top_n_pairs:
            break

    selected = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else eligible.head(0).copy()
    return selected, pd.DataFrame(decisions)


def validation_weights(selection: pd.DataFrame, method: str, maximum_weight: float) -> pd.Series:
    if selection.empty:
        return pd.Series(dtype=float)
    index = selection["pair"].astype(str)
    volatility = selection["validation_volatility"].replace(0.0, np.nan)
    fallback_vol = float(volatility.median()) if volatility.notna().any() else 1.0
    volatility = volatility.fillna(fallback_vol).clip(lower=1e-8)

    if method == "equal_weight":
        raw = pd.Series(1.0, index=index)
    elif method == "validation_score":
        raw = selection["robust_score"].clip(lower=0.0)
        raw.index = index
    elif method == "score_risk_budget":
        raw = selection["robust_score"].clip(lower=0.0) / volatility
        raw.index = index
    else:
        raw = 1.0 / volatility
        raw.index = index
    return capped_weights(raw, maximum_weight)


def lagged_volatility_scale(
    returns: pd.Series,
    target_volatility: float,
    lookback: int,
    maximum_leverage: float,
) -> pd.Series:
    trailing = returns.rolling(lookback).std(ddof=0) * np.sqrt(252)
    return (
        target_volatility / trailing.replace(0.0, np.nan)
    ).shift(1).clip(lower=0.0, upper=maximum_leverage).fillna(1.0)


def concentration_metrics(selection: pd.DataFrame, pair_pnl: pd.Series | None = None) -> dict[str, float]:
    if selection.empty:
        return {
            "selected_pairs": 0.0,
            "unique_tickers": 0.0,
            "peer_groups": 0.0,
            "maximum_pair_weight": 0.0,
            "pair_weight_hhi": 0.0,
            "maximum_ticker_weight": 0.0,
            "maximum_pair_pnl_share": 0.0,
        }
    weights = selection.set_index("pair")["weight"].astype(float)
    ticker_weights: Counter[str] = Counter()
    for row in selection.itertuples(index=False):
        pair_weight = float(row.weight)
        ticker_weights[str(row.ticker_y)] += pair_weight / 2.0
        ticker_weights[str(row.ticker_x)] += pair_weight / 2.0
    pnl_share = 0.0
    if pair_pnl is not None and pair_pnl.abs().sum() > 0:
        pnl_share = float((pair_pnl.abs() / pair_pnl.abs().sum()).max())
    return {
        "selected_pairs": float(len(selection)),
        "unique_tickers": float(len(ticker_weights)),
        "peer_groups": float(selection["peer_group"].nunique()),
        "maximum_pair_weight": float(weights.max()),
        "pair_weight_hhi": float((weights**2).sum()),
        "maximum_ticker_weight": float(max(ticker_weights.values())),
        "maximum_pair_pnl_share": pnl_share,
    }
