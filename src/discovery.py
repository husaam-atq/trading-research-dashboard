from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .backtest import run_pair_backtest
from .config import DiscoveryConfig
from .pairs import analyse_pair, half_life_z_window


def canonical_pair(ticker_a: str, ticker_b: str) -> str:
    return "/".join(sorted((ticker_a, ticker_b)))


def _subwindows(frame: pd.DataFrame, length: int) -> list[pd.DataFrame]:
    if len(frame) < length:
        return []
    maximum_start = len(frame) - length
    starts = sorted({0, maximum_start // 2, maximum_start})
    return [frame.iloc[start : start + length] for start in starts]


def _empty_record(
    mode: str,
    peer_group: str,
    ticker_y: str,
    ticker_x: str,
    coverage: float,
    correlation: float,
    reason: str,
) -> dict[str, object]:
    return {
        "universe_mode": mode,
        "peer_group": peer_group,
        "pair": canonical_pair(ticker_y, ticker_x),
        "ticker_y": ticker_y,
        "ticker_x": ticker_x,
        "coverage": coverage,
        "correlation": correlation,
        "selected_candidate": False,
        "rejection_reason": reason,
    }


def discover_training_candidates(
    training_prices: pd.DataFrame,
    universes: dict[str, dict[str, list[str]]],
    config: DiscoveryConfig,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Discover pairs using only one segment's training observations."""
    records: list[dict[str, object]] = []

    for mode, peer_groups in universes.items():
        for peer_group, members in peer_groups.items():
            available = [ticker for ticker in members if ticker in training_prices.columns]
            for ticker_y, ticker_x in combinations(available, 2):
                pair_prices = training_prices[[ticker_y, ticker_x]].dropna()
                coverage = len(pair_prices) / max(len(training_prices), 1)
                if len(pair_prices) < config.subwindow_length or coverage < config.minimum_coverage:
                    records.append(
                        _empty_record(mode, peer_group, ticker_y, ticker_x, coverage, np.nan, "insufficient_coverage")
                    )
                    continue

                log_prices = np.log(pair_prices)
                correlation = float(log_prices[ticker_y].corr(log_prices[ticker_x]))
                if not np.isfinite(correlation) or abs(correlation) < config.correlation_prescreen:
                    records.append(
                        _empty_record(mode, peer_group, ticker_y, ticker_x, coverage, correlation, "correlation_prescreen")
                    )
                    continue

                try:
                    full = analyse_pair(pair_prices[ticker_y], pair_prices[ticker_x], zscore_window=0, entry_threshold=1.5)
                except Exception:
                    records.append(_empty_record(mode, peer_group, ticker_y, ticker_x, coverage, correlation, "diagnostic_error"))
                    continue

                sub_stats: list[dict[str, float]] = []
                for subwindow in _subwindows(pair_prices, config.subwindow_length):
                    try:
                        sub_stats.append(
                            analyse_pair(subwindow[ticker_y], subwindow[ticker_x], zscore_window=0, entry_threshold=1.5)
                        )
                    except Exception:
                        continue

                coint_pass_rate = float(
                    np.mean([row["coint_pvalue"] <= config.maximum_coint_pvalue for row in sub_stats])
                ) if sub_stats else 0.0
                adf_pass_rate = float(
                    np.mean([row["adf_pvalue"] <= config.maximum_adf_pvalue for row in sub_stats])
                ) if sub_stats else 0.0
                half_life_pass_rate = float(
                    np.mean(
                        [
                            config.minimum_half_life <= row["half_life"] <= config.maximum_half_life
                            for row in sub_stats
                            if np.isfinite(row["half_life"])
                        ]
                    )
                ) if any(np.isfinite(row["half_life"]) for row in sub_stats) else 0.0
                sub_correlations = np.array([abs(row["correlation"]) for row in sub_stats], dtype=float)
                sub_betas = np.array([row["hedge_ratio"] for row in sub_stats], dtype=float)
                sub_spread_vols = np.array([row["residual_volatility"] for row in sub_stats], dtype=float)
                correlation_mean = float(np.nanmean(sub_correlations)) if sub_correlations.size else np.nan
                correlation_std = float(np.nanstd(sub_correlations)) if sub_correlations.size else np.nan
                beta_mean_abs = float(np.nanmean(np.abs(sub_betas))) if sub_betas.size else np.nan
                hedge_ratio_cv = (
                    float(np.nanstd(sub_betas) / beta_mean_abs)
                    if np.isfinite(beta_mean_abs) and beta_mean_abs > 1e-12
                    else np.inf
                )
                hedge_sign_consistency = (
                    float(max(np.mean(sub_betas >= 0), np.mean(sub_betas < 0))) if sub_betas.size else 0.0
                )
                spread_volatility_ratio = (
                    float(np.nanmax(sub_spread_vols) / np.nanmin(sub_spread_vols))
                    if sub_spread_vols.size and np.nanmin(sub_spread_vols) > 0
                    else np.inf
                )

                z_window = half_life_z_window(float(full["half_life"]))
                training_trade_count = 0.0
                training_sharpe = -np.inf
                training_max_drawdown = -1.0
                training_profit_factor = 0.0
                training_turnover = 0.0
                internal_fit = min(config.subwindow_length, len(pair_prices) // 2)
                if internal_fit >= 126 and len(pair_prices) - internal_fit >= 63:
                    try:
                        viability = run_pair_backtest(
                            pair_prices,
                            ticker_y,
                            ticker_x,
                            entry_threshold=1.5,
                            exit_threshold=0.5,
                            stop_threshold=3.0,
                            zscore_window=z_window,
                            transaction_cost_bps=transaction_cost_bps,
                            max_holding_period=20,
                            hedge_mode="static",
                            fit_window=internal_fit,
                            trade_start=pair_prices.index[internal_fit],
                        )
                        training_trade_count = float(viability.metrics["number_of_trades"])
                        training_sharpe = float(viability.metrics["sharpe_ratio"])
                        training_max_drawdown = float(viability.metrics["max_drawdown"])
                        training_profit_factor = float(viability.metrics["profit_factor"])
                        training_turnover = float(viability.metrics["turnover"])
                    except Exception:
                        pass

                structural_penalty = (
                    min(hedge_ratio_cv, 3.0)
                    + min(correlation_std if np.isfinite(correlation_std) else 1.0, 1.0)
                    + min(max(spread_volatility_ratio - 1.0, 0.0), 3.0) / 3.0
                ) / 3.0
                stability_score = (
                    0.25 * max(correlation_mean if np.isfinite(correlation_mean) else 0.0, 0.0)
                    + 0.25 * coint_pass_rate
                    + 0.20 * adf_pass_rate
                    + 0.10 * half_life_pass_rate
                    + 0.10 * hedge_sign_consistency
                    + 0.10 * max(0.0, 1.0 - structural_penalty)
                )
                activity_score = min(training_trade_count / 8.0, 1.0)
                training_score = (
                    stability_score
                    + 0.15 * np.clip(training_sharpe, -1.0, 1.5)
                    + 0.10 * activity_score
                    + 0.05 * np.clip(training_profit_factor - 1.0, -1.0, 2.0)
                )

                selected = bool(
                    abs(float(full["correlation"])) >= config.minimum_correlation
                    and float(full["coint_pvalue"]) <= config.maximum_coint_pvalue
                    and float(full["adf_pvalue"]) <= config.maximum_adf_pvalue
                    and config.minimum_half_life <= float(full["half_life"]) <= config.maximum_half_life
                    and float(full["threshold_crossings"]) >= config.minimum_crossings
                    and coint_pass_rate >= 1.0 / 3.0
                    and adf_pass_rate >= config.minimum_subwindow_pass_rate
                    and training_trade_count >= config.minimum_training_trades
                    and training_max_drawdown > -0.25
                )
                reason = "passed" if selected else "stability_or_viability_filter"
                records.append(
                    {
                        "universe_mode": mode,
                        "peer_group": peer_group,
                        "pair": canonical_pair(ticker_y, ticker_x),
                        "ticker_y": ticker_y,
                        "ticker_x": ticker_x,
                        "coverage": coverage,
                        **full,
                        "zscore_window": z_window,
                        "subwindow_count": len(sub_stats),
                        "correlation_mean": correlation_mean,
                        "correlation_std": correlation_std,
                        "coint_pass_rate": coint_pass_rate,
                        "adf_pass_rate": adf_pass_rate,
                        "half_life_pass_rate": half_life_pass_rate,
                        "hedge_ratio_cv": hedge_ratio_cv,
                        "hedge_sign_consistency": hedge_sign_consistency,
                        "spread_volatility_ratio": spread_volatility_ratio,
                        "structural_instability_penalty": structural_penalty,
                        "stability_score": stability_score,
                        "training_trade_count": training_trade_count,
                        "training_sharpe": training_sharpe,
                        "training_max_drawdown": training_max_drawdown,
                        "training_profit_factor": training_profit_factor,
                        "training_turnover": training_turnover,
                        "training_score": training_score,
                        "selected_candidate": selected,
                        "rejection_reason": reason,
                    }
                )

    diagnostics = pd.DataFrame(records)
    if diagnostics.empty:
        return diagnostics
    return diagnostics.sort_values(
        ["selected_candidate", "training_score", "stability_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
