# pipeline.py

from __future__ import annotations

import pandas as pd

from data_loader import download_prices, get_pair_frame
from pair_selection import rank_pairs
from core import analyse_pair, annualised_return, sharpe_ratio, max_drawdown
from strategy import tune_parameters_for_pair, generate_positions, simple_pair_returns
from portfolio import build_dynamic_portfolio_from_candidates


def _split_pair_df(
    pair_df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to less than 1.")

    n = len(pair_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))

    if train_end < 50 or val_end <= train_end or val_end >= n:
        raise ValueError("Not enough data for train/validation/test split.")

    train_df = pair_df.iloc[:train_end].copy()
    val_df = pair_df.iloc[train_end:val_end].copy()
    test_df = pair_df.iloc[val_end:].copy()
    test_start_date = pair_df.index[val_end]

    return train_df, val_df, test_df, test_start_date


def _score_on_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict,
    commission_bps: float,
    slippage_bps: float,
    require_entry_cross: bool,
) -> dict:
    train_stats = analyse_pair(train_df["y"], train_df["x"])

    val_spread = val_df["y"] - (train_stats.intercept + train_stats.hedge_ratio * val_df["x"])
    lookback = int(params["lookback"])
    val_z = (
        (val_spread - val_spread.rolling(lookback).mean())
        / val_spread.rolling(lookback).std(ddof=0)
    )

    val_pos = generate_positions(
        zscore=val_z,
        entry_z=float(params["entry_z"]),
        exit_z=float(params["exit_z"]),
        stop_z=float(params["stop_z"]),
        require_crossing=require_entry_cross,
    )

    val_returns = simple_pair_returns(
        y=val_df["y"],
        x=val_df["x"],
        hedge_ratio=train_stats.hedge_ratio,
        position=val_pos,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )

    val_sharpe = sharpe_ratio(val_returns)
    val_ann_return = annualised_return(val_returns)
    val_total_return = float((1.0 + val_returns.fillna(0.0)).prod() - 1.0)
    val_trade_days = int((val_pos.shift(1).fillna(0) != 0).sum())
    val_equity = (1.0 + val_returns.fillna(0.0)).cumprod()
    val_max_dd = max_drawdown(val_equity)

    validation_score = (
        0.60 * val_sharpe
        + 0.25 * val_ann_return
        + 0.20 * val_total_return
        - 0.20 * abs(val_max_dd)
        + 0.05 * min(val_trade_days / 20.0, 1.0)
    )

    return {
        "validation_score": validation_score,
        "validation_sharpe": val_sharpe,
        "validation_ann_return": val_ann_return,
        "validation_total_return": val_total_return,
        "validation_trade_days": val_trade_days,
        "validation_max_drawdown": val_max_dd,
    }


def run_research_pipeline(
    universe: list[str],
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
    train_ratio: float,
    validation_ratio: float,
    min_observations: int,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    top_n_pairs: int,
    lookback_grid: list[int],
    entry_z_grid: list[float],
    exit_z_grid: list[float],
    stop_z_grid: list[float],
    recalibration_window: int,
    recalibration_step: int,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    target_portfolio_vol: float,
    max_gross_leverage: float,
    max_pair_weight: float,
    min_pair_vol: float,
    risk_free_rate: float,
    cooldown_days: int,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    require_entry_cross: bool,
    quality_weight_strength: float,
) -> dict:
    tickers = sorted(set(universe + [benchmark_ticker]))
    prices = download_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    asset_prices = prices[[c for c in universe if c in prices.columns]].copy()

    if benchmark_ticker not in prices.columns:
        raise ValueError(f"Benchmark ticker '{benchmark_ticker}' was not found in downloaded data.")

    benchmark = prices[benchmark_ticker].dropna().copy()

    ranking = rank_pairs(
        prices=asset_prices,
        tickers=list(asset_prices.columns),
        train_ratio=train_ratio,
        max_coint_pvalue=max_coint_pvalue,
        max_adf_pvalue=max_adf_pvalue,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_observations=min_observations,
    )

    valid_candidates = ranking[ranking["valid"]].copy()
    if valid_candidates.empty:
        raise ValueError("No valid pairs passed screening.")

    candidate_pool_size = min(max(top_n_pairs * 5, top_n_pairs), len(valid_candidates))
    candidate_pairs = valid_candidates.head(candidate_pool_size).copy()

    validation_rows = []

    for _, row in candidate_pairs.iterrows():
        ticker_y = row["ticker_y"]
        ticker_x = row["ticker_x"]
        pair_name = f"{ticker_y}__{ticker_x}"

        pair_df = get_pair_frame(asset_prices, ticker_y, ticker_x)
        train_df, val_df, _, _ = _split_pair_df(
            pair_df=pair_df,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        train_stats = analyse_pair(train_df["y"], train_df["x"])

        best_params = tune_parameters_for_pair(
            spread=train_stats.spread,
            y=train_df["y"],
            x=train_df["x"],
            hedge_ratio=train_stats.hedge_ratio,
            lookback_grid=lookback_grid,
            entry_z_grid=entry_z_grid,
            exit_z_grid=exit_z_grid,
            stop_z_grid=stop_z_grid,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            require_crossing=require_entry_cross,
        )

        val_scores = _score_on_validation(
            train_df=train_df,
            val_df=val_df,
            params=best_params,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            require_entry_cross=require_entry_cross,
        )

        validation_rows.append({
            "pair_name": pair_name,
            "ticker_y": ticker_y,
            "ticker_x": ticker_x,
            "train_coint_pvalue": row["coint_pvalue"],
            "train_adf_pvalue": row["adf_pvalue"],
            "train_half_life": row["half_life"],
            **best_params,
            **val_scores,
        })

    validation_df = pd.DataFrame(validation_rows).sort_values(
        by=["validation_score", "validation_sharpe", "validation_ann_return", "validation_total_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    if validation_df.empty:
        raise ValueError("No candidate pairs were scored on validation.")

    # Keep a stricter subset for actual deployment/use
    deployable_validation_df = validation_df[
        (validation_df["validation_score"] > 0)
        & (
            (validation_df["validation_sharpe"] > 0)
            | (validation_df["validation_total_return"] > 0)
        )
    ].copy()

    # Fallback: if nothing is deployable, keep the original for transparency,
    # but the portfolio may end up staying in cash.
    if deployable_validation_df.empty:
        deployable_validation_df = validation_df.head(top_n_pairs).copy()

    selected_pairs = deployable_validation_df.head(top_n_pairs).copy()

    first_pair = get_pair_frame(
        asset_prices,
        selected_pairs.iloc[0]["ticker_y"],
        selected_pairs.iloc[0]["ticker_x"],
    )
    _, _, _, test_start_date = _split_pair_df(
        pair_df=first_pair,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    portfolio_output = build_dynamic_portfolio_from_candidates(
        prices=asset_prices,
        candidate_pairs_df=deployable_validation_df,
        benchmark=benchmark,
        test_start_date=test_start_date,
        recalibration_window=recalibration_window,
        recalibration_step=recalibration_step,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        target_portfolio_vol=target_portfolio_vol,
        max_gross_leverage=max_gross_leverage,
        max_pair_weight=max_pair_weight,
        min_pair_vol=min_pair_vol,
        risk_free_rate=risk_free_rate,
        top_n_pairs=top_n_pairs,
        max_coint_pvalue=max_coint_pvalue,
        max_adf_pvalue=max_adf_pvalue,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        cooldown_days=cooldown_days,
        trend_lookback=trend_lookback,
        max_trend_zscore_slope=max_trend_zscore_slope,
        vol_filter_lookback=vol_filter_lookback,
        max_recent_vol_multiplier=max_recent_vol_multiplier,
        require_entry_cross=require_entry_cross,
        quality_weight_strength=quality_weight_strength,
    )

    split_summary = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": 1.0 - train_ratio - validation_ratio,
        "test_start_date": test_start_date,
    }

    return {
        "prices": prices,
        "asset_prices": asset_prices,
        "benchmark": benchmark,
        "ranking": ranking,
        "candidate_pairs": candidate_pairs,
        "validation_df": validation_df,
        "selected_pairs": selected_pairs,
        "split_date": test_start_date,
        "split_summary": split_summary,
        "pair_params": {
            row["pair_name"]: {
                "lookback": row["lookback"],
                "entry_z": row["entry_z"],
                "exit_z": row["exit_z"],
                "stop_z": row["stop_z"],
            }
            for _, row in validation_df.iterrows()
        },
        "pair_params_df": validation_df.copy(),
        "portfolio_output": portfolio_output,
    }