# portfolio.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from data_loader import get_pair_frame
from core import (
    analyse_pair,
    compute_spread,
    rolling_zscore,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    realised_volatility,
    cap_weights,
)
from strategy import generate_positions, simple_pair_returns, compute_trade_filter


@dataclass
class PairBlockOutput:
    pair_name: str
    block_results: pd.DataFrame
    trade_log: pd.DataFrame
    recalibration_log: pd.DataFrame
    hist_vol: float
    current_coint_pvalue: float
    current_adf_pvalue: float
    current_half_life: float
    validation_sharpe: float
    validation_total_return: float


@dataclass
class PortfolioBacktestOutput:
    portfolio_results: pd.DataFrame
    pair_results: dict[str, pd.DataFrame]
    pair_trade_logs: dict[str, pd.DataFrame]
    pair_recalibration_logs: dict[str, pd.DataFrame]
    selection_log: pd.DataFrame
    metrics: dict[str, float]


def information_ratio(active_returns: pd.Series) -> float:
    r = active_returns.dropna()
    if r.empty:
        return 0.0
    vol = r.std(ddof=0)
    if vol == 0:
        return 0.0
    return float((r.mean() / vol) * np.sqrt(252))


def build_trade_log(
    position: pd.Series,
    spread: pd.Series,
    zscore: pd.Series,
    net_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    current = None

    if net_returns is None:
        net_returns = pd.Series(0.0, index=position.index)

    for dt in position.index:
        pos = float(position.loc[dt])

        if current is None and pos != 0:
            current = {
                "entry_date": dt,
                "entry_position": pos,
                "entry_spread": float(spread.loc[dt]) if pd.notna(spread.loc[dt]) else np.nan,
                "entry_zscore": float(zscore.loc[dt]) if pd.notna(zscore.loc[dt]) else np.nan,
                "trade_return": 0.0,
            }

        if current is not None:
            current["trade_return"] += float(net_returns.loc[dt]) if pd.notna(net_returns.loc[dt]) else 0.0

        if current is not None and pos == 0:
            current["exit_date"] = dt
            current["exit_spread"] = float(spread.loc[dt]) if pd.notna(spread.loc[dt]) else np.nan
            current["exit_zscore"] = float(zscore.loc[dt]) if pd.notna(zscore.loc[dt]) else np.nan
            current["holding_days"] = (dt - current["entry_date"]).days
            rows.append(current)
            current = None

    return pd.DataFrame(rows)


def _pair_weight_multiplier(
    validation_sharpe: float,
    validation_total_return: float,
    current_coint_pvalue: float,
    current_adf_pvalue: float,
    current_half_life: float,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    quality_weight_strength: float,
) -> float:
    val_component = np.clip(
        1.0 + 0.35 * validation_sharpe + 0.75 * validation_total_return,
        0.25,
        2.0,
    )

    coint_component = np.clip(1.0 - current_coint_pvalue / max(max_coint_pvalue, 1e-12), 0.0, 1.0)
    adf_component = np.clip(1.0 - current_adf_pvalue / max(max_adf_pvalue, 1e-12), 0.0, 1.0)

    mid_half = 0.5 * (min_half_life + max_half_life)
    half_range = max((max_half_life - min_half_life) / 2.0, 1e-12)
    half_life_component = np.clip(1.0 - abs(current_half_life - mid_half) / half_range, 0.0, 1.0)

    stability_component = 0.34 * coint_component + 0.33 * adf_component + 0.33 * half_life_component
    multiplier = val_component * (0.5 + stability_component)

    return float(max(0.20, multiplier ** max(quality_weight_strength, 0.0)))


def _simulate_historical_vol(
    calibration_sample: pd.DataFrame,
    hedge_ratio: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    commission_bps: float,
    slippage_bps: float,
    min_pair_vol: float,
    cooldown_days: int,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    require_entry_cross: bool,
) -> float:
    hist_spread = calibration_sample["spread_hist"]
    hist_z = rolling_zscore(hist_spread, lookback)

    baseline_spread_vol = float(hist_spread.diff().std(ddof=0))
    baseline_spread_vol = max(baseline_spread_vol, 1e-12)

    hist_filter = compute_trade_filter(
        spread=hist_spread,
        trend_lookback=trend_lookback,
        max_trend_zscore_slope=max_trend_zscore_slope,
        vol_filter_lookback=vol_filter_lookback,
        max_recent_vol_multiplier=max_recent_vol_multiplier,
        baseline_spread_vol=baseline_spread_vol,
    )

    hist_pos = generate_positions(
        zscore=hist_z,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        can_trade=hist_filter["can_trade"],
        cooldown_days=cooldown_days,
        require_crossing=require_entry_cross,
    )

    hist_returns = simple_pair_returns(
        y=calibration_sample["y"],
        x=calibration_sample["x"],
        hedge_ratio=hedge_ratio,
        position=hist_pos,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    return realised_volatility(hist_returns, min_vol=min_pair_vol)


def _simulate_pair_block(
    pair_df: pd.DataFrame,
    pair_name: str,
    params: dict,
    validation_sharpe: float,
    validation_total_return: float,
    block_dates: list[pd.Timestamp],
    recalibration_window: int,
    commission_bps: float,
    slippage_bps: float,
    min_pair_vol: float,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    cooldown_days: int,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    require_entry_cross: bool,
) -> PairBlockOutput | None:
    df = pair_df.copy().dropna()
    if df.empty:
        return None

    available_block_dates = [d for d in block_dates if d in df.index]
    if len(available_block_dates) < 5:
        return None

    block_start = available_block_dates[0]
    block_end = available_block_dates[-1]

    start_loc = df.index.get_loc(block_start)
    end_loc = df.index.get_loc(block_end)

    hist_start_loc = max(0, start_loc - recalibration_window)
    calibration_sample = df.iloc[hist_start_loc:start_loc].copy()
    if len(calibration_sample) < max(60, int(params["lookback"]) + 10):
        return None

    current_stats = analyse_pair(calibration_sample["y"], calibration_sample["x"])

    is_currently_valid = (
        current_stats.coint_pvalue <= max_coint_pvalue
        and current_stats.adf_pvalue <= max_adf_pvalue
        and not np.isnan(current_stats.half_life)
        and min_half_life <= current_stats.half_life <= max_half_life
    )

    if not is_currently_valid:
        return None
    if validation_sharpe < 0 and validation_total_return < 0:
        return None

    lookback = int(params["lookback"])
    entry_z = float(params["entry_z"])
    exit_z = float(params["exit_z"])
    stop_z = float(params["stop_z"])

    window_start_loc = max(0, start_loc - max(lookback, trend_lookback, vol_filter_lookback) + 1)
    window_df = df.iloc[window_start_loc:end_loc + 1].copy()

    spread_all = compute_spread(
        y=window_df["y"],
        x=window_df["x"],
        hedge_ratio=current_stats.hedge_ratio,
        intercept=current_stats.intercept,
    )
    z_all = rolling_zscore(spread_all, lookback)

    baseline_spread_vol = float(
        compute_spread(
            y=calibration_sample["y"],
            x=calibration_sample["x"],
            hedge_ratio=current_stats.hedge_ratio,
            intercept=current_stats.intercept,
        ).diff().std(ddof=0)
    )
    baseline_spread_vol = max(baseline_spread_vol, 1e-12)

    trade_filter_df = compute_trade_filter(
        spread=spread_all,
        trend_lookback=trend_lookback,
        max_trend_zscore_slope=max_trend_zscore_slope,
        vol_filter_lookback=vol_filter_lookback,
        max_recent_vol_multiplier=max_recent_vol_multiplier,
        baseline_spread_vol=baseline_spread_vol,
    )

    block_z = z_all.loc[available_block_dates]
    block_can_trade = trade_filter_df["can_trade"].loc[available_block_dates]

    block_pos = generate_positions(
        zscore=block_z,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        can_trade=block_can_trade,
        cooldown_days=cooldown_days,
        require_crossing=require_entry_cross,
    )

    full_df = window_df.copy()
    full_df["spread"] = spread_all
    full_df["zscore"] = z_all
    full_df["trend_score"] = trade_filter_df["trend_score"]
    full_df["trend_ok"] = trade_filter_df["trend_ok"]
    full_df["recent_spread_vol"] = trade_filter_df["recent_spread_vol"]
    full_df["regime_ok"] = trade_filter_df["regime_ok"]
    full_df["can_trade"] = trade_filter_df["can_trade"]

    full_df["position"] = 0.0
    full_df.loc[available_block_dates, "position"] = block_pos.astype(float)

    full_df["ret_y"] = full_df["y"].pct_change()
    full_df["ret_x"] = full_df["x"].pct_change()
    full_df["spread_return"] = full_df["ret_y"] - current_stats.hedge_ratio * full_df["ret_x"]
    full_df["position_lag"] = full_df["position"].shift(1).fillna(0.0)
    full_df["position_change"] = full_df["position"].diff().abs().fillna(full_df["position"].abs())

    total_cost_bps = commission_bps + slippage_bps
    full_df["trading_cost"] = (full_df["position_change"] * 2.0 * total_cost_bps) / 10000.0
    full_df["gross_return"] = full_df["position_lag"] * full_df["spread_return"]
    full_df["net_return_unscaled"] = full_df["gross_return"] - full_df["trading_cost"]

    block_results = full_df.loc[available_block_dates].copy()
    block_results = block_results.reindex(block_dates)
    for col in ["position", "position_lag", "position_change", "gross_return", "trading_cost", "net_return_unscaled"]:
        block_results[col] = block_results[col].fillna(0.0)

    block_results["pair_name"] = pair_name
    block_results["hedge_ratio"] = current_stats.hedge_ratio
    block_results["intercept"] = current_stats.intercept

    hist_sample = calibration_sample.copy()
    hist_sample["spread_hist"] = compute_spread(
        y=hist_sample["y"],
        x=hist_sample["x"],
        hedge_ratio=current_stats.hedge_ratio,
        intercept=current_stats.intercept,
    )
    hist_vol = _simulate_historical_vol(
        calibration_sample=hist_sample,
        hedge_ratio=current_stats.hedge_ratio,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        min_pair_vol=min_pair_vol,
        cooldown_days=cooldown_days,
        trend_lookback=trend_lookback,
        max_trend_zscore_slope=max_trend_zscore_slope,
        vol_filter_lookback=vol_filter_lookback,
        max_recent_vol_multiplier=max_recent_vol_multiplier,
        require_entry_cross=require_entry_cross,
    )

    trade_log = build_trade_log(
        position=block_results["position"],
        spread=block_results["spread"],
        zscore=block_results["zscore"],
        net_returns=block_results["net_return_unscaled"],
    )
    if not trade_log.empty:
        trade_log["pair_name"] = pair_name
        trade_log["block_start"] = block_start

    recalibration_log = pd.DataFrame([{
        "pair_name": pair_name,
        "date": block_start,
        "hedge_ratio": current_stats.hedge_ratio,
        "intercept": current_stats.intercept,
        "coint_pvalue": current_stats.coint_pvalue,
        "adf_pvalue": current_stats.adf_pvalue,
        "half_life": current_stats.half_life,
        "window_obs": len(calibration_sample),
        "validation_sharpe": validation_sharpe,
        "validation_total_return": validation_total_return,
        "baseline_spread_vol": baseline_spread_vol,
    }])

    return PairBlockOutput(
        pair_name=pair_name,
        block_results=block_results,
        trade_log=trade_log,
        recalibration_log=recalibration_log,
        hist_vol=hist_vol,
        current_coint_pvalue=current_stats.coint_pvalue,
        current_adf_pvalue=current_stats.adf_pvalue,
        current_half_life=current_stats.half_life,
        validation_sharpe=validation_sharpe,
        validation_total_return=validation_total_return,
    )


def build_dynamic_portfolio_from_candidates(
    prices: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    benchmark: pd.Series,
    test_start_date: pd.Timestamp,
    recalibration_window: int,
    recalibration_step: int,
    commission_bps: float,
    slippage_bps: float,
    initial_capital: float,
    target_portfolio_vol: float,
    max_gross_leverage: float,
    max_pair_weight: float,
    min_pair_vol: float,
    risk_free_rate: float,
    top_n_pairs: int,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    cooldown_days: int,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    require_entry_cross: bool,
    quality_weight_strength: float,
) -> PortfolioBacktestOutput:
    test_index = benchmark.loc[benchmark.index >= test_start_date].index.tolist()
    if not test_index:
        raise ValueError("Test period is empty.")

    pair_results_parts: dict[str, list[pd.DataFrame]] = {}
    pair_trade_logs_parts: dict[str, list[pd.DataFrame]] = {}
    pair_recal_logs_parts: dict[str, list[pd.DataFrame]] = {}
    selection_rows: list[dict] = []
    portfolio_blocks: list[pd.DataFrame] = []

    candidate_pairs_df = candidate_pairs_df.copy()

    for block_start_i in range(0, len(test_index), recalibration_step):
        block_dates = test_index[block_start_i:block_start_i + recalibration_step]
        if not block_dates:
            continue

        block_start = block_dates[0]
        candidate_outputs: list[PairBlockOutput] = []

        for _, row in candidate_pairs_df.iterrows():
            ticker_y = row["ticker_y"]
            ticker_x = row["ticker_x"]
            pair_name = row["pair_name"]

            try:
                pair_df = get_pair_frame(prices, ticker_y, ticker_x)
            except Exception:
                continue

            params = {
                "lookback": row["lookback"],
                "entry_z": row["entry_z"],
                "exit_z": row["exit_z"],
                "stop_z": row["stop_z"],
            }

            sim = _simulate_pair_block(
                pair_df=pair_df,
                pair_name=pair_name,
                params=params,
                validation_sharpe=float(row["validation_sharpe"]),
                validation_total_return=float(row["validation_total_return"]),
                block_dates=block_dates,
                recalibration_window=recalibration_window,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                min_pair_vol=min_pair_vol,
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
            )

            if sim is not None:
                candidate_outputs.append(sim)

        if candidate_outputs:
            candidate_outputs = sorted(
                candidate_outputs,
                key=lambda x: (
                    -x.validation_sharpe,
                    -x.validation_total_return,
                    x.current_coint_pvalue,
                    x.current_adf_pvalue,
                    x.current_half_life,
                ),
            )[:top_n_pairs]

        block_portfolio = pd.DataFrame(index=block_dates)
        block_portfolio["portfolio_return_pre_target"] = 0.0
        block_portfolio["active_pairs"] = ""

        if candidate_outputs:
            raw_weights = {}
            output_map = {out.pair_name: out for out in candidate_outputs}

            for out in candidate_outputs:
                inv_vol = 1.0 / max(out.hist_vol, min_pair_vol)
                multiplier = _pair_weight_multiplier(
                    validation_sharpe=out.validation_sharpe,
                    validation_total_return=out.validation_total_return,
                    current_coint_pvalue=out.current_coint_pvalue,
                    current_adf_pvalue=out.current_adf_pvalue,
                    current_half_life=out.current_half_life,
                    max_coint_pvalue=max_coint_pvalue,
                    max_adf_pvalue=max_adf_pvalue,
                    min_half_life=min_half_life,
                    max_half_life=max_half_life,
                    quality_weight_strength=quality_weight_strength,
                )
                raw_weights[out.pair_name] = inv_vol * multiplier

            base_weights = pd.Series(raw_weights, dtype=float)
            base_weights = base_weights / base_weights.sum()
            base_weights = cap_weights(base_weights, max_pair_weight=max_pair_weight)

            approx_portfolio_vol = float(sum(
                base_weights[pair_name] * max(output_map[pair_name].hist_vol, min_pair_vol)
                for pair_name in base_weights.index
            ))
            leverage = min(max_gross_leverage, target_portfolio_vol / max(approx_portfolio_vol, min_pair_vol))

            active_pair_names = []

            for out in candidate_outputs:
                pair_name = out.pair_name
                weight = float(base_weights.loc[pair_name])

                block_portfolio[f"weight_{pair_name}"] = weight
                block_portfolio[f"ret_{pair_name}"] = out.block_results["net_return_unscaled"].reindex(block_dates).fillna(0.0)
                block_portfolio["portfolio_return_pre_target"] += weight * block_portfolio[f"ret_{pair_name}"]
                active_pair_names.append(pair_name)

                pair_results_parts.setdefault(pair_name, []).append(out.block_results)
                if not out.trade_log.empty:
                    pair_trade_logs_parts.setdefault(pair_name, []).append(out.trade_log)
                pair_recal_logs_parts.setdefault(pair_name, []).append(out.recalibration_log)

                selection_rows.append({
                    "rebalance_date": block_start,
                    "pair_name": pair_name,
                    "weight": weight,
                    "validation_sharpe": out.validation_sharpe,
                    "validation_total_return": out.validation_total_return,
                    "current_coint_pvalue": out.current_coint_pvalue,
                    "current_adf_pvalue": out.current_adf_pvalue,
                    "current_half_life": out.current_half_life,
                    "hist_vol": out.hist_vol,
                })

            block_portfolio["active_pairs"] = ", ".join(active_pair_names)
            block_portfolio["leverage"] = leverage
            block_portfolio["portfolio_return"] = block_portfolio["portfolio_return_pre_target"] * leverage
        else:
            block_portfolio["leverage"] = 0.0
            block_portfolio["portfolio_return"] = 0.0
            block_portfolio["active_pairs"] = "CASH"

        portfolio_blocks.append(block_portfolio)

    if not portfolio_blocks:
        raise ValueError("No portfolio blocks were generated.")

    portfolio_results = pd.concat(portfolio_blocks, axis=0).sort_index()
    portfolio_results = portfolio_results[~portfolio_results.index.duplicated(keep="first")]
    portfolio_results["equity_curve"] = initial_capital * (1.0 + portfolio_results["portfolio_return"].fillna(0.0)).cumprod()

    benchmark_returns = benchmark.pct_change().reindex(portfolio_results.index).fillna(0.0)
    portfolio_results["benchmark_return"] = benchmark_returns
    portfolio_results["benchmark_equity_curve"] = initial_capital * (1.0 + benchmark_returns).cumprod()

    pair_results = {k: pd.concat(v, axis=0).sort_index() for k, v in pair_results_parts.items()}
    pair_trade_logs = {k: pd.concat(v, axis=0).reset_index(drop=True) for k, v in pair_trade_logs_parts.items()}
    pair_recalibration_logs = {k: pd.concat(v, axis=0).reset_index(drop=True) for k, v in pair_recal_logs_parts.items()}
    selection_log = pd.DataFrame(selection_rows)

    total_trades = float(sum(len(df) for df in pair_trade_logs.values()))
    total_turnover = 0.0
    total_pair_days = 0.0

    all_trade_logs = []
    for df in pair_trade_logs.values():
        if not df.empty:
            all_trade_logs.append(df)

    for df in pair_results.values():
        if "position_change" in df.columns:
            total_turnover += float(df["position_change"].fillna(0.0).sum())
        if "position_lag" in df.columns:
            total_pair_days += float((df["position_lag"].fillna(0.0) != 0).sum())

    benchmark_ann_return = annualised_return(benchmark_returns)
    active_returns = portfolio_results["portfolio_return"] - benchmark_returns
    pct_time_in_cash = float((portfolio_results["active_pairs"] == "CASH").mean())

    pair_hit_rate = 0.0
    avg_holding_period = 0.0
    if all_trade_logs:
        trades_df = pd.concat(all_trade_logs, axis=0, ignore_index=True)
        if "trade_return" in trades_df.columns and not trades_df.empty:
            pair_hit_rate = float((trades_df["trade_return"] > 0).mean())
        if "holding_days" in trades_df.columns and not trades_df.empty:
            avg_holding_period = float(trades_df["holding_days"].mean())

    metrics = {
        "Total Return": float(portfolio_results["equity_curve"].iloc[-1] / initial_capital - 1.0),
        "Annualised Return": annualised_return(portfolio_results["portfolio_return"]),
        "Annualised Volatility": annualised_volatility(portfolio_results["portfolio_return"]),
        "Sharpe Ratio": sharpe_ratio(portfolio_results["portfolio_return"], risk_free_rate=risk_free_rate),
        "Max Drawdown": max_drawdown(portfolio_results["equity_curve"]),
        "Calmar Ratio": calmar_ratio(portfolio_results["portfolio_return"], portfolio_results["equity_curve"]),
        "Portfolio Leverage Applied": float(portfolio_results["leverage"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()),
        "Number of Pairs": float(len(pair_results)),
        "Total Trades": total_trades,
        "Total Turnover": float(total_turnover),
        "Total Pair-Days in Market": float(total_pair_days),
        "Benchmark Return": float((1.0 + benchmark_returns).prod() - 1.0),
        "Excess Return vs Benchmark": float(annualised_return(portfolio_results["portfolio_return"]) - benchmark_ann_return),
        "Information Ratio": information_ratio(active_returns),
        "Percentage Time in Cash": pct_time_in_cash,
        "Pair Trade Hit Rate": pair_hit_rate,
        "Average Holding Period": avg_holding_period,
    }

    return PortfolioBacktestOutput(
        portfolio_results=portfolio_results,
        pair_results=pair_results,
        pair_trade_logs=pair_trade_logs,
        pair_recalibration_logs=pair_recalibration_logs,
        selection_log=selection_log,
        metrics=metrics,
    )