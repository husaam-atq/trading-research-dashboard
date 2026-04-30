from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .backtest import (
    calculate_dynamic_spread,
    combined_entry_filter,
    kalman_hedge_parameters,
    pair_leg_returns,
    rolling_hedge_parameters,
    run_pair_backtest,
    static_hedge_parameters,
    summarize_trades,
    training_volatility_limit,
    trade_metrics,
    volatility_entry_filter,
)
from .data import research_universes
from .metrics import benchmark_relative_metrics, performance_metrics
from .pairs import analyse_pair, half_life_z_window, screen_peer_groups
from .signals import generate_positions_with_reasons


def _group_mode_universes(mode: str) -> dict[str, list[str]]:
    return research_universes()[mode]


def _pair_name(row: pd.Series) -> str:
    return f"{row['ticker_y']}/{row['ticker_x']}"


def build_stability_diagnostics(
    prices: pd.DataFrame,
    mode: str,
    peer_groups: dict[str, list[str]],
    min_abs_correlation: float,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    min_threshold_crossings: int,
    min_training_trades: int,
    min_training_sharpe: float,
    min_training_max_drawdown: float,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    transaction_cost_bps: float,
    max_holding_period: int,
) -> pd.DataFrame:
    screening = screen_peer_groups(
        prices,
        peer_groups,
        min_abs_correlation=min_abs_correlation,
        max_coint_pvalue=max_coint_pvalue,
        max_adf_pvalue=max_adf_pvalue,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_threshold_crossings=min_threshold_crossings,
        zscore_window=0,
        entry_threshold=entry_threshold,
    )
    records: list[dict[str, object]] = []
    for row in screening.itertuples(index=False):
        z_window = half_life_z_window(float(row.half_life))
        try:
            result = run_pair_backtest(
                prices=prices,
                ticker_y=row.ticker_y,
                ticker_x=row.ticker_x,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_threshold=stop_threshold,
                zscore_window=z_window,
                transaction_cost_bps=transaction_cost_bps,
                max_holding_period=max_holding_period,
                hedge_mode="static",
                hedge_ratio=float(row.hedge_ratio),
                intercept=float(row.intercept),
                fit_window=len(prices),
            )
            training_trades = float(result.metrics["number_of_trades"])
            training_sharpe = float(result.metrics["sharpe_ratio"])
            training_max_drawdown = float(result.metrics["max_drawdown"])
            training_profit_factor = float(result.metrics["profit_factor"])
        except Exception:
            training_trades = 0.0
            training_sharpe = -np.inf
            training_max_drawdown = -1.0
            training_profit_factor = 0.0

        stable = (
            bool(row.selected_candidate)
            and training_trades >= min_training_trades
            and training_sharpe > min_training_sharpe
            and training_max_drawdown > min_training_max_drawdown
        )
        records.append(
            {
                **row._asdict(),
                "universe_mode": mode,
                "pair": f"{row.ticker_y}/{row.ticker_x}",
                "zscore_window": z_window,
                "training_trade_count": training_trades,
                "training_sharpe": training_sharpe,
                "training_max_drawdown": training_max_drawdown,
                "training_profit_factor": training_profit_factor,
                "stable_candidate": stable,
            }
        )

    return pd.DataFrame(records).sort_values(
        ["stable_candidate", "training_sharpe", "training_profit_factor", "coint_pvalue"],
        ascending=[False, False, False, True],
        na_position="last",
    )


def screen_all_modes(prices: pd.DataFrame, **kwargs: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for mode, groups in research_universes().items():
        frames.append(build_stability_diagnostics(prices, mode, groups, **kwargs))
    diagnostics = pd.concat(frames, ignore_index=True)
    screening = diagnostics.copy()
    screening["selected_candidate"] = screening["stable_candidate"]
    return screening, diagnostics


def _run_window_result(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    hedge_mode: str,
    entry: float,
    exit_: float,
    stop: float,
    max_hold: int,
    z_window: int,
    cost_bps: float,
    use_vol_filter: bool,
    vol_percentile: float,
    pair_drawdown_stop: float | None = -0.15,
    edge_threshold: float = 0.0,
    use_correlation_filter: bool = False,
    use_trend_filter: bool = False,
    cooldown_days: int = 0,
) -> object:
    return run_pair_backtest(
        prices,
        ticker_y,
        ticker_x,
        entry_threshold=entry,
        exit_threshold=exit_,
        stop_threshold=stop,
        zscore_window=z_window,
        transaction_cost_bps=cost_bps,
        max_holding_period=max_hold,
        hedge_mode=hedge_mode,
        fit_window=min(504, len(prices)),
        use_volatility_filter=use_vol_filter,
        volatility_percentile=vol_percentile,
        pair_drawdown_stop=pair_drawdown_stop,
        edge_threshold=edge_threshold,
        use_correlation_filter=use_correlation_filter,
        use_trend_filter=use_trend_filter,
        cooldown_days=cooldown_days,
    )


def _validate_candidate(
    full_window: pd.DataFrame,
    validation_index: pd.Index,
    row: pd.Series,
    hedge_mode: str,
    entry: float,
    exit_: float,
    stop: float,
    max_hold: int,
    cost_bps: float,
    use_vol_filter: bool,
    vol_percentile: float,
) -> dict[str, object]:
    result = _run_window_result(
        full_window,
        row["ticker_y"],
        row["ticker_x"],
        hedge_mode,
        entry,
        exit_,
        stop,
        max_hold,
        int(row["zscore_window"]),
        cost_bps,
        use_vol_filter,
        vol_percentile,
    )
    validation_daily = result.daily.loc[result.daily.index.intersection(validation_index)]
    validation_trades = result.trades[
        pd.to_datetime(result.trades["exit_date"]).isin(validation_daily.index)
    ] if not result.trades.empty else pd.DataFrame()
    metrics = performance_metrics(validation_daily["strategy_return"])
    profit_factor = float(result.metrics.get("profit_factor", 0.0))
    if not validation_trades.empty:
        winners = validation_trades.loc[validation_trades["net_return"] > 0, "net_return"].sum()
        losers = validation_trades.loc[validation_trades["net_return"] < 0, "net_return"].sum()
        profit_factor = float(winners / abs(losers)) if losers < 0 else float("inf") if winners > 0 else 0.0
    return {
        "pair": row["pair"],
        "universe_mode": row["universe_mode"],
        "peer_group": row["peer_group"],
        "ticker_y": row["ticker_y"],
        "ticker_x": row["ticker_x"],
        "hedge_mode": hedge_mode,
        "entry_threshold": entry,
        "exit_threshold": exit_,
        "stop_threshold": stop,
        "max_holding_period": max_hold,
        "zscore_window": int(row["zscore_window"]),
        "volatility_percentile": vol_percentile,
        "validation_total_return": float(metrics["total_return"]),
        "validation_sharpe": float(metrics["sharpe_ratio"]),
        "validation_profit_factor": profit_factor,
        "validation_max_drawdown": float(metrics["max_drawdown"]),
        "validation_volatility": float(metrics["annualised_volatility"]),
    }


def _diagnostics_from_pool(
    train: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in candidate_pool.iterrows():
        ticker_y = str(row["ticker_y"])
        ticker_x = str(row["ticker_x"])
        if ticker_y not in train.columns or ticker_x not in train.columns:
            continue
        pair_prices = train[[ticker_y, ticker_x]].dropna()
        if len(pair_prices) < 252:
            continue
        try:
            stats = analyse_pair(pair_prices[ticker_y], pair_prices[ticker_x], zscore_window=0, entry_threshold=1.5)
            z_window = half_life_z_window(float(stats["half_life"]))
            result = run_pair_backtest(
                train,
                ticker_y,
                ticker_x,
                entry_threshold=1.5,
                exit_threshold=0.5,
                stop_threshold=3.0,
                zscore_window=z_window,
                transaction_cost_bps=cost_bps,
                max_holding_period=20,
                hedge_mode="static",
                fit_window=len(pair_prices),
            )
            stable = (
                abs(float(stats["correlation"])) >= 0.85
                and float(stats["coint_pvalue"]) <= 0.05
                and float(stats["adf_pvalue"]) <= 0.05
                and 3.0 <= float(stats["half_life"]) <= 45.0
                and float(stats["threshold_crossings"]) >= 8
                and float(result.metrics["number_of_trades"]) >= 5
                and float(result.metrics["sharpe_ratio"]) > 0.25
                and float(result.metrics["max_drawdown"]) > -0.20
            )
            records.append(
                {
                    **row.to_dict(),
                    **stats,
                    "pair": f"{ticker_y}/{ticker_x}",
                    "zscore_window": z_window,
                    "training_trade_count": float(result.metrics["number_of_trades"]),
                    "training_sharpe": float(result.metrics["sharpe_ratio"]),
                    "training_max_drawdown": float(result.metrics["max_drawdown"]),
                    "training_profit_factor": float(result.metrics["profit_factor"]),
                    "stable_candidate": stable,
                    "selected_candidate": stable,
                }
            )
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["stable_candidate", "training_sharpe", "training_profit_factor", "coint_pvalue"],
        ascending=[False, False, False, True],
        na_position="last",
    )


def _hedge_params_for_mode(log_pair: pd.DataFrame, hedge_mode: str) -> pd.DataFrame:
    if hedge_mode == "static":
        return static_hedge_parameters(log_pair, fit_window=min(504, len(log_pair)))
    if hedge_mode == "rolling":
        return rolling_hedge_parameters(log_pair, window=min(252, max(30, len(log_pair) // 2)))
    if hedge_mode == "kalman":
        return kalman_hedge_parameters(log_pair)
    raise ValueError(f"Unsupported hedge mode: {hedge_mode}")


def _validation_grid_for_candidate(
    full_window: pd.DataFrame,
    validation_index: pd.Index,
    row: pd.Series,
    hedge_mode: str,
    entry_grid: list[float],
    exit_grid: list[float],
    stop_grid: list[float],
    max_hold_grid: list[int],
    cost_bps: float,
    use_vol_filter: bool,
    vol_percentiles: tuple[float, ...],
    edge_threshold: float = 0.0,
    use_correlation_filter: bool = False,
    use_trend_filter: bool = False,
    cooldown_days: int = 0,
) -> list[dict[str, object]]:
    ticker_y = str(row["ticker_y"])
    ticker_x = str(row["ticker_x"])
    pair = full_window[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    log_pair = np.log(pair)
    hedge_params = _hedge_params_for_mode(log_pair, hedge_mode)
    spread = calculate_dynamic_spread(log_pair, hedge_params)
    z_window = int(row["zscore_window"])
    from .signals import rolling_zscore

    zscore = rolling_zscore(spread, z_window)
    records: list[dict[str, object]] = []
    train_spread = spread.loc[spread.index.difference(validation_index)]
    candidate_settings: list[tuple[float, float, float, float, int]] = []
    baseline_stop = 3.0 if 3.0 in stop_grid else stop_grid[0]
    baseline_hold = 20 if 20 in max_hold_grid else max_hold_grid[0]
    baseline_vol = vol_percentiles[-1]
    for entry, exit_ in product(entry_grid, exit_grid):
        if exit_ < entry and baseline_stop > entry:
            candidate_settings.append((baseline_vol, entry, exit_, baseline_stop, baseline_hold))

    first_pass: list[dict[str, object]] = []
    for vol_pct, entry, exit_, stop, max_hold in candidate_settings:
        vol_limit = training_volatility_limit(train_spread, z_window, vol_pct)
        can_enter = combined_entry_filter(
            spread,
            zscore,
            log_pair,
            hedge_params["hedge_ratio"],
            z_window,
            vol_limit,
            cost_bps,
            edge_threshold=edge_threshold,
            use_volatility_filter=use_vol_filter,
            use_correlation_filter=use_correlation_filter,
            use_trend_filter=use_trend_filter,
        )
        position, exit_reasons = generate_positions_with_reasons(
            zscore,
            entry,
            exit_,
            stop,
            can_enter=can_enter,
            max_holding_period=max_hold,
            cooldown_days=cooldown_days,
        )
        daily = pair_leg_returns(pair, hedge_params["hedge_ratio"], position, cost_bps)
        daily["zscore"] = zscore
        daily["signal_zscore"] = daily["zscore"].shift(1)
        daily["target_exit_reason"] = exit_reasons
        daily["realized_exit_reason"] = daily["target_exit_reason"].shift(1).fillna("")
        validation_daily = daily.loc[daily.index.intersection(validation_index)]
        trades = summarize_trades(daily, str(row["pair"]))
        validation_trades = trades[pd.to_datetime(trades["exit_date"]).isin(validation_daily.index)] if not trades.empty else pd.DataFrame()
        metrics = performance_metrics(validation_daily["strategy_return"])
        trade_stats = trade_metrics(validation_daily, validation_trades)
        turnover = float(validation_daily["position_change"].sum())
        turnover_penalty = turnover / max(float(trade_stats["number_of_trades"]), 1.0)
        instability_penalty = max(0.0, abs(float(row["half_life"]) - 15.0) / 45.0) + float(row.get("spread_std", 0.0))
        robust_score = (
            float(metrics["sharpe_ratio"])
            + 0.25 * min(float(trade_stats["profit_factor"]), 5.0)
            - 0.50 * abs(float(metrics["max_drawdown"]))
            - 0.10 * turnover_penalty
            - 0.10 * instability_penalty
        )
        first_pass.append(
            {
                "pair": row["pair"],
                "universe_mode": row["universe_mode"],
                "peer_group": row["peer_group"],
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "hedge_mode": hedge_mode,
                "entry_threshold": entry,
                "exit_threshold": exit_,
                "stop_threshold": stop,
                "max_holding_period": max_hold,
                "zscore_window": z_window,
                "volatility_percentile": vol_pct,
                "validation_total_return": float(metrics["total_return"]),
                "validation_sharpe": float(metrics["sharpe_ratio"]),
                "validation_profit_factor": float(trade_stats["profit_factor"]),
                "validation_max_drawdown": float(metrics["max_drawdown"]),
                "validation_volatility": float(metrics["annualised_volatility"]),
                "validation_number_of_trades": float(trade_stats["number_of_trades"]),
                "validation_turnover": turnover,
                "validation_average_holding_period": float(trade_stats["average_holding_period"]),
                "validation_win_rate": float(trade_stats["win_rate"]),
                "spread_half_life": float(row["half_life"]),
                "spread_volatility": float(row.get("spread_std", np.nan)),
                "correlation": float(row.get("correlation", np.nan)),
                "coint_pvalue": float(row.get("coint_pvalue", np.nan)),
                "adf_pvalue": float(row.get("adf_pvalue", np.nan)),
                "edge_threshold": edge_threshold,
                "use_volatility_filter": use_vol_filter,
                "use_correlation_filter": use_correlation_filter,
                "use_trend_filter": use_trend_filter,
                "cooldown_days": cooldown_days,
                "turnover_penalty": turnover_penalty,
                "instability_penalty": instability_penalty,
                "robust_score": robust_score,
            }
        )
    if first_pass:
        best = sorted(
            first_pass,
            key=lambda item: (float(item["validation_sharpe"]), float(item["validation_profit_factor"]), float(item["validation_total_return"])),
            reverse=True,
        )[0]
        for vol_pct, stop, max_hold in product(vol_percentiles, stop_grid, max_hold_grid):
            entry = float(best["entry_threshold"])
            exit_ = float(best["exit_threshold"])
            if stop <= entry:
                continue
            candidate_settings.append((vol_pct, entry, exit_, stop, max_hold))

    seen: set[tuple[float, float, float, float, int]] = set()
    for vol_pct, entry, exit_, stop, max_hold in candidate_settings:
        key = (vol_pct, entry, exit_, stop, max_hold)
        if key in seen:
            continue
        seen.add(key)
        vol_limit = training_volatility_limit(train_spread, z_window, vol_pct)
        can_enter = combined_entry_filter(
            spread,
            zscore,
            log_pair,
            hedge_params["hedge_ratio"],
            z_window,
            vol_limit,
            cost_bps,
            edge_threshold=edge_threshold,
            use_volatility_filter=use_vol_filter,
            use_correlation_filter=use_correlation_filter,
            use_trend_filter=use_trend_filter,
        )
        position, exit_reasons = generate_positions_with_reasons(
            zscore,
            entry,
            exit_,
            stop,
            can_enter=can_enter,
            max_holding_period=max_hold,
            cooldown_days=cooldown_days,
        )
        daily = pair_leg_returns(pair, hedge_params["hedge_ratio"], position, cost_bps)
        daily["zscore"] = zscore
        daily["signal_zscore"] = daily["zscore"].shift(1)
        daily["target_exit_reason"] = exit_reasons
        daily["realized_exit_reason"] = daily["target_exit_reason"].shift(1).fillna("")
        validation_daily = daily.loc[daily.index.intersection(validation_index)]
        trades = summarize_trades(daily, str(row["pair"]))
        validation_trades = trades[pd.to_datetime(trades["exit_date"]).isin(validation_daily.index)] if not trades.empty else pd.DataFrame()
        metrics = performance_metrics(validation_daily["strategy_return"])
        trade_stats = trade_metrics(validation_daily, validation_trades)
        turnover = float(validation_daily["position_change"].sum())
        turnover_penalty = turnover / max(float(trade_stats["number_of_trades"]), 1.0)
        instability_penalty = max(0.0, abs(float(row["half_life"]) - 15.0) / 45.0) + float(row.get("spread_std", 0.0))
        robust_score = (
            float(metrics["sharpe_ratio"])
            + 0.25 * min(float(trade_stats["profit_factor"]), 5.0)
            - 0.50 * abs(float(metrics["max_drawdown"]))
            - 0.10 * turnover_penalty
            - 0.10 * instability_penalty
        )
        records.append(
            {
                "pair": row["pair"],
                "universe_mode": row["universe_mode"],
                "peer_group": row["peer_group"],
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "hedge_mode": hedge_mode,
                "entry_threshold": entry,
                "exit_threshold": exit_,
                "stop_threshold": stop,
                "max_holding_period": max_hold,
                "zscore_window": z_window,
                "volatility_percentile": vol_pct,
                "validation_total_return": float(metrics["total_return"]),
                "validation_sharpe": float(metrics["sharpe_ratio"]),
                "validation_profit_factor": float(trade_stats["profit_factor"]),
                "validation_max_drawdown": float(metrics["max_drawdown"]),
                "validation_volatility": float(metrics["annualised_volatility"]),
                "validation_number_of_trades": float(trade_stats["number_of_trades"]),
                "validation_turnover": turnover,
                "validation_average_holding_period": float(trade_stats["average_holding_period"]),
                "validation_win_rate": float(trade_stats["win_rate"]),
                "spread_half_life": float(row["half_life"]),
                "spread_volatility": float(row.get("spread_std", np.nan)),
                "correlation": float(row.get("correlation", np.nan)),
                "coint_pvalue": float(row.get("coint_pvalue", np.nan)),
                "adf_pvalue": float(row.get("adf_pvalue", np.nan)),
                "edge_threshold": edge_threshold,
                "use_volatility_filter": use_vol_filter,
                "use_correlation_filter": use_correlation_filter,
                "use_trend_filter": use_trend_filter,
                "cooldown_days": cooldown_days,
                "turnover_penalty": turnover_penalty,
                "instability_penalty": instability_penalty,
                "robust_score": robust_score,
            }
        )
    return records


def _portfolio_weights(selection: pd.DataFrame, method: str) -> pd.Series:
    if selection.empty:
        return pd.Series(dtype=float)
    index = selection["pair"].astype(str)
    if method == "equal_weight":
        weights = pd.Series(1.0, index=index)
    elif method == "inverse_volatility":
        weights = 1.0 / selection["validation_volatility"].replace(0.0, np.nan).fillna(selection["validation_volatility"].median())
        weights.index = index
    elif method == "sharpe_weighted":
        weights = selection["validation_sharpe"].clip(lower=0.0)
        weights.index = index
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=index)
    elif method == "robust_score_weighted":
        weights = selection["robust_score"].clip(lower=0.0)
        weights.index = index
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=index)
    elif method == "risk_capped_inverse_volatility":
        weights = 1.0 / selection["validation_volatility"].replace(0.0, np.nan).fillna(selection["validation_volatility"].median())
        weights.index = index
        max_weight = 0.60 if len(weights) < 3 else 0.40
        weights = weights / weights.sum()
        weights = weights.clip(upper=max_weight)
    else:
        weights = selection["validation_sharpe"].clip(lower=0.0)
        weights.index = index
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=index)
        weights = weights / weights.sum()
        weights = weights.clip(upper=0.40)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(1.0 / len(index), index=index)
    return weights / total


def _risk_scaled_returns(returns: pd.Series, target_vol: float = 0.10, max_leverage: float = 1.5, window: int = 63) -> pd.Series:
    trailing_vol = returns.rolling(window).std(ddof=0) * np.sqrt(252)
    leverage = (target_vol / trailing_vol.replace(0.0, np.nan)).shift(1).clip(upper=max_leverage).fillna(1.0)
    return returns * leverage


def nested_pair_selection_portfolio(
    prices: pd.DataFrame,
    train_window: int,
    validation_window: int,
    test_window: int,
    top_n_pairs: int,
    cost_bps: float,
    entry_grid: list[float],
    exit_grid: list[float],
    stop_grid: list[float],
    max_hold_grid: list[int],
    hedge_modes: list[str],
    max_validation_candidates: int = 10,
    use_vol_filter: bool = True,
    vol_percentiles: tuple[float, ...] = (0.80, 0.90),
    candidate_pool: pd.DataFrame | None = None,
    step_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_records: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    portfolio_returns: list[pd.DataFrame] = []
    test_pair_returns: list[pd.DataFrame] = []
    trade_logs: list[pd.DataFrame] = []
    segment = 0
    start = 0
    if step_size is None:
        step_size = test_window

    while start + train_window + validation_window + 5 < len(prices):
        train = prices.iloc[start : start + train_window]
        validation = prices.iloc[start + train_window : start + train_window + validation_window]
        test = prices.iloc[start + train_window + validation_window : start + train_window + validation_window + test_window]
        if len(test) < 5:
            break

        if candidate_pool is not None:
            diagnostics = _diagnostics_from_pool(train, candidate_pool, cost_bps)
        else:
            frames: list[pd.DataFrame] = []
            for mode, groups in research_universes().items():
                screened = screen_peer_groups(
                    train,
                    groups,
                    min_abs_correlation=0.85,
                    max_coint_pvalue=0.05,
                    max_adf_pvalue=0.05,
                    min_half_life=3.0,
                    max_half_life=45.0,
                    min_threshold_crossings=8,
                    zscore_window=0,
                    entry_threshold=1.5,
                )
                screened["universe_mode"] = mode
                screened["pair"] = screened["ticker_y"] + "/" + screened["ticker_x"]
                screened["zscore_window"] = screened["half_life"].apply(half_life_z_window)
                frames.append(screened)
            diagnostics = pd.concat(frames, ignore_index=True).sort_values(
                ["selected_candidate", "coint_pvalue", "half_life", "abs_correlation"],
                ascending=[False, True, True, False],
                na_position="last",
            )
        if diagnostics.empty:
            start += test_window
            segment += 1
            continue
        candidates = diagnostics[diagnostics["selected_candidate"]].head(max_validation_candidates)
        if candidates.empty:
            candidates = diagnostics.head(max(3, min(max_validation_candidates, len(diagnostics))))

        validation_records: list[dict[str, object]] = []
        full_validation_window = pd.concat([train, validation]).dropna()
        for _, candidate in candidates.iterrows():
            mode_baselines: list[dict[str, object]] = []
            for hedge_mode in hedge_modes:
                mode_baselines.extend(
                    _validation_grid_for_candidate(
                        full_validation_window,
                        validation.index,
                        candidate,
                        hedge_mode,
                        [1.5],
                        [0.5],
                        [3.0],
                        [20],
                        cost_bps,
                        use_vol_filter,
                        (0.90,),
                        edge_threshold=0.0,
                        use_correlation_filter=False,
                        use_trend_filter=False,
                        cooldown_days=0,
                    )
                )
            if not mode_baselines:
                continue
            best_mode = sorted(
                mode_baselines,
                key=lambda item: (float(item["validation_sharpe"]), float(item["validation_profit_factor"]), float(item["validation_total_return"])),
                reverse=True,
            )[0]["hedge_mode"]
            validation_records.extend(
                _validation_grid_for_candidate(
                    full_validation_window,
                    validation.index,
                    candidate,
                    str(best_mode),
                    entry_grid,
                    exit_grid,
                    stop_grid,
                    max_hold_grid,
                    cost_bps,
                    use_vol_filter,
                    vol_percentiles,
                    edge_threshold=0.0,
                    use_correlation_filter=False,
                    use_trend_filter=False,
                    cooldown_days=0,
                )
            )

        validation_table = pd.DataFrame(validation_records)
        if validation_table.empty:
            start += test_window
            segment += 1
            continue
        validation_table["segment"] = segment
        validation_table["train_start"] = train.index[0]
        validation_table["train_end"] = train.index[-1]
        validation_table["validation_start"] = validation.index[0]
        validation_table["validation_end"] = validation.index[-1]
        validation_table["test_start"] = test.index[0]
        validation_table["test_end"] = test.index[-1]
        score_frames.append(validation_table)
        eligible = validation_table[
            (validation_table["validation_sharpe"] > 0.25)
            & (validation_table["validation_profit_factor"] > 1.10)
            & (validation_table["validation_max_drawdown"] > -0.10)
            & (validation_table["validation_number_of_trades"] >= 4)
            & (validation_table["validation_average_holding_period"].between(2, 30))
            & (validation_table["spread_half_life"].between(3, 45))
            & (validation_table["correlation"] >= 0.85)
            & (validation_table["coint_pvalue"] <= 0.05)
            & (validation_table["adf_pvalue"] <= 0.05)
        ].sort_values(["robust_score", "validation_sharpe", "validation_profit_factor"], ascending=False)
        breadth_limited = len(eligible.drop_duplicates("pair")) < 2
        selected = eligible.drop_duplicates("pair").head(top_n_pairs if not breadth_limited else 1)
        if selected.empty:
            breadth_limited = True
            selected = validation_table.sort_values(["robust_score", "validation_sharpe", "validation_profit_factor"], ascending=False).drop_duplicates("pair").head(1)

        segment_returns: list[pd.Series] = []
        full_test_window = pd.concat([train, validation, test]).dropna()
        for _, selected_row in selected.iterrows():
            result = _run_window_result(
                full_test_window,
                str(selected_row["ticker_y"]),
                str(selected_row["ticker_x"]),
                str(selected_row["hedge_mode"]),
                float(selected_row["entry_threshold"]),
                float(selected_row["exit_threshold"]),
                float(selected_row["stop_threshold"]),
                int(selected_row["max_holding_period"]),
                int(selected_row["zscore_window"]),
                cost_bps,
                use_vol_filter,
                float(selected_row["volatility_percentile"]),
                edge_threshold=float(selected_row.get("edge_threshold", 0.0)),
                use_correlation_filter=bool(selected_row.get("use_correlation_filter", False)),
                use_trend_filter=bool(selected_row.get("use_trend_filter", False)),
                cooldown_days=int(selected_row.get("cooldown_days", 0)),
            )
            test_daily = result.daily.loc[result.daily.index.intersection(test.index)]
            pair_label = str(selected_row["pair"])
            segment_returns.append(test_daily["strategy_return"].rename(pair_label))
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["peer_group"] = selected_row["peer_group"]
                trades["universe_mode"] = selected_row["universe_mode"]
                trades["hedge_mode"] = selected_row["hedge_mode"]
                trades["entry_threshold"] = selected_row["entry_threshold"]
                trades["exit_threshold"] = selected_row["exit_threshold"]
                trades["stop_threshold"] = selected_row["stop_threshold"]
                trades["max_holding_period"] = selected_row["max_holding_period"]
                trades = trades[pd.to_datetime(trades["exit_date"]).isin(test.index)]
                trade_logs.append(trades)
            selection_records.append(
                {
                    "segment": segment,
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "validation_start": validation.index[0],
                    "validation_end": validation.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "breadth_limited": breadth_limited,
                    **selected_row.to_dict(),
                }
            )

        if segment_returns:
            pair_return_frame = pd.concat(segment_returns, axis=1).fillna(0.0)
            test_pair_returns.append(pair_return_frame.add_prefix(f"segment_{segment}_"))
            portfolio_frame = pd.DataFrame(index=pair_return_frame.index)
            for method in ["equal_weight", "inverse_volatility", "sharpe_weighted", "robust_score_weighted", "risk_capped_inverse_volatility", "risk_capped"]:
                weights = _portfolio_weights(selected, method)
                weighted = pair_return_frame.mul(weights.reindex(pair_return_frame.columns).fillna(0.0), axis=1).sum(axis=1)
                if method == "risk_capped":
                    weighted = _risk_scaled_returns(weighted)
                portfolio_frame[method] = weighted
            portfolio_returns.append(portfolio_frame)

        start += step_size
        segment += 1

    daily = pd.concat(portfolio_returns).sort_index() if portfolio_returns else pd.DataFrame()
    nested_pair_returns = pd.concat(test_pair_returns, axis=1).fillna(0.0) if test_pair_returns else pd.DataFrame()
    selection = pd.DataFrame(selection_records)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()
    comparison_records = []
    for column in daily.columns:
        metrics = performance_metrics(daily[column])
        comparison_records.append({"portfolio_method": column, **metrics})
    comparison = pd.DataFrame(comparison_records)
    results = comparison.rename(columns={"portfolio_method": "strategy"})
    return selection, results, daily, nested_pair_returns, trade_log, scores


def cost_sensitivity_from_selection(
    prices: pd.DataFrame,
    selection: pd.DataFrame,
    costs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost in costs:
        segment_returns: list[pd.DataFrame] = []
        total_trades = 0.0
        for segment, segment_selection in selection.groupby("segment"):
            test_start = pd.to_datetime(segment_selection["test_start"].iloc[0])
            test_end = pd.to_datetime(segment_selection["test_end"].iloc[0])
            test_index = prices.loc[test_start:test_end].index
            pair_returns: list[pd.Series] = []
            for _, selected_row in segment_selection.iterrows():
                window = prices.loc[:test_end]
                result = _run_window_result(
                    window,
                    str(selected_row["ticker_y"]),
                    str(selected_row["ticker_x"]),
                    str(selected_row["hedge_mode"]),
                    float(selected_row["entry_threshold"]),
                    float(selected_row["exit_threshold"]),
                    float(selected_row["stop_threshold"]),
                    int(selected_row["max_holding_period"]),
                    int(selected_row["zscore_window"]),
                    cost,
                    True,
                    float(selected_row["volatility_percentile"]),
                    edge_threshold=float(selected_row.get("edge_threshold", 0.0)),
                    use_correlation_filter=bool(selected_row.get("use_correlation_filter", False)),
                    use_trend_filter=bool(selected_row.get("use_trend_filter", False)),
                    cooldown_days=int(selected_row.get("cooldown_days", 0)),
                )
                test_trades = result.trades[pd.to_datetime(result.trades["exit_date"]).isin(test_index)] if not result.trades.empty else pd.DataFrame()
                total_trades += float(len(test_trades))
                pair_returns.append(result.daily.loc[result.daily.index.intersection(test_index), "strategy_return"].rename(str(selected_row["pair"])))
            if pair_returns:
                frame = pd.concat(pair_returns, axis=1).fillna(0.0)
                segment_frame = pd.DataFrame(index=frame.index)
                for method in ["equal_weight", "inverse_volatility", "sharpe_weighted", "robust_score_weighted", "risk_capped_inverse_volatility", "risk_capped"]:
                    weights = _portfolio_weights(segment_selection, method)
                    weighted = frame.mul(weights.reindex(frame.columns).fillna(0.0), axis=1).sum(axis=1)
                    if method == "risk_capped":
                        weighted = _risk_scaled_returns(weighted)
                    segment_frame[method] = weighted
                segment_returns.append(segment_frame)
        daily = pd.concat(segment_returns).sort_index() if segment_returns else pd.DataFrame()
        for column in daily.columns:
            returns = daily[column]
            winners = returns[returns > 0].sum()
            losers = returns[returns < 0].sum()
            profit_factor = float(winners / abs(losers)) if losers < 0 else float("inf") if winners > 0 else 0.0
            rows.append(
                {
                    "transaction_cost_bps": cost,
                    "strategy": column,
                    **performance_metrics(returns),
                    "profit_factor": profit_factor,
                    "number_of_trades": total_trades,
                }
            )
    return pd.DataFrame(rows)


def benchmark_metrics_table(returns: pd.DataFrame, spy_returns: pd.Series | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in returns.columns:
        rows.append({"strategy": column, **benchmark_relative_metrics(returns[column], spy_returns)})
    return pd.DataFrame(rows)
