from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_pair_backtest
from .config import ExperimentConfig
from .discovery import discover_training_candidates
from .metrics import performance_metrics
from .portfolio import (
    concentration_metrics,
    lagged_volatility_scale,
    select_diversified_pairs,
    validation_weights,
)


@dataclass
class WalkForwardResult:
    boundaries: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    validation_scores: pd.DataFrame
    selection_decisions: pd.DataFrame
    selected_pairs: pd.DataFrame
    validation_daily_returns: pd.DataFrame
    pair_daily_returns: pd.DataFrame
    portfolio_daily_returns: pd.DataFrame
    trade_log: pd.DataFrame
    segment_metrics: pd.DataFrame
    concentration: pd.DataFrame


def walk_forward_boundaries(
    index: pd.DatetimeIndex,
    config: ExperimentConfig,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
) -> pd.DataFrame:
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Walk-forward index must be sorted and unique.")
    first_test = config.train_window + config.validation_window
    if evaluation_start is not None:
        first_test = int(index.searchsorted(pd.Timestamp(evaluation_start), side="left"))
    if first_test < config.train_window + config.validation_window:
        raise ValueError("Not enough pre-evaluation history for train and validation windows.")
    last_allowed = pd.Timestamp(evaluation_end) if evaluation_end is not None else index[-1]

    records: list[dict[str, object]] = []
    segment = 0
    test_start = first_test
    while test_start < len(index) and index[test_start] <= last_allowed:
        test_stop = min(test_start + config.test_window, len(index))
        test_positions = np.arange(test_start, test_stop)
        test_positions = test_positions[index[test_positions] <= last_allowed]
        if len(test_positions) < 20:
            break
        validation_start = test_start - config.validation_window
        train_start = validation_start - config.train_window
        if train_start < 0:
            raise ValueError("Walk-forward boundary would use unavailable pre-training observations.")
        records.append(
            {
                "segment": segment,
                "train_start": index[train_start],
                "train_end": index[validation_start - 1],
                "validation_start": index[validation_start],
                "validation_end": index[test_start - 1],
                "test_start": index[test_positions[0]],
                "test_end": index[test_positions[-1]],
                "train_observations": config.train_window,
                "validation_observations": config.validation_window,
                "test_observations": len(test_positions),
            }
        )
        segment += 1
        test_start += config.step_size
    return pd.DataFrame(records)


def assert_boundary_integrity(boundaries: pd.DataFrame) -> None:
    if boundaries.empty:
        raise ValueError("No walk-forward boundaries were generated.")
    for row in boundaries.itertuples(index=False):
        if not (row.train_start <= row.train_end < row.validation_start <= row.validation_end < row.test_start <= row.test_end):
            raise AssertionError(f"Invalid chronological boundary in segment {row.segment}.")
    ordered = boundaries.sort_values("segment")
    previous_end: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        if previous_end is not None and row.test_start <= previous_end:
            raise AssertionError("Test windows overlap.")
        previous_end = row.test_end


def _profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    winners = float(trades.loc[trades["net_return"] > 0, "net_return"].sum())
    losers = float(trades.loc[trades["net_return"] < 0, "net_return"].sum())
    return winners / abs(losers) if losers < 0 else 5.0 if winners > 0 else 0.0


def _hedge_spec(label: str) -> tuple[str, int]:
    if label.startswith("rolling_"):
        return "rolling", int(label.split("_", maxsplit=1)[1])
    return label, 252


def _run_period(
    combined: pd.DataFrame,
    active_index: pd.DatetimeIndex,
    candidate: pd.Series,
    settings: dict[str, object],
    cost_bps: float,
    fit_window: int,
) -> BacktestResult:
    hedge_mode, hedge_window = _hedge_spec(str(settings["hedge_mode"]))
    return run_pair_backtest(
        combined,
        str(candidate["ticker_y"]),
        str(candidate["ticker_x"]),
        entry_threshold=float(settings["entry_threshold"]),
        exit_threshold=float(settings["exit_threshold"]),
        stop_threshold=float(settings["stop_threshold"]),
        zscore_window=int(settings["zscore_window"]),
        transaction_cost_bps=cost_bps,
        max_holding_period=int(settings["max_holding_period"]),
        hedge_mode=hedge_mode,
        hedge_training_window=hedge_window,
        volatility_percentile=float(settings.get("volatility_percentile", 0.90)),
        use_volatility_filter=bool(settings.get("use_volatility_filter", False)),
        cooldown_days=int(settings.get("cooldown_days", 0)),
        fit_window=fit_window,
        pair_drawdown_stop=-0.15,
        trade_start=active_index[0],
    )


def _score_validation(
    result: BacktestResult,
    active_index: pd.DatetimeIndex,
    candidate: pd.Series,
    settings: dict[str, object],
    stage: str,
) -> tuple[dict[str, object], pd.Series]:
    daily = result.daily.reindex(active_index).fillna(0.0)
    trades = result.trades[
        (pd.to_datetime(result.trades["entry_date"]) >= active_index[0])
        & (pd.to_datetime(result.trades["exit_date"]) <= active_index[-1])
    ] if not result.trades.empty else pd.DataFrame()
    metrics = performance_metrics(daily["strategy_return"])
    trade_count = len(trades)
    profit_factor = _profit_factor(trades)
    turnover = float(daily["position_change"].sum())
    reliability = min(1.0, np.sqrt(trade_count / 6.0))
    complexity_penalty = 0.03 if str(settings["hedge_mode"]) == "kalman" else 0.01 if str(settings["hedge_mode"]).startswith("rolling") else 0.0
    robust_score = (
        float(metrics["sharpe_ratio"]) * reliability
        + 0.20 * np.log1p(min(profit_factor, 5.0))
        - 0.50 * abs(float(metrics["max_drawdown"]))
        - 0.025 * turnover
        + 0.20 * float(candidate["stability_score"])
        - 0.10 * float(candidate["structural_instability_penalty"])
        - complexity_penalty
    )
    record = {
        "pair": candidate["pair"],
        "universe_mode": candidate["universe_mode"],
        "peer_group": candidate["peer_group"],
        "ticker_y": candidate["ticker_y"],
        "ticker_x": candidate["ticker_x"],
        "stage": stage,
        **settings,
        "stability_score": float(candidate["stability_score"]),
        "training_score": float(candidate["training_score"]),
        "training_candidate_pass": bool(candidate["selected_candidate"]),
        "validation_total_return": float(metrics["total_return"]),
        "validation_sharpe": float(metrics["sharpe_ratio"]),
        "validation_max_drawdown": float(metrics["max_drawdown"]),
        "validation_volatility": float(metrics["annualised_volatility"]),
        "validation_profit_factor": profit_factor,
        "validation_trades": float(trade_count),
        "validation_turnover": turnover,
        "validation_average_holding_period": float(trades["holding_period"].mean()) if not trades.empty else 0.0,
        "validation_win_rate": float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0,
        "robust_score": robust_score,
    }
    return record, daily["strategy_return"].rename(str(candidate["pair"]))


def _validate_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    candidate: pd.Series,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object], pd.Series]:
    pair_columns = [str(candidate["ticker_y"]), str(candidate["ticker_x"])]
    combined = pd.concat([train[pair_columns], validation[pair_columns]]).dropna()
    validation_index = combined.index.intersection(validation.index)
    fit_window = int((combined.index < validation_index[0]).sum())
    base_z = int(candidate["zscore_window"])
    all_records: list[dict[str, object]] = []

    mode_records: list[tuple[dict[str, object], pd.Series]] = []
    for hedge_mode in config.validation.hedge_modes:
        settings = {
            "hedge_mode": hedge_mode,
            "entry_threshold": 1.5,
            "exit_threshold": 0.5,
            "stop_threshold": 3.0,
            "max_holding_period": 20,
            "zscore_window": base_z,
            "use_volatility_filter": False,
            "volatility_percentile": config.validation.volatility_percentile,
            "cooldown_days": 0,
        }
        try:
            result = _run_period(combined, validation_index, candidate, settings, config.costs.transaction_cost_bps_per_leg, fit_window)
            scored = _score_validation(result, validation_index, candidate, settings, "hedge_mode")
            all_records.append(scored[0])
            mode_records.append(scored)
        except Exception:
            continue
    if not mode_records:
        return pd.DataFrame(), {}, pd.Series(dtype=float)
    best_mode = max(mode_records, key=lambda item: (float(item[0]["robust_score"]), -len(str(item[0]["hedge_mode"]))))[0]["hedge_mode"]

    threshold_records: list[tuple[dict[str, object], pd.Series]] = []
    for entry in config.validation.entry_thresholds:
        for exit_ in config.validation.exit_thresholds:
            if exit_ >= entry:
                continue
            settings = {
                "hedge_mode": best_mode,
                "entry_threshold": entry,
                "exit_threshold": exit_,
                "stop_threshold": 3.5 if entry >= 2.0 else 3.0,
                "max_holding_period": 20,
                "zscore_window": base_z,
                "use_volatility_filter": False,
                "volatility_percentile": config.validation.volatility_percentile,
                "cooldown_days": 0,
            }
            result = _run_period(combined, validation_index, candidate, settings, config.costs.transaction_cost_bps_per_leg, fit_window)
            scored = _score_validation(result, validation_index, candidate, settings, "threshold")
            all_records.append(scored[0])
            threshold_records.append(scored)
    best_threshold = max(threshold_records, key=lambda item: float(item[0]["robust_score"]))[0]

    refinement_records: list[tuple[dict[str, object], pd.Series]] = []
    for multiplier in config.validation.zscore_multipliers:
        z_window = int(np.clip(round(base_z * multiplier), 20, 90))
        for stop in config.validation.stop_thresholds:
            if stop <= float(best_threshold["entry_threshold"]):
                continue
            for max_hold in config.validation.maximum_holding_periods:
                settings = {
                    "hedge_mode": best_mode,
                    "entry_threshold": best_threshold["entry_threshold"],
                    "exit_threshold": best_threshold["exit_threshold"],
                    "stop_threshold": stop,
                    "max_holding_period": max_hold,
                    "zscore_window": z_window,
                    "use_volatility_filter": False,
                    "volatility_percentile": config.validation.volatility_percentile,
                    "cooldown_days": 0,
                }
                result = _run_period(combined, validation_index, candidate, settings, config.costs.transaction_cost_bps_per_leg, fit_window)
                scored = _score_validation(result, validation_index, candidate, settings, "refinement")
                all_records.append(scored[0])
                refinement_records.append(scored)
    best_refinement = max(refinement_records, key=lambda item: float(item[0]["robust_score"]))

    filter_records: list[tuple[dict[str, object], pd.Series]] = [best_refinement]
    for use_filter in config.validation.volatility_filter_options:
        settings = {
            key: best_refinement[0][key]
            for key in [
                "hedge_mode",
                "entry_threshold",
                "exit_threshold",
                "stop_threshold",
                "max_holding_period",
                "zscore_window",
                "volatility_percentile",
                "cooldown_days",
            ]
        }
        settings["use_volatility_filter"] = use_filter
        result = _run_period(combined, validation_index, candidate, settings, config.costs.transaction_cost_bps_per_leg, fit_window)
        scored = _score_validation(result, validation_index, candidate, settings, "filter")
        all_records.append(scored[0])
        filter_records.append(scored)

    best = max(filter_records, key=lambda item: float(item[0]["robust_score"]))
    return pd.DataFrame(all_records), best[0], best[1]


def _slice_period(prices: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    return prices.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def run_walk_forward_experiment(
    prices: pd.DataFrame,
    universes: dict[str, dict[str, list[str]]],
    config: ExperimentConfig,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
) -> WalkForwardResult:
    boundaries = walk_forward_boundaries(prices.index, config, evaluation_start, evaluation_end)
    assert_boundary_integrity(boundaries)
    diagnostics_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    validation_daily_frames: list[pd.DataFrame] = []
    pair_daily_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    segment_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []

    for boundary in boundaries.itertuples(index=False):
        train = _slice_period(prices, boundary.train_start, boundary.train_end)
        validation = _slice_period(prices, boundary.validation_start, boundary.validation_end)
        test = _slice_period(prices, boundary.test_start, boundary.test_end)
        diagnostics = discover_training_candidates(
            train,
            universes,
            config.discovery,
            config.costs.transaction_cost_bps_per_leg,
        )
        diagnostics["segment"] = boundary.segment
        diagnostics["train_start"] = boundary.train_start
        diagnostics["train_end"] = boundary.train_end
        diagnostics_frames.append(diagnostics)

        core = diagnostics[
            diagnostics.get("coint_pvalue", pd.Series(index=diagnostics.index, dtype=float)).le(config.discovery.maximum_coint_pvalue)
            & diagnostics.get("adf_pvalue", pd.Series(index=diagnostics.index, dtype=float)).le(config.discovery.maximum_adf_pvalue)
            & diagnostics.get("half_life", pd.Series(index=diagnostics.index, dtype=float)).between(
                config.discovery.minimum_half_life, config.discovery.maximum_half_life
            )
            & diagnostics.get("correlation", pd.Series(index=diagnostics.index, dtype=float)).abs().ge(config.discovery.minimum_correlation)
        ]
        candidates = pd.concat(
            [diagnostics[diagnostics["selected_candidate"]], core], ignore_index=True
        ).drop_duplicates("pair").sort_values(["selected_candidate", "training_score"], ascending=False)
        candidates = candidates.head(config.discovery.maximum_validation_candidates)

        best_records: list[dict[str, object]] = []
        validation_return_series: list[pd.Series] = []
        for _, candidate in candidates.iterrows():
            scores, best, returns = _validate_candidate(train, validation, candidate, config)
            if scores.empty:
                continue
            scores["segment"] = boundary.segment
            scores["train_end"] = boundary.train_end
            scores["validation_end"] = boundary.validation_end
            scores["test_start"] = boundary.test_start
            score_frames.append(scores)
            best_records.append(best)
            validation_return_series.append(returns)

        best_table = pd.DataFrame(best_records)
        validation_returns = pd.concat(validation_return_series, axis=1) if validation_return_series else pd.DataFrame(index=validation.index)
        selection_table = best_table
        validation_for_selection = config.validation
        if config.validation.pair_selection_method == "training_ranked":
            selection_table = best_table[best_table["training_candidate_pass"]].copy()
            selection_table["robust_score"] = selection_table["training_score"]
            validation_for_selection = replace(
                config.validation,
                minimum_trades=0,
                minimum_sharpe=-np.inf,
                minimum_profit_factor=-np.inf,
                minimum_max_drawdown=-np.inf,
            )
        selected, decisions = select_diversified_pairs(
            selection_table,
            validation_returns,
            validation_for_selection,
            config.portfolio,
        )
        if not decisions.empty:
            decisions["segment"] = boundary.segment
            decision_frames.append(decisions)

        if selected.empty:
            cash = pd.DataFrame(
                {
                    "segment": boundary.segment,
                    "portfolio_return_unscaled": 0.0,
                    "portfolio_gross_return_unscaled": 0.0,
                    "pair_turnover": 0.0,
                    "leverage": 0.0,
                    "strategy_return": 0.0,
                    "active_pairs": 0,
                },
                index=test.index,
            )
            portfolio_frames.append(cash)
            segment_rows.append({"segment": boundary.segment, "selected_pairs": 0.0, **performance_metrics(cash["strategy_return"])})
            concentration_rows.append({"segment": boundary.segment, **concentration_metrics(selected)})
            continue

        weights = validation_weights(selected, config.portfolio.weighting_method, config.portfolio.maximum_pair_weight)
        selected = selected.copy()
        selected["weight"] = selected["pair"].map(weights)
        for name in boundaries.columns:
            selected[name] = getattr(boundary, name)
        selection_frames.append(selected)

        selected_validation = validation_returns[selected["pair"].astype(str).tolist()].copy()
        selected_validation.index.name = "date"
        selected_validation = selected_validation.stack().rename("strategy_return").reset_index()
        selected_validation.columns = ["date", "pair", "strategy_return"]
        selected_validation["segment"] = boundary.segment
        validation_daily_frames.append(selected_validation)

        combined = pd.concat([train, validation, test])
        test_pair_returns: list[pd.Series] = []
        test_pair_gross_returns: list[pd.Series] = []
        test_pair_turnover: list[pd.Series] = []
        weighted_validation_returns: list[pd.Series] = []
        segment_trades: list[pd.DataFrame] = []
        pair_pnl: dict[str, float] = {}
        for _, selected_row in selected.iterrows():
            settings = selected_row.to_dict()
            pair_columns = [str(selected_row["ticker_y"]), str(selected_row["ticker_x"])]
            pair_combined = combined[pair_columns].dropna()
            test_index = pair_combined.index.intersection(test.index)
            fit_window = int((pair_combined.index < test_index[0]).sum())
            result = _run_period(
                pair_combined,
                test_index,
                selected_row,
                settings,
                config.costs.transaction_cost_bps_per_leg,
                fit_window,
            )
            pair_name = str(selected_row["pair"])
            pair_returns = result.daily.reindex(test.index)["strategy_return"].fillna(0.0).rename(pair_name)
            pair_gross = result.daily.reindex(test.index)["strategy_return_gross"].fillna(0.0).rename(pair_name)
            pair_turnover = result.daily.reindex(test.index)["position_change"].fillna(0.0).rename(pair_name)
            test_pair_returns.append(pair_returns)
            test_pair_gross_returns.append(pair_gross)
            test_pair_turnover.append(pair_turnover)
            pair_pnl[pair_name] = float((1.0 + pair_returns).prod() - 1.0)
            validation_pair = validation_returns[pair_name].reindex(validation.index).fillna(0.0)
            weighted_validation_returns.append((validation_pair * float(selected_row["weight"])).rename(pair_name))
            if not result.trades.empty:
                trades = result.trades.copy()
                trades = trades[
                    (pd.to_datetime(trades["entry_date"]) >= test.index[0])
                    & (pd.to_datetime(trades["exit_date"]) <= test.index[-1])
                ]
                if not trades.empty:
                    trades["pair"] = pair_name
                    trades["segment"] = boundary.segment
                    trades["universe_mode"] = selected_row["universe_mode"]
                    trades["peer_group"] = selected_row["peer_group"]
                    trades["hedge_mode"] = selected_row["hedge_mode"]
                    trades["weight"] = selected_row["weight"]
                    segment_trades.append(trades)

        pair_frame = pd.concat(test_pair_returns, axis=1).fillna(0.0)
        gross_frame = pd.concat(test_pair_gross_returns, axis=1).fillna(0.0)
        turnover_frame = pd.concat(test_pair_turnover, axis=1).fillna(0.0)
        long_pair = pd.concat(
            [
                pair_frame.stack().rename("strategy_return"),
                gross_frame.stack().rename("strategy_return_gross"),
                turnover_frame.stack().rename("position_change"),
            ],
            axis=1,
        ).reset_index()
        long_pair.columns = ["date", "pair", "strategy_return", "strategy_return_gross", "position_change"]
        long_pair["segment"] = boundary.segment
        long_pair["weight"] = long_pair["pair"].map(weights)
        long_pair["weighted_return"] = long_pair["strategy_return"] * long_pair["weight"]
        pair_daily_frames.append(long_pair)
        unscaled = pair_frame.mul(weights, axis=1).sum(axis=1)
        gross_unscaled = gross_frame.mul(weights, axis=1).sum(axis=1)
        turnover_unscaled = turnover_frame.mul(weights, axis=1).sum(axis=1)

        leverage = pd.Series(1.0, index=test.index)
        if config.portfolio.target_volatility is not None:
            validation_portfolio = pd.concat(weighted_validation_returns, axis=1).sum(axis=1)
            history = pd.concat([validation_portfolio, unscaled])
            history_leverage = lagged_volatility_scale(
                history,
                config.portfolio.target_volatility,
                config.portfolio.volatility_lookback,
                config.portfolio.maximum_leverage,
            )
            leverage = history_leverage.reindex(test.index).fillna(1.0)
        portfolio = pd.DataFrame(
            {
                "segment": boundary.segment,
                "portfolio_return_unscaled": unscaled,
                "portfolio_gross_return_unscaled": gross_unscaled,
                "pair_turnover": turnover_unscaled,
                "leverage": leverage,
                "strategy_return": unscaled * leverage,
                "active_pairs": len(selected),
            }
        )
        portfolio_frames.append(portfolio)
        if segment_trades:
            trade_frames.append(pd.concat(segment_trades, ignore_index=True))
        segment_metrics = performance_metrics(portfolio["strategy_return"])
        segment_rows.append(
            {
                "segment": boundary.segment,
                "selected_pairs": float(len(selected)),
                "completed_trades": float(sum(len(frame) for frame in segment_trades)),
                **segment_metrics,
            }
        )
        concentration_rows.append(
            {"segment": boundary.segment, **concentration_metrics(selected, pd.Series(pair_pnl))}
        )

    return WalkForwardResult(
        boundaries=boundaries,
        candidate_diagnostics=pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame(),
        validation_scores=pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(),
        selection_decisions=pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame(),
        selected_pairs=pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame(),
        validation_daily_returns=pd.concat(validation_daily_frames, ignore_index=True) if validation_daily_frames else pd.DataFrame(),
        pair_daily_returns=pd.concat(pair_daily_frames, ignore_index=True) if pair_daily_frames else pd.DataFrame(),
        portfolio_daily_returns=pd.concat(portfolio_frames).sort_index() if portfolio_frames else pd.DataFrame(),
        trade_log=pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        segment_metrics=pd.DataFrame(segment_rows),
        concentration=pd.DataFrame(concentration_rows),
    )


def reweight_walk_forward_result(result: WalkForwardResult, config: ExperimentConfig) -> WalkForwardResult:
    selected = result.selected_pairs.copy()
    pair_daily = result.pair_daily_returns.copy()
    portfolio_daily = result.portfolio_daily_returns.copy()
    trade_log = result.trade_log.copy()
    segment_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []

    for segment, segment_selection in selected.groupby("segment"):
        weights = validation_weights(
            segment_selection,
            config.portfolio.weighting_method,
            config.portfolio.maximum_pair_weight,
        )
        selected.loc[selected["segment"] == segment, "weight"] = selected.loc[
            selected["segment"] == segment, "pair"
        ].map(weights)
        pair_mask = pair_daily["segment"] == segment
        pair_daily.loc[pair_mask, "weight"] = pair_daily.loc[pair_mask, "pair"].map(weights)
        pair_daily.loc[pair_mask, "weighted_return"] = (
            pair_daily.loc[pair_mask, "strategy_return"] * pair_daily.loc[pair_mask, "weight"]
        )
        segment_pair = pair_daily.loc[pair_mask]
        net = segment_pair.pivot(index="date", columns="pair", values="strategy_return").fillna(0.0).mul(weights, axis=1).sum(axis=1)
        gross = segment_pair.pivot(index="date", columns="pair", values="strategy_return_gross").fillna(0.0).mul(weights, axis=1).sum(axis=1)
        turnover = segment_pair.pivot(index="date", columns="pair", values="position_change").fillna(0.0).mul(weights, axis=1).sum(axis=1)
        portfolio_mask = portfolio_daily["segment"] == segment
        segment_index = portfolio_daily.index[portfolio_mask]
        portfolio_daily.loc[portfolio_mask, "portfolio_return_unscaled"] = net.reindex(segment_index).fillna(0.0)
        portfolio_daily.loc[portfolio_mask, "portfolio_gross_return_unscaled"] = gross.reindex(segment_index).fillna(0.0)
        portfolio_daily.loc[portfolio_mask, "pair_turnover"] = turnover.reindex(segment_index).fillna(0.0)
        portfolio_daily.loc[portfolio_mask, "leverage"] = 1.0
        portfolio_daily.loc[portfolio_mask, "strategy_return"] = net.reindex(segment_index).fillna(0.0)
        if not trade_log.empty:
            trade_mask = trade_log["segment"] == segment
            trade_log.loc[trade_mask, "pair"] = trade_log.loc[trade_mask, "pair"].map(
                lambda value: "/".join(sorted(str(value).split("/")))
            )
            trade_log.loc[trade_mask, "weight"] = trade_log.loc[trade_mask, "pair"].map(weights)
        segment_returns = portfolio_daily.loc[portfolio_mask, "strategy_return"]
        segment_trades = trade_log[trade_log["segment"] == segment] if not trade_log.empty else pd.DataFrame()
        segment_rows.append(
            {
                "segment": segment,
                "selected_pairs": float(len(segment_selection)),
                "completed_trades": float(len(segment_trades)),
                **performance_metrics(segment_returns),
            }
        )
        pair_pnl = segment_pair.groupby("pair")["weighted_return"].sum()
        concentration_rows.append(
            {
                "segment": segment,
                **concentration_metrics(selected[selected["segment"] == segment], pair_pnl),
            }
        )

    return WalkForwardResult(
        boundaries=result.boundaries.copy(),
        candidate_diagnostics=result.candidate_diagnostics.copy(),
        validation_scores=result.validation_scores.copy(),
        selection_decisions=result.selection_decisions.copy(),
        selected_pairs=selected,
        validation_daily_returns=result.validation_daily_returns.copy(),
        pair_daily_returns=pair_daily,
        portfolio_daily_returns=portfolio_daily,
        trade_log=trade_log,
        segment_metrics=pd.DataFrame(segment_rows),
        concentration=pd.DataFrame(concentration_rows),
    )
