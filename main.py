# main.py

from __future__ import annotations

from config import (
    UNIVERSE,
    BENCHMARK,
    START_DATE,
    END_DATE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    MIN_OBSERVATIONS,
    MAX_COINTEGRATION_PVALUE,
    MAX_ADF_PVALUE,
    MIN_HALF_LIFE,
    MAX_HALF_LIFE,
    TOP_N_PAIRS,
    LOOKBACK_GRID,
    ENTRY_Z_GRID,
    EXIT_Z_GRID,
    STOP_Z_GRID,
    RECALIBRATION_WINDOW,
    RECALIBRATION_STEP,
    INITIAL_CAPITAL,
    COMMISSION_BPS,
    SLIPPAGE_BPS,
    TARGET_PORTFOLIO_VOL,
    MAX_GROSS_LEVERAGE,
    MAX_PAIR_WEIGHT,
    MIN_PAIR_VOL,
    QUALITY_WEIGHT_STRENGTH,
    RISK_FREE_RATE,
    REQUIRE_ENTRY_CROSS,
    COOLDOWN_DAYS,
    TREND_LOOKBACK,
    MAX_TREND_ZSCORE_SLOPE,
    VOL_FILTER_LOOKBACK,
    MAX_RECENT_VOL_MULTIPLIER,
    SAVE_RESULTS,
    RESULTS_DIR,
    PLOT_FIGURES,
)
from pipeline import run_research_pipeline
from reporting import (
    print_header,
    print_metrics,
    print_selected_pairs,
    print_pair_params,
    save_outputs,
    plot_portfolio_results,
)


def main() -> None:
    print_header("RUNNING RESEARCH PIPELINE")

    out = run_research_pipeline(
        universe=UNIVERSE,
        benchmark_ticker=BENCHMARK,
        start_date=START_DATE,
        end_date=END_DATE,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        min_observations=MIN_OBSERVATIONS,
        max_coint_pvalue=MAX_COINTEGRATION_PVALUE,
        max_adf_pvalue=MAX_ADF_PVALUE,
        min_half_life=MIN_HALF_LIFE,
        max_half_life=MAX_HALF_LIFE,
        top_n_pairs=TOP_N_PAIRS,
        lookback_grid=LOOKBACK_GRID,
        entry_z_grid=ENTRY_Z_GRID,
        exit_z_grid=EXIT_Z_GRID,
        stop_z_grid=STOP_Z_GRID,
        recalibration_window=RECALIBRATION_WINDOW,
        recalibration_step=RECALIBRATION_STEP,
        initial_capital=INITIAL_CAPITAL,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        target_portfolio_vol=TARGET_PORTFOLIO_VOL,
        max_gross_leverage=MAX_GROSS_LEVERAGE,
        max_pair_weight=MAX_PAIR_WEIGHT,
        min_pair_vol=MIN_PAIR_VOL,
        risk_free_rate=RISK_FREE_RATE,
        cooldown_days=COOLDOWN_DAYS,
        trend_lookback=TREND_LOOKBACK,
        max_trend_zscore_slope=MAX_TREND_ZSCORE_SLOPE,
        vol_filter_lookback=VOL_FILTER_LOOKBACK,
        max_recent_vol_multiplier=MAX_RECENT_VOL_MULTIPLIER,
        require_entry_cross=REQUIRE_ENTRY_CROSS,
        quality_weight_strength=QUALITY_WEIGHT_STRENGTH,
    )

    ranking = out["ranking"]
    selected_pairs = out["selected_pairs"]
    pair_params_df = out["pair_params_df"]
    portfolio_output = out["portfolio_output"]
    split_summary = out["split_summary"]

    print_header("SPLIT SUMMARY")
    print(split_summary)

    selected_cols = [
        c for c in [
            "ticker_y",
            "ticker_x",
            "train_coint_pvalue",
            "train_adf_pvalue",
            "train_half_life",
            "validation_score",
            "validation_sharpe",
            "validation_ann_return",
            "validation_total_return",
        ]
        if c in selected_pairs.columns
    ]
    print_selected_pairs(selected_pairs[selected_cols])

    param_cols = [
        c for c in [
            "pair_name",
            "ticker_y",
            "ticker_x",
            "lookback",
            "entry_z",
            "exit_z",
            "stop_z",
            "validation_score",
            "validation_sharpe",
            "validation_ann_return",
            "validation_total_return",
        ]
        if c in pair_params_df.columns
    ]
    print_pair_params(pair_params_df[param_cols])

    print_metrics(portfolio_output.metrics)

    if hasattr(portfolio_output, "selection_log") and portfolio_output.selection_log is not None:
        print_header("SELECTION LOG (HEAD)")
        if portfolio_output.selection_log.empty:
            print("No dynamic selections recorded.")
        else:
            print(portfolio_output.selection_log.head(20).to_string(index=False))

    if SAVE_RESULTS:
        save_outputs(
            results_dir=RESULTS_DIR,
            pair_ranking=ranking,
            selected_pairs=selected_pairs,
            pair_params=pair_params_df,
            portfolio_results=portfolio_output.portfolio_results,
            pair_results=portfolio_output.pair_results,
            pair_trade_logs=portfolio_output.pair_trade_logs,
            pair_recalibration_logs=portfolio_output.pair_recalibration_logs,
        )
        print(f"\nSaved outputs to: {RESULTS_DIR}")

    if PLOT_FIGURES:
        plot_portfolio_results(portfolio_output.portfolio_results)


if __name__ == "__main__":
    main()