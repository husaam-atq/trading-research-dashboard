from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from src.config import (
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    ExperimentConfig,
    ValidationConfig,
    freeze_config,
    load_frozen_config,
)
from src.data import flatten_universes, load_or_download_adjusted_close, research_universes
from src.experiment import reweight_walk_forward_result, run_walk_forward_experiment
from src.reporting import format_summary_markdown, load_walk_forward_result, save_walk_forward_result
from src.statistics import portfolio_method_comparison, volatility_target_comparison
from src.universes import expanded_research_universes, flatten_expanded_universe


REPORTS_DIR = Path("reports")
DEVELOPMENT_DIR = REPORTS_DIR / "development"
CONFIRMATION_DIR = REPORTS_DIR / "confirmation"
FROZEN_CONFIG_PATH = Path("config/final_methodology_v1.json")
DEVELOPMENT_CACHE = Path("data/cache/development_prices.csv")
CONFIRMATION_CACHE = Path("data/cache/confirmation_prices.csv")


def _existing_universe_config() -> ExperimentConfig:
    base = ExperimentConfig(name="rolling_discovery_existing_universe")
    return replace(base, discovery=replace(base.discovery, maximum_validation_candidates=6))


def _expanded_universe_config() -> ExperimentConfig:
    base = ExperimentConfig(name="expanded_rolling_stability")
    return replace(base, discovery=replace(base.discovery, maximum_validation_candidates=8))


def _parsimonious_config() -> ExperimentConfig:
    base = ExperimentConfig(name="parsimonious_expanded_kalman")
    validation = ValidationConfig(
        hedge_modes=("kalman",),
        entry_thresholds=(2.0,),
        exit_thresholds=(0.0,),
        stop_thresholds=(3.5,),
        maximum_holding_periods=(20,),
        zscore_multipliers=(1.0,),
        volatility_filter_options=(False,),
        volatility_percentile=0.90,
        minimum_trades=2,
        minimum_sharpe=0.0,
        minimum_profit_factor=1.0,
        minimum_max_drawdown=-0.15,
    )
    return replace(
        base,
        discovery=replace(base.discovery, maximum_validation_candidates=8),
        validation=validation,
        portfolio=replace(base.portfolio, target_volatility=None),
    )


def _training_ranked_config() -> ExperimentConfig:
    base = _parsimonious_config()
    validation = replace(base.validation, pair_selection_method="training_ranked")
    portfolio = replace(base.portfolio, weighting_method="equal_weight", target_volatility=None)
    return replace(base, name="training_ranked_fixed_kalman", validation=validation, portfolio=portfolio)


def _final_config() -> ExperimentConfig:
    base = _training_ranked_config()
    portfolio = replace(base.portfolio, weighting_method="constrained_inverse_volatility")
    return replace(base, name="final_training_ranked_fixed_kalman", portfolio=portfolio)


def _save_sensitivity_tables(result: object, config: ExperimentConfig, output_dir: Path) -> None:
    methods, method_daily = portfolio_method_comparison(
        result.selected_pairs,
        result.validation_daily_returns,
        result.pair_daily_returns,
        config,
        result.portfolio_daily_returns.index,
    )
    methods.to_csv(output_dir / "portfolio_method_comparison.csv", index=False)
    method_daily.to_csv(output_dir / "portfolio_method_daily_returns.csv", index_label="date")
    targets, target_daily = volatility_target_comparison(
        result.selected_pairs,
        result.validation_daily_returns,
        result.pair_daily_returns,
        config,
        all_index=result.portfolio_daily_returns.index,
    )
    targets.to_csv(output_dir / "volatility_target_comparison.csv", index=False)
    target_daily.to_csv(output_dir / "volatility_target_daily_returns.csv", index_label="date")


def _run_and_save(
    prices: pd.DataFrame,
    universes: dict[str, dict[str, list[str]]],
    config: ExperimentConfig,
    output_dir: Path,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    required = output_dir / "portfolio_daily_returns.csv"
    if resume and required.exists():
        print(f"Resuming reporting for {config.name} from saved deterministic tables...")
        result = load_walk_forward_result(output_dir)
    else:
        print(f"Running {config.name}...")
        result = run_walk_forward_experiment(
            prices,
            universes,
            config,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
    spy_returns = prices["SPY"].pct_change().reindex(result.portfolio_daily_returns.index).fillna(0.0)
    tables = save_walk_forward_result(
        result,
        output_dir,
        spy_returns,
        config.costs.sensitivity_bps,
        config.costs.borrow_sensitivity_rates,
        config.name.replace("_", " ").title(),
    )
    _save_sensitivity_tables(result, config, output_dir)
    summary = tables["performance_metrics"].copy()
    summary.insert(0, "experiment", config.name)
    (output_dir / "summary.md").write_text(
        format_summary_markdown(config.name.replace("_", " ").title(), summary.iloc[0]),
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return summary


def run_development(experiment: str, refresh_data: bool, resume: bool) -> None:
    prices = load_or_download_adjusted_close(
        flatten_expanded_universe(),
        DEVELOPMENT_START,
        DEVELOPMENT_END,
        DEVELOPMENT_CACHE,
        refresh=refresh_data,
    )
    DEVELOPMENT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []

    if experiment in {"existing", "both", "all"}:
        config = _existing_universe_config()
        summaries.append(
            _run_and_save(
                prices,
                research_universes(),
                config,
                DEVELOPMENT_DIR / "experiments" / config.name,
                resume=resume,
            )
        )
    if experiment in {"expanded", "both", "all"}:
        config = _expanded_universe_config()
        summaries.append(
            _run_and_save(
                prices,
                expanded_research_universes(),
                config,
                DEVELOPMENT_DIR / "experiments" / config.name,
                resume=resume,
            )
        )
    if experiment in {"parsimonious", "all"}:
        config = _parsimonious_config()
        summaries.append(
            _run_and_save(
                prices,
                expanded_research_universes(),
                config,
                DEVELOPMENT_DIR / "experiments" / config.name,
                resume=resume,
            )
        )
    if experiment in {"training_ranked", "all"}:
        config = _training_ranked_config()
        summaries.append(
            _run_and_save(
                prices,
                expanded_research_universes(),
                config,
                DEVELOPMENT_DIR / "experiments" / config.name,
                resume=resume,
            )
        )
    if experiment == "final":
        config = load_frozen_config(FROZEN_CONFIG_PATH)[0] if FROZEN_CONFIG_PATH.exists() else _final_config()
        summaries.append(
            _run_and_save(
                prices,
                expanded_research_universes(),
                config,
                DEVELOPMENT_DIR / "final",
                resume=resume,
            )
        )

    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(DEVELOPMENT_DIR / "experiment_comparison.csv", index=False)
    lines = [
        "# Development Experiment Comparison",
        "",
        "All rows use 2015-2024 data only. Test windows are contiguous and start flat after model selection.",
        "",
        "| Experiment | Return | Sharpe | Max DD | Trades | Avg pairs | Unique pairs | Groups | Positive segments | Max pair P&L share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.total_return:.2%} | {row.sharpe_ratio:.2f} | {row.max_drawdown:.2%} | "
            f"{int(row.completed_trades)} | {row.average_selected_pairs:.2f} | {int(row.unique_pairs)} | "
            f"{int(row.peer_groups)} | {row.positive_segment_rate:.1%} | {row.maximum_pair_pnl_share:.1%} |"
        )
    (DEVELOPMENT_DIR / "development_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved development evidence to {DEVELOPMENT_DIR.resolve()}")


def run_confirmation(end_date: str, refresh_data: bool, allow_rerun: bool) -> None:
    marker = CONFIRMATION_DIR / "CONFIRMATION_RUN.json"
    if marker.exists() and not allow_rerun:
        raise RuntimeError("Confirmation evidence already exists. Refusing to rerun without --allow-confirmation-rerun.")
    if not FROZEN_CONFIG_PATH.exists():
        raise RuntimeError("Freeze config/final_methodology_v1.json before requesting confirmation data.")
    config, gates, digest = load_frozen_config(FROZEN_CONFIG_PATH)
    expected_end = gates.get("confirmation_period_end")
    if expected_end and end_date != expected_end:
        raise RuntimeError(f"Confirmation end {end_date} does not match frozen end {expected_end}.")

    prices = load_or_download_adjusted_close(
        flatten_expanded_universe(),
        DEVELOPMENT_START,
        end_date,
        CONFIRMATION_CACHE,
        refresh=refresh_data,
    )
    summary = _run_and_save(
        prices,
        expanded_research_universes(),
        config,
        CONFIRMATION_DIR,
        evaluation_start=CONFIRMATION_START,
        evaluation_end=end_date,
    )
    marker_payload = {
        "frozen_config": str(FROZEN_CONFIG_PATH),
        "frozen_config_sha256": digest,
        "confirmation_start": CONFIRMATION_START,
        "confirmation_end": end_date,
        "observed_summary": summary.iloc[0].to_dict(),
    }
    marker.write_text(json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Confirmation run recorded at {marker.resolve()}")


def freeze_final_methodology() -> None:
    source_dir = DEVELOPMENT_DIR / "experiments" / "training_ranked_fixed_kalman"
    if not (source_dir / "portfolio_daily_returns.csv").exists():
        raise RuntimeError("Run the training-ranked development experiment before freezing methodology.")
    config = _final_config()
    prices = load_or_download_adjusted_close(
        flatten_expanded_universe(),
        DEVELOPMENT_START,
        DEVELOPMENT_END,
        DEVELOPMENT_CACHE,
    )
    source = load_walk_forward_result(source_dir)
    final_result = reweight_walk_forward_result(source, config)
    final_dir = DEVELOPMENT_DIR / "final"
    spy_returns = prices["SPY"].pct_change().reindex(final_result.portfolio_daily_returns.index).fillna(0.0)
    tables = save_walk_forward_result(
        final_result,
        final_dir,
        spy_returns,
        config.costs.sensitivity_bps,
        config.costs.borrow_sensitivity_rates,
        "Final Development Methodology",
    )
    _save_sensitivity_tables(final_result, config, final_dir)
    gates = {
        "confirmation_period_start": CONFIRMATION_START,
        "confirmation_period_end": "2026-07-31",
        "aspirational_net_sharpe": 0.75,
        "minimum_evidence_trades": 75,
        "minimum_unique_pairs": 10,
        "minimum_peer_groups": 5,
        "preferred_maximum_drawdown": -0.15,
        "preferred_maximum_pair_pnl_share": 0.25,
        "preferred_positive_segment_rate": 0.50,
        "transaction_cost_bps_per_leg": config.costs.transaction_cost_bps_per_leg,
    }
    digest = freeze_config(config, FROZEN_CONFIG_PATH, gates)
    summary = tables["performance_metrics"].iloc[0]
    method_text = f"""# Final Frozen Methodology

- Frozen configuration: `{FROZEN_CONFIG_PATH}`
- SHA-256: `{digest}`
- Development period: 2015-2024
- Confirmation period: 2025-01-01 through 2026-07-31

## Decision

The final method uses rolling training-only pair discovery, repeated 252-day stability diagnostics, a fixed Kalman hedge, fixed 2.0 entry / 0.0 exit / 3.5 stop thresholds, a 20-day time stop, training-ranked diversified pair selection, constrained inverse-volatility weights, no volatility target, and 5 bps per leg per one-way position change.

It was selected because it is simpler and less dependent on noisy 126-day validation winners than the staged optimiser. Constrained inverse-volatility weighting improved development drawdown and risk-adjusted performance without changing pair identities, strategy parameters, or trade count. The more complex score-risk weighting was rejected despite a slightly less negative development result.

## Development Evidence At Freeze

{format_summary_markdown('Final Development Methodology', summary)}

The development result is negative. The methodology is frozen for research credibility and breadth, not because it demonstrates profitable alpha.
"""
    (REPORTS_DIR / "final_methodology.md").write_text(method_text, encoding="utf-8")
    print(f"Frozen methodology SHA-256: {digest}")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-controlled pairs-trading research.")
    parser.add_argument("--phase", choices=["development", "freeze", "confirmation"], default="development")
    parser.add_argument(
        "--experiment",
        choices=["final", "existing", "expanded", "both", "parsimonious", "training_ranked", "all"],
        default="final",
    )
    parser.add_argument("--confirmation-end", default="2026-07-31")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--allow-confirmation-rerun", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "development":
        run_development(args.experiment, args.refresh_data, args.resume)
    elif args.phase == "freeze":
        freeze_final_methodology()
    else:
        run_confirmation(args.confirmation_end, args.refresh_data, args.allow_confirmation_rerun)


if __name__ == "__main__":
    main()
