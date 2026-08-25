from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


DEVELOPMENT_START = "2015-01-01"
DEVELOPMENT_END = "2024-12-31"
CONFIRMATION_START = "2025-01-01"


@dataclass(frozen=True)
class DiscoveryConfig:
    minimum_coverage: float = 0.95
    correlation_prescreen: float = 0.75
    minimum_correlation: float = 0.80
    maximum_coint_pvalue: float = 0.10
    maximum_adf_pvalue: float = 0.10
    minimum_half_life: float = 2.0
    maximum_half_life: float = 60.0
    minimum_crossings: int = 6
    minimum_training_trades: int = 4
    minimum_subwindow_pass_rate: float = 0.50
    subwindow_length: int = 252
    maximum_validation_candidates: int = 10


@dataclass(frozen=True)
class ValidationConfig:
    pair_selection_method: str = "validation_score"
    hedge_modes: tuple[str, ...] = ("static", "rolling_126", "rolling_252", "kalman")
    entry_thresholds: tuple[float, ...] = (1.50, 1.75, 2.00)
    exit_thresholds: tuple[float, ...] = (0.0, 0.5)
    stop_thresholds: tuple[float, ...] = (3.0, 3.5)
    maximum_holding_periods: tuple[int, ...] = (10, 20, 30)
    zscore_multipliers: tuple[float, ...] = (0.75, 1.0, 1.5)
    volatility_filter_options: tuple[bool, ...] = (False, True)
    volatility_percentile: float = 0.90
    minimum_trades: int = 2
    minimum_sharpe: float = 0.0
    minimum_profit_factor: float = 1.0
    minimum_max_drawdown: float = -0.15


@dataclass(frozen=True)
class PortfolioConfig:
    top_n_pairs: int = 4
    weighting_method: str = "constrained_inverse_volatility"
    maximum_pair_weight: float = 0.35
    maximum_pairs_per_ticker: int = 1
    maximum_pairs_per_peer_group: int = 2
    maximum_pair_return_correlation: float = 0.80
    target_volatility: float | None = 0.08
    volatility_lookback: int = 63
    maximum_leverage: float = 1.25


@dataclass(frozen=True)
class CostConfig:
    transaction_cost_bps_per_leg: float = 5.0
    annual_short_borrow_rate: float = 0.0
    sensitivity_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0)
    borrow_sensitivity_rates: tuple[float, ...] = (0.0, 0.01, 0.03)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "expanded_rolling_discovery"
    train_window: int = 504
    validation_window: int = 126
    test_window: int = 63
    step_size: int = 63
    universe_version: str = "expanded_liquid_peers_v1"
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(config: ExperimentConfig, evaluation_gates: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"methodology": config.to_dict()}
    if evaluation_gates is not None:
        payload["evaluation_gates"] = evaluation_gates
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def freeze_config(
    config: ExperimentConfig,
    path: Path,
    evaluation_gates: dict[str, Any],
) -> str:
    text = canonical_json(config, evaluation_gates)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Frozen methodology already exists with different contents: {path}")
    path.write_bytes(text.encode("utf-8"))
    digest = sha256(text.encode("utf-8")).hexdigest()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    digest_path.write_bytes(f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def load_frozen_config(path: Path) -> tuple[ExperimentConfig, dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    method = payload["methodology"]
    config = ExperimentConfig(
        name=method["name"],
        train_window=method["train_window"],
        validation_window=method["validation_window"],
        test_window=method["test_window"],
        step_size=method["step_size"],
        universe_version=method["universe_version"],
        discovery=DiscoveryConfig(**method["discovery"]),
        validation=ValidationConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in method["validation"].items()}),
        portfolio=PortfolioConfig(**method["portfolio"]),
        costs=CostConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in method["costs"].items()}),
    )
    return config, payload.get("evaluation_gates", {}), sha256(text.encode("utf-8")).hexdigest()
