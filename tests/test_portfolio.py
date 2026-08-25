from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PortfolioConfig, ValidationConfig
from src.portfolio import capped_weights, lagged_volatility_scale, select_diversified_pairs


def test_capped_weights_sum_and_respect_cap() -> None:
    weights = capped_weights(pd.Series({"A/B": 100.0, "C/D": 2.0, "E/F": 1.0}), 0.40)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.40 + 1e-12


def test_ticker_concentration_rejects_reused_ticker() -> None:
    summary = pd.DataFrame(
        [
            {"pair": "A/B", "ticker_y": "A", "ticker_x": "B", "peer_group": "g1", "validation_trades": 4, "validation_sharpe": 1.0, "validation_profit_factor": 2.0, "validation_max_drawdown": -0.05, "robust_score": 1.0, "stability_score": 0.9},
            {"pair": "A/C", "ticker_y": "A", "ticker_x": "C", "peer_group": "g2", "validation_trades": 4, "validation_sharpe": 0.9, "validation_profit_factor": 1.8, "validation_max_drawdown": -0.04, "robust_score": 0.9, "stability_score": 0.8},
            {"pair": "D/E", "ticker_y": "D", "ticker_x": "E", "peer_group": "g2", "validation_trades": 4, "validation_sharpe": 0.8, "validation_profit_factor": 1.7, "validation_max_drawdown": -0.04, "robust_score": 0.8, "stability_score": 0.8},
        ]
    )
    returns = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 3)), columns=summary["pair"])
    selected, decisions = select_diversified_pairs(summary, returns, ValidationConfig(), PortfolioConfig())
    assert "A/C" not in set(selected["pair"])
    assert "ticker_concentration" in set(decisions["selection_decision"])


def test_volatility_scaling_is_lagged() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.01, 0.01, 0.03, -0.02])
    changed = returns.copy()
    changed.iloc[5] = 0.50
    original_scale = lagged_volatility_scale(returns, 0.08, 3, 1.5)
    changed_scale = lagged_volatility_scale(changed, 0.08, 3, 1.5)
    assert original_scale.iloc[5] == changed_scale.iloc[5]
