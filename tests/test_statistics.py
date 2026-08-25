from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics import block_bootstrap_intervals, newey_west_mean_test


def test_statistical_evidence_is_deterministic() -> None:
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0.0002, 0.01, 500))
    first = block_bootstrap_intervals(returns, simulations=100, seed=5)
    second = block_bootstrap_intervals(returns, simulations=100, seed=5)
    assert first == second
    inference = newey_west_mean_test(returns)
    assert set(inference) == {"annualised_mean_return", "newey_west_t_stat", "newey_west_pvalue"}
