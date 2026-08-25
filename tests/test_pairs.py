from __future__ import annotations

import numpy as np
import pandas as pd

from src.pairs import analyse_pair, estimate_hedge_ratio, screen_pairs


def test_cointegrated_pair_is_identified(synthetic_prices: pd.DataFrame) -> None:
    diagnostics = analyse_pair(synthetic_prices["A"], synthetic_prices["B"], zscore_window=60)
    assert diagnostics["coint_pvalue"] < 0.05
    assert diagnostics["adf_pvalue"] < 0.05
    assert diagnostics["hedge_ratio"] > 0


def test_unrelated_pair_is_not_selected(synthetic_prices: pd.DataFrame) -> None:
    screened = screen_pairs(
        synthetic_prices[["A", "C"]],
        min_abs_correlation=0.80,
        max_coint_pvalue=0.05,
        max_adf_pvalue=0.05,
        min_half_life=2,
        max_half_life=60,
        min_threshold_crossings=4,
    )
    assert not bool(screened.iloc[0]["selected_candidate"])


def test_ols_hedge_ratio_scale_is_sensible() -> None:
    rng = np.random.default_rng(4)
    x = pd.Series(np.linspace(1.0, 10.0, 500))
    y = 2.0 + 1.5 * x + pd.Series(rng.normal(0.0, 0.05, len(x)))
    hedge_ratio, intercept = estimate_hedge_ratio(y, x)
    assert abs(hedge_ratio - 1.5) < 0.02
    assert abs(intercept - 2.0) < 0.1
