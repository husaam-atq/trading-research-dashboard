from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def business_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=800)


@pytest.fixture
def synthetic_prices(business_index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    common = np.cumsum(rng.normal(0.0003, 0.01, len(business_index)))
    noise = rng.normal(0.0, 0.015, len(business_index))
    independent = np.cumsum(rng.normal(0.0002, 0.012, len(business_index)))
    return pd.DataFrame(
        {
            "A": 100.0 * np.exp(common),
            "B": 80.0 * np.exp(0.9 * common + noise),
            "C": 60.0 * np.exp(independent),
            "SPY": 100.0 * np.exp(common + rng.normal(0.0, 0.002, len(business_index))),
        },
        index=business_index,
    )
