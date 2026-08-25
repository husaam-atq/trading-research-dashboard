from __future__ import annotations

import pandas as pd
import pytest

from src.data import validate_price_data


def test_valid_prices_are_chronological_and_unique(synthetic_prices: pd.DataFrame) -> None:
    validate_price_data(synthetic_prices)


def test_duplicate_timestamps_are_rejected(synthetic_prices: pd.DataFrame) -> None:
    duplicated = pd.concat([synthetic_prices.iloc[:2], synthetic_prices.iloc[[1]], synthetic_prices.iloc[2:]])
    with pytest.raises(ValueError, match="unique"):
        validate_price_data(duplicated)


def test_nonpositive_prices_are_rejected(synthetic_prices: pd.DataFrame) -> None:
    invalid = synthetic_prices.copy()
    invalid.iloc[10, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        validate_price_data(invalid)


def test_missing_values_are_allowed_per_instrument(synthetic_prices: pd.DataFrame) -> None:
    prices = synthetic_prices.copy()
    prices.iloc[:20, 0] = float("nan")
    validate_price_data(prices)
