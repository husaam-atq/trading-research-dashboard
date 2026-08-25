from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.config import ExperimentConfig
from src.experiment import assert_boundary_integrity, walk_forward_boundaries


def test_walk_forward_boundaries_are_ordered_and_contiguous(business_index: pd.DatetimeIndex) -> None:
    config = ExperimentConfig(train_window=252, validation_window=63, test_window=21, step_size=21)
    boundaries = walk_forward_boundaries(business_index, config)
    assert_boundary_integrity(boundaries)
    for previous, current in zip(boundaries.itertuples(index=False), boundaries.iloc[1:].itertuples(index=False)):
        previous_position = business_index.get_loc(previous.test_end)
        current_position = business_index.get_loc(current.test_start)
        assert current_position == previous_position + 1


def test_confirmation_start_never_enters_model_selection(business_index: pd.DatetimeIndex) -> None:
    config = ExperimentConfig(train_window=252, validation_window=63, test_window=21, step_size=21)
    confirmation_start = business_index[500].strftime("%Y-%m-%d")
    boundaries = walk_forward_boundaries(business_index, config, evaluation_start=confirmation_start)
    assert (boundaries["validation_end"] < boundaries["test_start"]).all()
    assert (boundaries["test_start"] >= pd.Timestamp(confirmation_start)).all()


def test_deterministic_boundaries(business_index: pd.DatetimeIndex) -> None:
    config = replace(ExperimentConfig(), train_window=252, validation_window=63, test_window=21, step_size=21)
    pd.testing.assert_frame_equal(
        walk_forward_boundaries(business_index, config),
        walk_forward_boundaries(business_index, config),
    )
