import numpy as np
import pandas as pd
import pytest

from model_trainer import ModelTrainer


def _make_dataset(n_rows=100):
    index = pd.date_range(
        "2026-01-01",
        periods=n_rows,
        freq="5min",
    )

    X = pd.DataFrame(
        {"feature": np.arange(n_rows, dtype=float)},
        index=index,
    )

    y = pd.Series(
        np.where(np.arange(n_rows) % 2 == 0, "UP", "DOWN"),
        index=index,
        name="label",
    )

    return X, y


def test_time_based_split_purges_forward_label_overlap():
    """
    F13: A training label must not consume data from the protected
    test interval.

    Test boundary = row 80.
    Horizon = 6 bars.

    Row 74 would use row 80 as its future label endpoint, so row 74
    must not remain in training.
    """
    trainer = ModelTrainer()

    X, y = _make_dataset(100)

    X_train, X_test, y_train, y_test = trainer.time_based_split(
        X,
        y,
        test_fraction=0.20,
        purge_window=6,
    )

    assert X_test.index[0] == X.index[80]

    # Last training row must be row 73.
    assert X_train.index[-1] == X.index[73]

    # Therefore no training label endpoint can reach row 80.
    training_positions = np.arange(len(X_train))
    label_end_positions = training_positions + 6

    assert label_end_positions.max() < 80

    # X/y alignment must be preserved.
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_time_based_split_excludes_exact_boundary_for_252_bar_horizon():
    """
    F13: Repeat the same boundary rule for a 252-bar horizon.
    """
    trainer = ModelTrainer()

    n_rows = 600
    X, y = _make_dataset(n_rows)

    X_train, X_test, y_train, y_test = trainer.time_based_split(
        X,
        y,
        test_fraction=0.20,
        purge_window=252,
    )

    split_idx = 480

    assert X_test.index[0] == X.index[split_idx]

    # No training label may end at or after the test boundary.
    if len(X_train) > 0:
        last_train_position = len(X_train) - 1
        assert last_train_position + 252 < split_idx

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_time_based_split_rejects_unsorted_or_duplicate_timestamps():
    trainer = ModelTrainer()

    index = pd.date_range("2025-01-01", periods=100, freq="h")
    X = pd.DataFrame({"feature": range(100)}, index=index)
    y = pd.Series(range(100), index=index)

    X_unsorted = X.iloc[::-1].copy()
    y_unsorted = y.iloc[::-1].copy()

    with pytest.raises(ValueError):
        trainer.time_based_split(
            X_unsorted,
            y_unsorted,
            test_fraction=0.20,
            purge_window=6,
        )

    duplicate_index = X.index.copy()
    duplicate_index = duplicate_index.delete(50)
    duplicate_index = duplicate_index.insert(49, X.index[49])

    X_duplicate = X.copy()
    y_duplicate = y.copy()
    X_duplicate.index = duplicate_index
    y_duplicate.index = duplicate_index

    with pytest.raises(ValueError):
        trainer.time_based_split(
            X_duplicate,
            y_duplicate,
            test_fraction=0.20,
            purge_window=6,
        )