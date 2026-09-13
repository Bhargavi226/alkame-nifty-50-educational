import numpy as np
import pandas as pd
import pytest

from model_trainer import ModelTrainer


def test_f31_synthetic_user_cost_is_stable_within_each_day():
    days = pd.bdate_range("2026-01-05", periods=10)
    timestamps = pd.DatetimeIndex(
        [
            day + pd.Timedelta(hours=9, minutes=15 + 5 * bar)
            for day in days
            for bar in range(3)
        ]
    )
    df = pd.DataFrame(
        {"Close": np.arange(len(timestamps), dtype=float) + 100.0},
        index=timestamps,
    )

    trainer = ModelTrainer.__new__(ModelTrainer)
    trainer.simulate_user_cost(df, seed=42)

    active_position_rows_found = False

    for _, one_day in df.groupby(df.index.date):
        active = one_day[one_day["has_position"] == 1.0]
        if active.empty:
            continue

        active_position_rows_found = True
        inferred_cost = active["Close"] / (
            1.0 + active["pct_from_user_avg_cost"] / 100.0
        )

        assert inferred_cost.max() == pytest.approx(inferred_cost.min())

    assert active_position_rows_found