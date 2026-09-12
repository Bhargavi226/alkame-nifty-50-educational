import pandas as pd
import pytest

from feature_engineer import FeatureEngineer


def test_daily_outperformance_is_five_percentage_points():
    timestamps = pd.to_datetime([
        "2026-09-08 09:15",
        "2026-09-08 15:15",
    ])

    stock_close = pd.Series([100.0, 110.0], index=timestamps)
    index_close = pd.Series([100.0, 105.0], index=timestamps)

    outperformance, flag = FeatureEngineer.compute_outperformance(
        stock_close,
        index_close,
    )

    assert outperformance.iloc[-1] == pytest.approx(5.0)
    assert flag.iloc[-1]