import numpy as np
import pandas as pd
import pytest

from feature_engineer import FeatureEngineer


def test_daily_outperformance_uses_consecutive_closes():
    idx = pd.date_range("2026-01-01", periods=2, freq="D")

    stock = pd.Series([100.0, 110.0], index=idx)
    index = pd.Series([100.0, 105.0], index=idx)

    outperformance, flag = (
        FeatureEngineer.compute_outperformance(stock, index)
    )

    assert np.isnan(outperformance.iloc[0])
    assert outperformance.iloc[1] == pytest.approx(5.0)
    assert bool(flag.iloc[1]) is True