import numpy as np
import pandas as pd

from feature_engineer import FeatureEngineer


def test_f29_rsi_rising_falling_flat():
    rising = pd.Series(range(100, 131))
    falling = pd.Series(range(130, 99, -1))
    flat = pd.Series([100.0] * 31)

    rsi_rising = FeatureEngineer.compute_rsi(rising, period=14)
    rsi_falling = FeatureEngineer.compute_rsi(falling, period=14)
    rsi_flat = FeatureEngineer.compute_rsi(flat, period=14)

    assert rsi_rising.iloc[-1] == 100.0
    assert rsi_falling.iloc[-1] == 0.0
    assert rsi_flat.iloc[-1] == 50.0

    valid = pd.concat([rsi_rising, rsi_falling, rsi_flat]).dropna()
    assert valid.between(0, 100).all()


def test_f29_rsi_warmup():
    fourteen = pd.Series(range(100, 114))
    fifteen = pd.Series(range(100, 115))

    rsi_14 = FeatureEngineer.compute_rsi(fourteen, period=14)
    rsi_15 = FeatureEngineer.compute_rsi(fifteen, period=14)

    assert rsi_14.iloc[:-1].isna().all()
    assert rsi_15.iloc[-1] == 100.0


def test_f29_rsi_invalid_inputs_do_not_produce_finite_scores():
    bad = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0])

    result = FeatureEngineer.compute_rsi(bad, period=14)

    assert not np.isfinite(result.dropna()).any()


def test_f29_rsi_period_zero_is_not_valid():
    values = pd.Series(range(100, 120))

    result = FeatureEngineer.compute_rsi(values, period=0)

    assert not np.isfinite(result.dropna()).any()
