"""Test asymmetric linear regression."""
import numpy as np


from test.test_data_generate import generate_saw
from nlnorm.asymmetric.linear_regression import AsymmetricLinearRegression


def test_saw_with_trend():
    """Test detrending saw with trend"""

    saw = generate_saw(12, 3, 6)
    trend = np.arange(12) * 0.5
    data = saw + trend

    asym_lin_reg = AsymmetricLinearRegression(0.0001, 1e-6)
    predicted_trend = asym_lin_reg.predict(np.arange(12), data)

    assert np.std(trend-predicted_trend) < 0.001


def test_saw_with_trend_and_anomaly():
    """Test detrending saw with trend and anomaly outlier."""

    saw = generate_saw(12, 3, 6)
    saw[4] = -100

    trend = np.arange(12) * 0.5
    data = saw + trend

    asym_lin_reg = AsymmetricLinearRegression(0.0001, 1e-6, outliers=0.01)
    predicted_trend = asym_lin_reg.predict(np.arange(12), data)

    assert np.std(trend-predicted_trend) < 0.001
