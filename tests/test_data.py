import numpy as np
import pandas as pd

from src.data import compute_returns


def test_compute_returns_flat_prices_yield_zero():
    """If all prices are constant, all returns must be exactly zero."""
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    prices = pd.DataFrame({"AAA": [100.0] * 90, "BBB": [50.0] * 90}, index=dates)

    returns = compute_returns(prices)

    assert (returns == 0.0).all().all(), "Flat prices should produce zero returns"


def test_compute_returns_known_ramp():
    """A 10% monthly increase should produce 10% monthly returns."""
    # Pick the last day of each month explicitly
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    prices = pd.DataFrame({"X": [100.0, 110.0, 121.0, 133.1]}, index=dates)

    returns = compute_returns(prices)

    expected = pd.Series([0.10, 0.10, 0.10], name="X")
    np.testing.assert_array_almost_equal(returns["X"].values, expected.values, decimal=10)


def test_compute_returns_output_shape():
    """Output should have one fewer row than the number of months in input."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    prices = pd.DataFrame({"X": np.linspace(100.0, 200.0, len(dates))}, index=dates)

    returns = compute_returns(prices)

    # 12 months of input → 11 monthly returns (first is dropped)
    assert returns.shape == (11, 1)
    assert returns.columns.tolist() == ["X"]
