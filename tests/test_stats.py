import numpy as np
import pandas as pd
import pytest

from src.stats import arithmetic_mean, covariance_matrix, geometric_mean, volatility


def test_arithmetic_mean_of_constant_returns():
    """Mean of constant series equals the constant."""
    returns = pd.DataFrame({"X": [0.05] * 10, "Y": [-0.02] * 10})
    result = arithmetic_mean(returns)
    assert result["X"] == pytest.approx(0.05)
    assert result["Y"] == pytest.approx(-0.02)


def test_geometric_mean_of_constant_returns():
    """For a constant return r, geometric mean equals r exactly."""
    returns = pd.DataFrame({"X": [0.10] * 12})
    result = geometric_mean(returns)
    assert result["X"] == pytest.approx(0.10)


def test_geometric_mean_le_arithmetic_mean():
    """Jensen's inequality: geometric mean <= arithmetic mean, strict if any volatility."""
    rng = np.random.default_rng(seed=42)
    returns = pd.DataFrame({"X": rng.normal(0.01, 0.05, size=120)})
    arith = arithmetic_mean(returns)["X"]
    geo = geometric_mean(returns)["X"]
    assert geo < arith
    # The gap is approximately σ²/2; with σ=0.05, expect ~0.00125
    assert (arith - geo) < 0.005  # loose upper bound


def test_covariance_diagonal_equals_variance():
    """Diagonal of covariance matrix should equal variance (volatility squared)."""
    rng = np.random.default_rng(seed=42)
    returns = pd.DataFrame(
        {
            "X": rng.normal(0.01, 0.05, size=120),
            "Y": rng.normal(0.005, 0.03, size=120),
        }
    )
    cov = covariance_matrix(returns)
    vol = volatility(returns)
    np.testing.assert_array_almost_equal(np.diag(cov.values), (vol**2).values, decimal=12)
