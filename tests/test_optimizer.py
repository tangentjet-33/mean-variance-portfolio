import numpy as np
import pandas as pd
import pytest

from src.optimizer import solve_mvo_cvxpy, solve_mvo_scipy


def _toy_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Synthetic 3-asset universe for testing the optimizer."""
    tickers = ["A", "B", "C"]
    mu = pd.Series([0.01, 0.005, 0.012], index=tickers)
    # A simple PSD covariance matrix
    sigma = pd.DataFrame(
        [
            [0.0040, 0.0010, 0.0015],
            [0.0010, 0.0020, 0.0008],
            [0.0015, 0.0008, 0.0050],
        ],
        index=tickers,
        columns=tickers,
    )
    return mu, sigma


def test_scipy_solver_basic_constraints():
    """scipy solver: weights sum to 1, all >= 0, return >= target."""
    mu, sigma = _toy_inputs()
    target = 0.008
    w = solve_mvo_scipy(mu, sigma, target_return=target)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()  # tolerate tiny floating-point negatives
    assert (mu * w).sum() >= target - 1e-8


def test_cvxpy_solver_basic_constraints():
    """cvxpy solver: weights sum to 1, all >= 0, return >= target."""
    mu, sigma = _toy_inputs()
    target = 0.008
    w = solve_mvo_cvxpy(mu, sigma, target_return=target)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()
    assert (mu * w).sum() >= target - 1e-8


def test_scipy_and_cvxpy_agree():
    """Both solvers solve the same convex problem; weights must agree."""
    mu, sigma = _toy_inputs()
    target = 0.008
    w_scipy = solve_mvo_scipy(mu, sigma, target_return=target)
    w_cvxpy = solve_mvo_cvxpy(mu, sigma, target_return=target)
    np.testing.assert_array_almost_equal(w_scipy.values, w_cvxpy.values, decimal=4)
