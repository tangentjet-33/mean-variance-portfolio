"""
Quadratic-program solvers for mean-variance optimization.

Provides two implementations of the same MVO problem — one using scipy.optimize
(low-level), one using cvxpy (high-level convex modeling) — for cross-validation
and pedagogical comparison.
"""

import logging

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def solve_mvo_scipy(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Solve the mean-variance optimization problem using ``scipy.optimize.minimize``.

    Minimizes portfolio variance subject to a target return, full investment,
    and weight bounds::

        min   x.T @ Σ @ x
        s.t.  μ.T @ x >= target_return
              sum(x)   == 1
              min_weight <= x_i <= max_weight  for all i

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (μ), indexed by ticker.
    cov_matrix : pd.DataFrame
        Covariance matrix (Σ) of asset returns, square, indexed by ticker on
        both axes.
    target_return : float
        Minimum acceptable portfolio expected return.
    min_weight : float
        Lower bound on each individual asset weight. Default ``0.0`` (long-only).
        Set to a negative value to allow short-selling, or to a positive value
        like ``0.05`` to enforce a minimum allocation per asset.
    max_weight : float
        Upper bound on each individual asset weight. Default ``1.0``.

    Returns
    -------
    pd.Series
        Optimal weights summing to 1, indexed by ticker.

    Raises
    ------
    RuntimeError
        If the solver fails to converge.
    """
    # 1. Convert pandas types to numpy
    mu = expected_returns.to_numpy()
    sigma = cov_matrix.to_numpy()
    n = len(mu)
    tickers = expected_returns.index

    # 2. Define objective: x.T @ Σ @ x
    def objective(x):
        return x @ sigma @ x

    def objective_jac(x):
        return 2 * sigma @ x

    # 3. Define constraints
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: x.sum() - 1.0,
            "jac": lambda x: np.ones(n),
        },
        {
            "type": "ineq",
            "fun": lambda x: mu @ x - target_return,
            "jac": lambda x: mu,
        },
    ]

    # 4. Define bounds
    bounds = [(min_weight, max_weight)] * n

    # 5. Initial guess: equal weights
    x0 = np.ones(n) / n

    # 6. Solve
    result = minimize(
        objective,
        x0,
        jac=objective_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    # 7. Check convergence
    if not result.success:
        raise RuntimeError(f"scipy optimizer failed: {result.message}")

    logger.info(
        f"scipy MVO converged: variance={objective(result.x):.6f}, return={mu @ result.x:.6f}"
    )

    # 8. Wrap weights back into pd.Series with ticker labels
    return pd.Series(result.x, index=tickers, name="weights")


def solve_mvo_cvxpy(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Solve the mean-variance optimization problem using ``cvxpy``.

    Identical formulation to ``solve_mvo_scipy``; provided as a sanity check
    and to demonstrate the higher-level convex-modeling style.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (μ), indexed by ticker.
    cov_matrix : pd.DataFrame
        Covariance matrix (Σ) of asset returns, square, indexed by ticker on
        both axes.
    target_return : float
        Minimum acceptable portfolio expected return.
    min_weight : float
        Lower bound on each individual asset weight. Default ``0.0``.
    max_weight : float
        Upper bound on each individual asset weight. Default ``1.0``.

    Returns
    -------
    pd.Series
        Optimal weights summing to 1, indexed by ticker.

    Raises
    ------
    RuntimeError
        If the cvxpy problem does not solve to optimality.
    """
    # 1. Convert pandas to numpy (cvxpy works on numpy under the hood)
    mu = expected_returns.to_numpy()
    sigma = cov_matrix.to_numpy()
    n = len(mu)
    tickers = expected_returns.index

    # 2. Declare optimization variable: n-dimensional weight vector
    x = cp.Variable(n)

    # 3. Define objective
    objective = cp.Minimize(cp.quad_form(x, sigma))

    # 4. Define constraints
    constraints = [
        cp.sum(x) == 1,
        mu @ x >= target_return,
        x >= min_weight,
        x <= max_weight,
    ]

    # 5. Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # 6. Check status
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxpy optimizer failed: status={problem.status}")

    logger.info(
        f"cvxpy MVO converged: variance={problem.value:.6f}, "
        f"return={(mu @ x.value):.6f}, status={problem.status}"
    )

    # 7. Wrap into pd.Series
    return pd.Series(x.value, index=tickers, name="weights")
