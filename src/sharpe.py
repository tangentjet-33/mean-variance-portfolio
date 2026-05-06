"""
Sharpe ratio calculation and tangency-portfolio identification.

The tangency portfolio is the maximum-Sharpe-ratio point on the efficient
frontier. Under CAPM assumptions, it is the optimal risky portfolio
regardless of the investor's risk tolerance.
"""

import logging
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd

from src.frontier import efficient_frontier

logger = logging.getLogger(__name__)


def sharpe_ratio(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute the Sharpe ratio of a portfolio.

    Defined as the excess return per unit of risk::

        Sharpe = (E[r_p] - r_f) / σ_p

    Parameters
    ----------
    expected_return : float
        Portfolio expected return.
    volatility : float
        Portfolio standard deviation of returns.
    risk_free_rate : float
        Risk-free rate. Default ``0.0``. Must be in the same frequency
        as ``expected_return`` (e.g., monthly).

    Returns
    -------
    float
        Sharpe ratio. Positive values mean the portfolio outperforms the
        risk-free rate per unit of volatility.
    """
    if volatility == 0.0:
        raise ZeroDivisionError("Volatility cannot be zero.")
    return (expected_return - risk_free_rate) / volatility


def tangency_portfolio_grid(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    n_points: int = 100,
) -> pd.Series:
    """
    Tangency portfolio via grid sweep of the efficient frontier (Method A).

    Computes the efficient frontier on a fine grid of target returns and
    returns the portfolio with the highest Sharpe ratio. Discrete; accuracy
    limited by ``n_points``.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (μ), indexed by ticker.
    cov_matrix : pd.DataFrame
        Covariance matrix (Σ) of asset returns.
    risk_free_rate : float
        Risk-free rate. Default ``0.0``.
    min_weight : float
        Per-asset weight lower bound. Default ``0.0``.
    max_weight : float
        Per-asset weight upper bound. Default ``1.0``.
    n_points : int
        Number of grid points along the frontier. Default ``100``.

    Returns
    -------
    pd.Series
        Optimal tangency-portfolio weights, indexed by ticker. The
        ``.attrs`` dict carries ``sharpe``, ``return``, and ``volatility``.
    """
    # 1. Determine target-return sweep range.
    # Use the range from a small positive number to just below the max single-asset return.
    mu_min = max(expected_returns.min(), risk_free_rate + 1e-6)
    mu_max = (
        expected_returns.max() * 0.99
    )  # leave headroom; max-return target is rarely feasible exactly
    targets = np.linspace(mu_min, mu_max, n_points)

    # 2. Compute the efficient frontier across the grid.
    frontier = efficient_frontier(
        expected_returns,
        cov_matrix,
        targets,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # 3. Compute Sharpe at each frontier point and pick the max.
    sharpes = (frontier["return"] - risk_free_rate) / frontier["volatility"]
    best_target = sharpes.idxmax()
    best_sharpe = sharpes.loc[best_target]

    # 4. Build the result Series with attrs.
    weights = frontier.loc[best_target].drop(["volatility", "return"])
    weights = weights.astype(float)  # ensure no object-dtype shenanigans
    weights.attrs = {
        "sharpe": float(best_sharpe),
        "return": float(frontier.loc[best_target, "return"]),
        "volatility": float(frontier.loc[best_target, "volatility"]),
        "method": "grid",
    }

    logger.info(
        f"Tangency (grid): Sharpe={best_sharpe:.4f}, "
        f"return={weights.attrs['return']:.4f}, "
        f"volatility={weights.attrs['volatility']:.4f}"
    )
    return weights


def tangency_portfolio_qp(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Tangency portfolio via direct QP transformation (Method B).

    Reformulates Sharpe maximization as a single quadratic program by
    exploiting Sharpe's scale-invariance. Specifically, with excess
    returns ``μ̃ = μ - r_f``, define ``y = x / (μ̃.T @ x)`` so the original
    problem reduces to::

        min   y.T @ Σ @ y
        s.t.  μ̃.T @ y == 1
              y_i >= 0  (and other bound constraints, scaled appropriately)

    After solving, recover ``x* = y* / sum(y*)``.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (μ), indexed by ticker.
    cov_matrix : pd.DataFrame
        Covariance matrix (Σ) of asset returns.
    risk_free_rate : float
        Risk-free rate. Default ``0.0``.
    min_weight : float
        Per-asset weight lower bound. Default ``0.0``.
    max_weight : float
        Per-asset weight upper bound. Default ``1.0``.

    Returns
    -------
    pd.Series
        Optimal tangency-portfolio weights, indexed by ticker. The
        ``.attrs`` dict carries ``sharpe``, ``return``, and ``volatility``.

    Raises
    ------
    RuntimeError
        If no portfolio has positive excess return (transformation
        infeasible) or if the QP fails to converge.
    """
    if min_weight != 0.0 or max_weight != 1.0:
        raise NotImplementedError(
            "tangency_portfolio_qp only supports the default long-only bounds "
            "(min_weight=0.0, max_weight=1.0). For non-default bounds, use "
            "tangency_portfolio_grid (Method A) instead."
        )

    # 1. Convert pandas to numpy
    mu = expected_returns.to_numpy()
    sigma = cov_matrix.to_numpy()
    n = len(mu)
    tickers = expected_returns.index

    # 2. Compute excess returns: μ̃ = μ - r_f
    mu_excess = mu - risk_free_rate

    # 3. Sanity check: at least one asset must have positive excess return
    if (mu_excess <= 0).all():
        raise RuntimeError(
            f"No asset has return > risk_free_rate={risk_free_rate}; "
            "tangency portfolio is undefined."
        )

    # 4. Solve the transformed QP for y
    #    min   y.T @ Σ @ y
    #    s.t.  μ̃.T @ y == 1
    #          y >= 0  (long-only-equivalent; bound regimes other than long-only
    #                   don't translate cleanly through the y substitution)
    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, sigma))
    constraints = [
        mu_excess @ y == 1,
        y >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxpy tangency QP failed: status={problem.status}")

    # 5. Recover x from y
    y_star = y.value
    x_star = y_star / y_star.sum()

    # 6. Compute portfolio metrics in x-space
    portfolio_return = float(mu @ x_star)
    portfolio_variance = float(x_star @ sigma @ x_star)
    portfolio_vol = float(np.sqrt(portfolio_variance))
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    # 7. Build result Series with attrs
    weights = pd.Series(x_star, index=tickers, name="weights").astype(float)
    weights.attrs = {
        "sharpe": portfolio_sharpe,
        "return": portfolio_return,
        "volatility": portfolio_vol,
        "method": "qp",
    }

    logger.info(
        f"Tangency (QP): Sharpe={portfolio_sharpe:.4f}, "
        f"return={portfolio_return:.4f}, volatility={portfolio_vol:.4f}"
    )
    return weights


def tangency_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    method: Literal["qp", "grid"] = "qp",
) -> pd.Series:
    """
    Public API for the tangency portfolio.

    Dispatches to either the QP solver (``method="qp"``, default and exact)
    or the grid-sweep approximation (``method="grid"``) for cross-validation.

    Parameters
    ----------
    [...same as the underlying methods...]
    method : {"qp", "grid"}
        Which approach to use. ``"qp"`` is exact; ``"grid"`` is approximate
        but easier to explain and validate.

    Returns
    -------
    pd.Series
        Tangency-portfolio weights with metadata in ``.attrs``.
    """
    if method == "qp":
        return tangency_portfolio_qp(
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    elif method == "grid":
        return tangency_portfolio_grid(
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'qp' or 'grid'.")
