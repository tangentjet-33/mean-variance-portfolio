"""
Efficient frontier construction for mean-variance optimization.

Sweeps the MVO optimizer across a range of target returns to trace the
Pareto-optimal frontier in (volatility, return) space.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.optimizer import solve_mvo_cvxpy, solve_mvo_scipy

logger = logging.getLogger(__name__)


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_returns: np.ndarray | list[float],
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    solver: Literal["scipy", "cvxpy"] = "cvxpy",
) -> pd.DataFrame:
    """
    Compute the efficient frontier by solving MVO at each target return.

    For each target return ``R`` in ``target_returns``, solves the MVO
    problem and records the resulting portfolio's volatility, achieved
    return, and weights. Targets that are infeasible under the bound
    constraints are skipped with a warning.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return per asset (μ), indexed by ticker.
    cov_matrix : pd.DataFrame
        Covariance matrix (Σ) of asset returns, indexed by ticker on both
        axes.
    target_returns : np.ndarray or list[float]
        Sequence of target returns to sweep across the frontier.
    min_weight : float
        Per-asset weight lower bound. ``0.0`` for long-only, negative for
        short-selling-allowed, positive for minimum-allocation enforcement.
    max_weight : float
        Per-asset weight upper bound. Default ``1.0``.
    solver : {"scipy", "cvxpy"}
        Which optimizer backend to use. Default ``"cvxpy"`` for speed and
        numerical stability.

    Returns
    -------
    pd.DataFrame
        One row per feasible target return, indexed by target. Columns:
        ``volatility``, ``return``, and one column per asset for weights.

    Notes
    -----
    Targets below the global minimum-variance portfolio's return produce
    a flat segment at the base of the frontier (the constraint is slack).
    Targets above the maximum single-asset return are infeasible under
    long-only constraints and are skipped.
    """
    # 1. Pick solver
    solvers = {
        "scipy": solve_mvo_scipy,
        "cvxpy": solve_mvo_cvxpy,
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver!r}. Use 'scipy' or 'cvxpy'.")
    solve = solvers[solver]

    # 2. Pre-compute pandas → numpy for the post-solve return/vol calculations
    sigma = cov_matrix.to_numpy()
    mu = expected_returns.to_numpy()

    # 3. Sweep target returns
    rows = []
    skipped = []
    for target in target_returns:
        try:
            weights = solve(
                expected_returns,
                cov_matrix,
                target_return=target,
                min_weight=min_weight,
                max_weight=max_weight,
            )
        except RuntimeError as exc:
            logger.warning(f"Target {target:.4f} infeasible, skipping ({exc})")
            skipped.append(float(target))
            continue

        w = weights.to_numpy()
        variance = w @ sigma @ w
        volatility = np.sqrt(variance)
        achieved_return = float(mu @ w)

        row = {"volatility": volatility, "return": achieved_return}
        # Add one column per ticker for the weights
        row.update(weights.to_dict())
        rows.append((float(target), row))

    if not rows:
        raise RuntimeError("No feasible target returns; all skipped.")

    # 4. Build result DataFrame
    targets = [r[0] for r in rows]
    data = [r[1] for r in rows]
    result = pd.DataFrame(data, index=pd.Index(targets, name="target_return"))

    logger.info(
        f"Frontier built: {len(result)} feasible points, {len(skipped)} skipped "
        f"(solver={solver}, min_weight={min_weight}, max_weight={max_weight})"
    )
    return result
