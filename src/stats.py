"""Return statistics: means, covariance, standard deviation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def arithmetic_mean(returns: pd.DataFrame) -> pd.Series:
    """
    Compute the arithmetic mean return per asset.

    Defined as the simple average across the time dimension::

        r̄_i = (1/T) * Σ r_it

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns indexed by date, with one column per asset.

    Returns
    -------
    pd.Series
        Arithmetic mean return per asset, indexed by ticker.

    Notes
    -----
    Used as the input ``μ`` to the mean-variance optimizer because portfolio
    return aggregates linearly across assets only for arithmetic returns.
    """

    return returns.mean(axis=0)


def geometric_mean(returns: pd.DataFrame) -> pd.Series:
    """
    Compute the geometric (compounded) mean return per asset.

    Defined as the T-th root of the cumulative compound return::

        μ_i = (Π (1 + r_it))^(1/T) - 1

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns indexed by date, with one column per asset.

    Returns
    -------
    pd.Series
        Geometric mean return per asset, indexed by ticker.

    Notes
    -----
    The geometric mean reflects actual compounded wealth growth and is always
    less than or equal to the arithmetic mean (Jensen's inequality), with the
    gap approximately ``σ²/2`` (the volatility drag). Reported alongside the
    arithmetic mean for descriptive purposes; not used in the MVO objective.
    """

    return (1 + returns).prod() ** (1 / len(returns)) - 1


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sample covariance matrix of asset returns.

    Defined element-wise as the sample covariance with Bessel's correction::

        σ_ij = (1/(T-1)) * Σ (r_it - r̄_i)(r_jt - r̄_j)

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns indexed by date, with one column per asset.

    Returns
    -------
    pd.DataFrame
        Square covariance matrix of shape ``(n_assets, n_assets)``, with
        tickers on both axes.

    Notes
    -----
    Uses the sample covariance estimator (``ddof=1``, divide by ``T-1``)
    because the asset means are estimated from the same sample. This is
    pandas' default and the standard convention in academic finance.
    The sample covariance matrix is the noisy ``Σ`` consumed by the
    mean-variance optimizer; in practice, shrinkage estimators (e.g.,
    Ledoit-Wolf) are often preferred for stability.
    """

    return returns.cov()


def volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Compute the per-asset standard deviation of returns.

    Equivalent to the square root of the diagonal of the covariance matrix::

        σ_i = sqrt(σ_ii)

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns indexed by date, with one column per asset.

    Returns
    -------
    pd.Series
        Per-asset standard deviation, indexed by ticker.

    Notes
    -----
    Uses the sample standard deviation (``ddof=1``), pandas' default. Reports
    volatility in the same frequency as the input returns (monthly here);
    callers wishing to annualize should multiply by ``sqrt(12)``.
    """

    return returns.std(axis=0)
