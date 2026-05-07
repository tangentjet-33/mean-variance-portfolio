"""
Out-of-sample backtest for mean-variance optimization.

Splits historical returns into training and test windows, computes optimal
portfolios on training data only, then evaluates their realized performance
on the held-out test period. Compares the MVO portfolios against an equal-
weight (1/N) benchmark to demonstrate the well-known out-of-sample
instability of sample-based MVO (DeMiguel, Garlappi, and Uppal, 2009).
"""

import logging

import numpy as np
import pandas as pd

from src.optimizer import solve_mvo_cvxpy
from src.sharpe import tangency_portfolio
from src.stats import arithmetic_mean, covariance_matrix

logger = logging.getLogger(__name__)


def split_train_test(
    returns: pd.DataFrame,
    train_fraction: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a returns DataFrame into chronological train and test sets.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns indexed chronologically by date.
    train_fraction : float
        Fraction of observations used for training. Default ``0.6``
        (60%/40% split).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train, test)`` — non-overlapping, chronologically ordered.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")

    n = len(returns)
    n_train = int(n * train_fraction)
    train = returns.iloc[:n_train]
    test = returns.iloc[n_train:]

    logger.info(
        f"Split {n} observations: train={len(train)} ({train.index[0].date()} to "
        f"{train.index[-1].date()}), test={len(test)} ({test.index[0].date()} to "
        f"{test.index[-1].date()})"
    )
    return train, test


def compute_portfolio_returns(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute realized portfolio returns given fixed weights and asset returns.

    Assumes weights are held constant over the period (no rebalancing). For
    each period, the portfolio return is the weighted sum of asset returns.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights, indexed by ticker. Must sum to 1.
    returns : pd.DataFrame
        Asset returns indexed by date, with one column per ticker. Tickers
        must match those in ``weights``.

    Returns
    -------
    pd.Series
        Realized portfolio returns, indexed by date.
    """
    # Verify weights and returns have matching tickers
    missing = set(weights.index) - set(returns.columns)
    if missing:
        raise ValueError(f"Weights reference unknown tickers: {missing}")

    # Align weights to the column order of returns and compute weighted sum
    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    portfolio_returns = (returns * aligned_weights).sum(axis=1)
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def backtest_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """
    Compute summary statistics for a portfolio return series.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Realized portfolio returns indexed by date.
    risk_free_rate : float
        Per-period risk-free rate (in the same frequency as the returns).
        Default ``0.0``.
    periods_per_year : int
        Number of return periods per year, for annualization. Default ``12``
        for monthly returns.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``cumulative_return``: total compound return over the period
        - ``annualized_return``: per-year arithmetic mean return
        - ``annualized_volatility``: per-year standard deviation
        - ``annualized_sharpe``: annualized Sharpe ratio
        - ``max_drawdown``: worst peak-to-trough decline (negative number)
        - ``final_value``: final value of $1 invested at the start
    """
    n = len(portfolio_returns)
    if n == 0:
        raise ValueError("portfolio_returns is empty")

    # 1. Cumulative return: (1+r1)(1+r2)...(1+rn) - 1
    cumulative_return = float((1 + portfolio_returns).prod() - 1)

    # 2. Final value of $1
    final_value = 1.0 + cumulative_return

    # 3. Annualized arithmetic mean return
    mean_return = portfolio_returns.mean()
    annualized_return = float(mean_return * periods_per_year)

    # 4. Annualized volatility
    period_vol = portfolio_returns.std()
    annualized_volatility = float(period_vol * np.sqrt(periods_per_year))

    # 5. Annualized Sharpe ratio
    excess = mean_return - risk_free_rate
    annualized_sharpe = (
        float(excess / period_vol * np.sqrt(periods_per_year)) if period_vol > 0 else float("nan")
    )

    # 6. Max drawdown
    cumulative_value = (1 + portfolio_returns).cumprod()
    running_max = cumulative_value.cummax()
    drawdowns = (cumulative_value - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


def run_backtest(
    returns: pd.DataFrame,
    train_fraction: float = 0.6,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Run the end-to-end out-of-sample backtest comparing MVO to 1/N benchmark.

    Splits the input returns into train/test windows. On the training window,
    estimates μ and Σ, then computes three portfolios:

    1. Tangency (max-Sharpe) portfolio
    2. Global minimum-variance portfolio
    3. Equal-weight (1/N) benchmark

    The frozen weights are applied to the test window and evaluated using
    standard metrics: annualized return, volatility, Sharpe, max drawdown.

    Parameters
    ----------
    returns : pd.DataFrame
        Full historical returns, indexed by date, with one column per ticker.
    train_fraction : float
        Fraction of observations used for training. Default ``0.6``.
    risk_free_rate : float
        Per-period risk-free rate. Default ``0.0``.

    Returns
    -------
    pd.DataFrame
        Comparison table indexed by portfolio name (``"tangency"``,
        ``"min_variance"``, ``"equal_weight"``), with columns for each
        metric in ``backtest_metrics``.
    """
    # 1. Split into train/test
    train, test = split_train_test(returns, train_fraction=train_fraction)

    # 2. Compute training-set statistics
    mu_train = arithmetic_mean(train)
    sigma_train = covariance_matrix(train)
    n_assets = len(mu_train)

    # 3. Compute the three portfolios using ONLY train data
    weights_tangency = tangency_portfolio(
        mu_train, sigma_train, risk_free_rate=risk_free_rate, method="qp"
    )
    weights_minvar = solve_mvo_cvxpy(mu_train, sigma_train, target_return=float(mu_train.min()))
    weights_equal = pd.Series(np.ones(n_assets) / n_assets, index=mu_train.index, name="weights")

    # 4. Apply each set of frozen weights to the test set
    portfolios = {
        "tangency": weights_tangency,
        "min_variance": weights_minvar,
        "equal_weight": weights_equal,
    }

    rows = {}
    for name, weights in portfolios.items():
        portfolio_returns = compute_portfolio_returns(weights, test)
        metrics = backtest_metrics(portfolio_returns, risk_free_rate=risk_free_rate)
        rows[name] = metrics

    result = pd.DataFrame(rows).T
    logger.info(f"Backtest complete; results:\n{result.round(4)}")
    return result
