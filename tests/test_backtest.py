import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_metrics,
    compute_portfolio_returns,
    run_backtest,
    split_train_test,
)


def _toy_returns(n_periods: int = 50, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    """Synthetic returns for testing backtest helpers."""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2020-01-31", periods=n_periods, freq="ME")
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    data = rng.normal(0.005, 0.04, size=(n_periods, n_assets))
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_split_train_test_sizes():
    """A 60/40 split on 100 observations gives 60 train, 40 test."""
    returns = _toy_returns(n_periods=100)
    train, test = split_train_test(returns, train_fraction=0.6)
    assert len(train) == 60
    assert len(test) == 40
    # No overlap and chronological order
    assert train.index[-1] < test.index[0]


def test_split_train_test_invalid_fraction_raises():
    """Fractions outside (0, 1) are rejected."""
    returns = _toy_returns(n_periods=50)
    for bad in [0.0, 1.0, -0.5, 1.5]:
        with pytest.raises(ValueError, match="train_fraction"):
            split_train_test(returns, train_fraction=bad)


def test_compute_portfolio_returns_equal_weight_is_mean():
    """Equal-weight portfolio return equals the row-wise mean of asset returns."""
    returns = _toy_returns(n_periods=10, n_assets=4)
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=returns.columns)
    portfolio = compute_portfolio_returns(weights, returns)
    np.testing.assert_array_almost_equal(portfolio.values, returns.mean(axis=1).values, decimal=12)


def test_compute_portfolio_returns_unknown_ticker_raises():
    """Weights referencing tickers not in returns columns raise."""
    returns = _toy_returns(n_periods=5, n_assets=3)
    weights = pd.Series([0.5, 0.5], index=["NOT_HERE", "ALSO_NOT"])
    with pytest.raises(ValueError, match="unknown tickers"):
        compute_portfolio_returns(weights, returns)


def test_backtest_metrics_zero_returns():
    """A series of zero returns yields zero everything (vol gives nan Sharpe)."""
    portfolio_returns = pd.Series([0.0] * 24)
    metrics = backtest_metrics(portfolio_returns)
    assert metrics["cumulative_return"] == pytest.approx(0.0)
    assert metrics["annualized_return"] == pytest.approx(0.0)
    assert metrics["annualized_volatility"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["final_value"] == pytest.approx(1.0)
    assert np.isnan(metrics["annualized_sharpe"])  # zero vol → nan


def test_backtest_metrics_known_values():
    """Spot-check on a known small series."""
    # 5% per period, 12 periods → 79.6% cumulative, $1 → $1.796
    portfolio_returns = pd.Series([0.05] * 12)
    metrics = backtest_metrics(portfolio_returns, periods_per_year=12)
    expected_cumulative = (1.05) ** 12 - 1
    assert metrics["cumulative_return"] == pytest.approx(expected_cumulative)
    assert metrics["annualized_return"] == pytest.approx(0.05 * 12)
    # Max drawdown is 0 (only positive returns)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_run_backtest_output_shape():
    """run_backtest returns a 3-row DataFrame with the expected metric columns."""
    returns = _toy_returns(n_periods=72, n_assets=4)
    result = run_backtest(returns, train_fraction=0.6)
    assert result.shape == (3, 6)
    assert set(result.index) == {"tangency", "min_variance", "equal_weight"}
    expected_cols = {
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "annualized_sharpe",
        "max_drawdown",
        "final_value",
    }
    assert set(result.columns) == expected_cols
