import pandas as pd
import pytest

from src.sharpe import (
    sharpe_ratio,
    tangency_portfolio,
    tangency_portfolio_grid,
    tangency_portfolio_qp,
)


def _toy_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Synthetic 3-asset universe where the tangency portfolio is well-defined."""
    tickers = ["A", "B", "C"]
    mu = pd.Series([0.01, 0.005, 0.012], index=tickers)
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


def test_sharpe_ratio_basic():
    """Sharpe of a 1% return / 5% vol portfolio with rf=0 is 0.2."""
    assert sharpe_ratio(0.01, 0.05, 0.0) == pytest.approx(0.2)


def test_sharpe_ratio_with_risk_free():
    """Excess return is subtracted before dividing by vol."""
    assert sharpe_ratio(0.01, 0.05, 0.005) == pytest.approx(0.1)


def test_sharpe_ratio_zero_volatility_raises():
    """Volatility of zero is undefined."""
    with pytest.raises(ZeroDivisionError):
        sharpe_ratio(0.01, 0.0, 0.0)


def test_tangency_grid_basic_constraints():
    """Grid tangency: weights sum to 1, non-negative, positive Sharpe."""
    mu, sigma = _toy_inputs()
    w = tangency_portfolio_grid(mu, sigma, risk_free_rate=0.0)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()
    assert w.attrs["sharpe"] > 0


def test_tangency_qp_basic_constraints():
    """QP tangency: same invariants as grid."""
    mu, sigma = _toy_inputs()
    w = tangency_portfolio_qp(mu, sigma, risk_free_rate=0.0)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()
    assert w.attrs["sharpe"] > 0


def test_tangency_grid_and_qp_agree():
    """Grid and QP methods should produce nearly identical Sharpe ratios."""
    mu, sigma = _toy_inputs()
    w_grid = tangency_portfolio_grid(mu, sigma, risk_free_rate=0.0)
    w_qp = tangency_portfolio_qp(mu, sigma, risk_free_rate=0.0)
    # QP is exact; grid is discrete. QP Sharpe must be >= grid Sharpe.
    assert w_qp.attrs["sharpe"] >= w_grid.attrs["sharpe"] - 1e-6
    # Difference should be tiny.
    assert abs(w_qp.attrs["sharpe"] - w_grid.attrs["sharpe"]) < 1e-3


def test_tangency_qp_no_positive_excess_return_raises():
    """If risk-free rate exceeds all asset returns, tangency is undefined."""
    mu, sigma = _toy_inputs()
    # Risk-free rate higher than any asset's return
    with pytest.raises(RuntimeError, match="No asset has return"):
        tangency_portfolio_qp(mu, sigma, risk_free_rate=0.10)


def test_tangency_dispatcher_routes_correctly():
    """Public API routes to the right implementation based on `method`."""
    mu, sigma = _toy_inputs()
    w_qp = tangency_portfolio(mu, sigma, method="qp")
    w_grid = tangency_portfolio(mu, sigma, method="grid")
    assert w_qp.attrs["method"] == "qp"
    assert w_grid.attrs["method"] == "grid"
