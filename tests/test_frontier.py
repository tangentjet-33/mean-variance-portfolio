import numpy as np
import pandas as pd

from src.frontier import efficient_frontier


def _toy_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Synthetic 3-asset universe for testing the frontier."""
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


def test_frontier_output_shape():
    """Each feasible target produces one row; columns are vol, return, weights."""
    mu, sigma = _toy_inputs()
    targets = np.linspace(0.005, 0.011, 7)
    f = efficient_frontier(mu, sigma, targets)
    assert f.shape[0] == len(targets)
    assert "volatility" in f.columns
    assert "return" in f.columns
    for ticker in mu.index:
        assert ticker in f.columns


def test_frontier_volatility_monotonic():
    """Volatility should be non-decreasing as target return increases."""
    mu, sigma = _toy_inputs()
    targets = np.linspace(0.005, 0.011, 7)
    f = efficient_frontier(mu, sigma, targets)
    vols = f["volatility"].values
    diffs = np.diff(vols)
    # Allow tiny floating-point negatives; require monotonic up to 1e-8.
    assert (diffs >= -1e-8).all(), f"Volatility not monotonic: diffs={diffs}"


def test_frontier_constraint_tightening_increases_volatility():
    """Adding constraints can only increase (or equal) volatility at each target."""
    mu, sigma = _toy_inputs()
    targets = np.linspace(0.006, 0.011, 6)

    f_short = efficient_frontier(mu, sigma, targets, min_weight=-1.0)
    f_long = efficient_frontier(mu, sigma, targets, min_weight=0.0)

    common = f_short.index.intersection(f_long.index)
    for tgt in common:
        v_short = f_short.loc[tgt, "volatility"]
        v_long = f_long.loc[tgt, "volatility"]
        assert (
            v_long >= v_short - 1e-8
        ), f"Long-only volatility {v_long} should be >= short-allowed {v_short} at target {tgt}"
