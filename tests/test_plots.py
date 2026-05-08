import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.plots import (
    plot_backtest_cumulative_returns,
    plot_efficient_frontier,
    plot_tangency_with_cml,
)


def _toy_frontier() -> pd.DataFrame:
    """Synthetic frontier-like DataFrame for plot tests."""
    return pd.DataFrame(
        {
            "volatility": [0.02, 0.03, 0.04, 0.05],
            "return": [0.005, 0.008, 0.011, 0.014],
            "A": [0.5, 0.4, 0.3, 0.2],
            "B": [0.5, 0.6, 0.7, 0.8],
        },
        index=[0.005, 0.008, 0.011, 0.014],
    )


def _toy_returns() -> pd.DataFrame:
    """Synthetic monthly returns DataFrame for backtest plot test."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        rng.normal(0.005, 0.04, size=(24, 2)),
        index=dates,
        columns=["A", "B"],
    )


def test_plot_efficient_frontier_creates_file():
    """Function runs without error and saves a non-empty PNG."""
    f = _toy_frontier()
    mu = pd.Series([0.005, 0.008], index=["A", "B"])
    vols = pd.Series([0.02, 0.03], index=["A", "B"])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "frontier.png"
        plot_efficient_frontier(f, f, f, mu, vols, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_tangency_with_cml_creates_file():
    """Tangency + CML plot runs and saves a file."""
    f = _toy_frontier()
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    weights.attrs = {"sharpe": 0.5, "return": 0.011, "volatility": 0.022}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "tangency.png"
        plot_tangency_with_cml(f, weights, risk_free_rate=0.0, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_backtest_creates_file():
    """Backtest cumulative-returns plot runs and saves a file."""
    returns = _toy_returns()
    weights_dict = {
        "tangency": pd.Series([0.6, 0.4], index=["A", "B"]),
        "equal_weight": pd.Series([0.5, 0.5], index=["A", "B"]),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "backtest.png"
        plot_backtest_cumulative_returns(returns, weights_dict, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
