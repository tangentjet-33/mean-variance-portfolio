"""
Plotting utilities for mean-variance optimization results.

Produces publication-ready PNG figures for:
1. The efficient frontier across constraint regimes
2. The tangency portfolio and Capital Market Line
3. Out-of-sample backtest cumulative returns
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Consistent colors across all plots
COLORS = {
    "short_ok": "#1f77b4",  # blue
    "long_only": "#ff7f0e",  # orange
    "min_5pct": "#2ca02c",  # green
    "tangency": "#d62728",  # red
    "min_variance": "#9467bd",  # purple
    "equal_weight": "#1f77b4",  # blue (reused — context distinguishes)
    "single_assets": "#7f7f7f",  # grey
}


def plot_efficient_frontier(
    frontier_short_ok: pd.DataFrame,
    frontier_long_only: pd.DataFrame,
    frontier_min_5pct: pd.DataFrame,
    expected_returns: pd.Series,
    volatilities: pd.Series,
    save_path: Path | None = None,
) -> None:
    """
    Plot the efficient frontier for three constraint regimes.

    Overlays the short-selling-allowed, long-only, and long-only-with-min-5%-
    weight frontiers on a single chart, with single-asset positions shown as
    scatter points for reference.

    Parameters
    ----------
    frontier_short_ok : pd.DataFrame
        Frontier output (from ``efficient_frontier``) with shorting allowed.
    frontier_long_only : pd.DataFrame
        Frontier output for the long-only regime.
    frontier_min_5pct : pd.DataFrame
        Frontier output for the long-only-with-5%-minimum-weight regime.
    expected_returns : pd.Series
        Per-asset expected returns, indexed by ticker. Used for scatter.
    volatilities : pd.Series
        Per-asset volatilities, indexed by ticker. Used for scatter.
    save_path : Path | None
        Where to save the PNG. Defaults to ``config.FRONTIER_PLOT_PATH``.
    """
    save_path = save_path or config.FRONTIER_PLOT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the three frontier curves
    ax.plot(
        frontier_short_ok["volatility"],
        frontier_short_ok["return"],
        color=COLORS["short_ok"],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Short-selling allowed",
    )
    ax.plot(
        frontier_long_only["volatility"],
        frontier_long_only["return"],
        color=COLORS["long_only"],
        linewidth=2,
        marker="s",
        markersize=4,
        label="Long-only",
    )
    ax.plot(
        frontier_min_5pct["volatility"],
        frontier_min_5pct["return"],
        color=COLORS["min_5pct"],
        linewidth=2,
        marker="^",
        markersize=4,
        label="Long-only, min 5% weight",
    )

    # Scatter individual assets
    ax.scatter(
        volatilities.values,
        expected_returns.values,
        color=COLORS["single_assets"],
        s=80,
        zorder=5,
        edgecolor="black",
        linewidth=0.5,
        label="Individual assets",
    )
    # Annotate each asset with its ticker
    for ticker in expected_returns.index:
        ax.annotate(
            ticker,
            xy=(volatilities[ticker], expected_returns[ticker]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )

    ax.set_xlabel("Volatility (monthly)", fontsize=12)
    ax.set_ylabel("Expected return (monthly)", fontsize=12)
    ax.set_title("Efficient Frontier — Three Constraint Regimes", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved efficient frontier plot: {save_path}")


def plot_tangency_with_cml(
    frontier: pd.DataFrame,
    tangency_weights: pd.Series,
    risk_free_rate: float = 0.0,
    save_path: Path | None = None,
) -> None:
    """
    Plot the long-only efficient frontier with the tangency portfolio and
    Capital Market Line.

    Highlights the tangency point and draws the CML from the risk-free rate
    on the y-axis through the tangency point, extending into the upper-right.

    Parameters
    ----------
    frontier : pd.DataFrame
        Frontier output for the relevant constraint regime (typically long-only).
    tangency_weights : pd.Series
        Weights of the tangency portfolio. Must have ``.attrs`` with keys
        ``"sharpe"``, ``"return"``, and ``"volatility"``.
    risk_free_rate : float
        Per-period risk-free rate, on the same scale as the frontier returns.
        Default ``0.0``.
    save_path : Path | None
        Where to save the PNG. Defaults to ``config.TANGENCY_PLOT_PATH``.
    """
    save_path = save_path or config.TANGENCY_PLOT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull tangency stats from .attrs
    t_return = tangency_weights.attrs["return"]
    t_vol = tangency_weights.attrs["volatility"]
    t_sharpe = tangency_weights.attrs["sharpe"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the efficient frontier
    ax.plot(
        frontier["volatility"],
        frontier["return"],
        color=COLORS["long_only"],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Efficient frontier (long-only)",
    )

    # Plot the Capital Market Line: from (0, r_f) through (t_vol, t_return), extended.
    # slope = (t_return - r_f) / t_vol = Sharpe
    # Extend out to ~1.2x the tangency volatility for visual room
    cml_x = np.array([0.0, t_vol * 1.2])
    cml_y = risk_free_rate + t_sharpe * cml_x
    ax.plot(
        cml_x,
        cml_y,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"Capital Market Line (Sharpe = {t_sharpe:.3f})",
        zorder=3,
    )

    # Highlight the tangency portfolio
    ax.scatter(
        [t_vol],
        [t_return],
        color=COLORS["tangency"],
        s=200,
        zorder=10,
        edgecolor="black",
        linewidth=1.5,
        marker="p",
        label=f"Tangency portfolio (return={t_return:.4f}, vol={t_vol:.4f})",
    )

    # Risk-free point on the y-axis
    ax.scatter(
        [0.0],
        [risk_free_rate],
        color="black",
        s=80,
        zorder=10,
        marker="o",
        label=f"Risk-free rate ({risk_free_rate:.4f})",
    )

    ax.set_xlabel("Volatility (monthly)", fontsize=12)
    ax.set_ylabel("Expected return (monthly)", fontsize=12)
    ax.set_title("Tangency Portfolio and Capital Market Line", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)

    # Make sure (0, 0) is visible — the CML originates from the y-axis
    ax.set_xlim(left=-0.001)
    ax.set_ylim(bottom=min(0.0, risk_free_rate) - 0.001)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved tangency/CML plot: {save_path}")


def plot_backtest_cumulative_returns(
    test_returns: pd.DataFrame,
    weights_dict: dict[str, pd.Series],
    save_path: Path | None = None,
) -> None:
    """
    Plot cumulative wealth curves for multiple portfolios over the test period.

    Computes ``(1 + r).cumprod()`` for each portfolio's realized returns and
    plots them on a shared time axis, starting at $1.

    Parameters
    ----------
    test_returns : pd.DataFrame
        Out-of-sample asset returns indexed by date.
    weights_dict : dict[str, pd.Series]
        Mapping from portfolio name (e.g., ``"tangency"``) to weights Series.
        Each portfolio's frozen weights are applied to ``test_returns``.
    save_path : Path | None
        Where to save the PNG. Defaults to ``config.BACKTEST_PLOT_PATH``.
    """
    save_path = save_path or config.BACKTEST_PLOT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map portfolio name → color (fall back to grey for unknown names)
    name_to_color = {
        "tangency": COLORS["tangency"],
        "min_variance": COLORS["min_variance"],
        "equal_weight": COLORS["equal_weight"],
    }

    for name, weights in weights_dict.items():
        # Compute portfolio returns from frozen weights and asset returns
        aligned = weights.reindex(test_returns.columns).fillna(0.0)
        portfolio_returns = (test_returns * aligned).sum(axis=1)
        cumulative_value = (1 + portfolio_returns).cumprod()
        # Prepend a 1.0 starting point so the curve begins at $1 on day 0
        starting_index = test_returns.index[0] - pd.tseries.offsets.MonthEnd(1)
        cumulative_with_start = pd.concat(
            [
                pd.Series([1.0], index=[starting_index]),
                cumulative_value,
            ]
        )

        ax.plot(
            cumulative_with_start.index,
            cumulative_with_start.values,
            color=name_to_color.get(name, "grey"),
            linewidth=2,
            label=name.replace("_", " ").title(),
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio value (start = $1)", fontsize=12)
    ax.set_title("Out-of-Sample Cumulative Returns", fontsize=14)
    ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved backtest cumulative returns plot: {save_path}")
