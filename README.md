# Mean-Variance Portfolio Optimization

Markowitz mean-variance optimization in Python. Two solver backends. Tangency portfolio. Out-of-sample backtest that shows the well-known instability of sample-based MVO.

![Out-of-sample cumulative returns](outputs/backtest_cumulative.png)

## What this project does

Runs the full MVO pipeline on 8 assets (SPY, GOVT, EEMV, CME, BR, CBOE, ICE, ACN) using monthly returns from 2015 to 2024:

- Pulls adjusted-close prices from Tiingo and caches them as parquet
- Computes returns, μ, Σ, σ
- Solves the MVO quadratic program two ways: `scipy.optimize.minimize` with hand-coded Jacobians, and `cvxpy`
- Sweeps the efficient frontier in three regimes: short-selling allowed, long-only, long-only with 5% minimum weight
- Finds the tangency (max-Sharpe) portfolio by grid search and by an exact QP transformation
- Runs a 60/40 train/test backtest comparing the tangency, minimum-variance, and equal-weight (1/N) portfolios

## Key result

Train: 2015–2020. Test: 2021–2024. The 1/N benchmark beat both MVO portfolios:

| Portfolio | Annualized Return | Annualized Volatility | Annualized Sharpe | Max Drawdown |
|---|---|---|---|---|
| Tangency (max-Sharpe) | 1.78% | 8.36% | 0.21 | -17.78% |
| Minimum-Variance | 0.31% | 6.78% | 0.05 | -15.54% |
| **Equal-Weight (1/N)** | **10.53%** | **13.44%** | **0.78** | **-22.01%** |

This matches [DeMiguel, Garlappi, and Uppal (2009)](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901): when estimation error in expected returns is large, sample-based MVO loses to 1/N out of sample.

## Methodology

### The MVO problem

A convex quadratic program:

~~~
min   xᵀ Σ x
s.t.  μᵀ x ≥ R         (return target)
      Σ xᵢ = 1          (full investment)
      ℓ ≤ xᵢ ≤ u        (weight bounds)
~~~

The objective is portfolio variance: `Var(xᵀ r) = xᵀ Σ x`. Σ is positive semidefinite, so the problem is convex and has a unique global optimum.

### Estimators

- **Expected returns (μ):** sample arithmetic mean. Portfolio expected return is a linear combination, so the arithmetic mean is the right input.
- **Covariance matrix (Σ):** sample covariance with Bessel's correction (ddof=1).
- **Frequency:** monthly returns from end-of-month prices.

### Two solver implementations

`scipy.optimize.minimize` with `method="SLSQP"`. Provides analytical Jacobians for the objective (`∇(xᵀΣx) = 2Σx`) and for the linear constraints. Tolerance set to `ftol=1e-10` so SLSQP doesn't stop early.

`cvxpy` with the standard pattern: `cp.Variable`, `cp.quad_form`, `cp.Problem`. cvxpy spots the QP structure and routes to OSQP.

A cross-check test runs both solvers on the same input and asserts the weights agree to 1e-4. The two implementations exercise different solver paths, so agreement is meaningful.

### Tangency portfolio

Two implementations, validated against each other:

1. **Grid sweep** — sample the long-only efficient frontier on a fine grid, pick the point with the highest Sharpe ratio. Discrete and approximate.
2. **QP transformation** — Sharpe is scale-invariant in x. Substituting y = x / (μ̃ᵀx) where μ̃ = μ - rf turns Sharpe maximization into `min yᵀΣy s.t. μ̃ᵀy = 1, y ≥ 0`, a single QP. Recover `x* = y* / sum(y*)`.

The QP version is exact and about 10× faster. The grid version is easier to explain and acts as a sanity check.

### Out-of-sample backtest

60/40 chronological split. No rolling window, no rebalancing — weights are frozen at the end of training and held through test. Three portfolios:

1. **Tangency** — uses both μ̂ and Σ̂
2. **Minimum-variance** — uses only Σ̂
3. **Equal-weight (1/N)** — uses neither

Each portfolio's frozen weights are applied to the test-period returns. Metrics (cumulative return, annualized Sharpe, max drawdown) come from the resulting wealth path.

## Plots

### Efficient frontier across constraint regimes

![Efficient frontier](outputs/efficient_frontier.png)

Short-selling allowed (blue) hits the lowest volatility for any return target. Long-only (orange) sits to the right. Long-only with 5% minimum weight (green) sits further right — the cost of forced diversification. All eight individual assets fall below and to the right of the long-only frontier, which is the diversification benefit shown directly.

### Tangency portfolio and Capital Market Line

![Tangency and CML](outputs/tangency_cml.png)

The Capital Market Line (dashed) goes from the risk-free rate on the y-axis through the tangency portfolio (red marker) and is tangent to the long-only frontier there. Its slope is the tangency portfolio's Sharpe ratio: 0.32 monthly, about 1.12 annualized. Under CAPM, every rational investor holds a mix of this portfolio and the risk-free asset.

### Out-of-sample backtest

![Backtest](outputs/backtest_cumulative.png)

Cumulative wealth from $1 invested at the start of the test window (Jan 2021). Equal-weight (blue) ends at $1.47. Tangency (red) ends at $1.06. Minimum-variance (purple) ends at $1.00. Markowitz instability on real data.

## Reproduction

~~~bash
git clone https://github.com/tangentjet-33/mean-variance-portfolio
cd mean-variance-portfolio

# 1. Create the conda environment
conda env create -f environment.yml
conda activate mvo

# 2. Install the project as an editable package
pip install -e .

# 3. Set up your Tiingo API key
cp .env.example .env
# Edit .env and paste your Tiingo token (free at tiingo.com)

# 4. Run the test suite (31 tests, ~3 seconds)
pytest tests/ -v
~~~

The plots in `outputs/` are checked in. To regenerate them, see the function-level entry points in `src/plots.py`.

## Project structure

~~~
mean-variance-portfolio/
├── src/
│   ├── config.py        # Project-wide constants (tickers, dates, paths)
│   ├── data.py          # Tiingo download + caching + return computation
│   ├── stats.py         # Arithmetic/geometric mean, covariance, volatility
│   ├── optimizer.py     # MVO solvers (scipy + cvxpy)
│   ├── frontier.py      # Efficient frontier sweep
│   ├── sharpe.py        # Sharpe ratio + tangency portfolio (grid + QP)
│   ├── backtest.py      # Train/test split, portfolio returns, OOS metrics
│   └── plots.py         # Matplotlib figures
├── tests/               # 31 unit tests (pytest)
├── outputs/             # Generated plots (PNG, 300 DPI)
├── environment.yml      # Conda environment specification
└── pyproject.toml       # Package config + black/ruff config
~~~

## Limitations and future work

This is plain sample-based MVO, on purpose. The 1/N result above is not a bug — it is the textbook behavior of the sample estimator, and showing it cleanly was the goal. Several extensions would address the instability:

- **Shrinkage estimators for Σ.** Ledoit-Wolf shrinkage replaces the noisy sample covariance with a convex combination of the sample matrix and a structured target. Cuts estimation error.
- **Shrinkage estimators for μ.** James-Stein-type shrinkage pulls extreme sample means toward a common prior. Cuts the over-weighting of assets that got lucky in the sample.
- **Black-Litterman.** Combines investor views with market-implied equilibrium returns. Produces more stable weights than pure sample MVO.
- **Factor models.** Replaces the n×n sample covariance with a low-rank factor representation (e.g., Fama-French), so there are fewer parameters to estimate.
- **Resampled / robust optimization.** Treats μ̂ as uncertain and optimizes over a confidence region rather than a point estimate (Michaud's resampled efficiency).

Each of these has hyperparameters, so adding them would mean moving from train/test to train/validation/test.

Other things out of scope here:

- Rolling-window backtests with periodic rebalancing
- Transaction costs and turnover penalties
- Higher-frequency data (daily or weekly rebalancing)
- Real risk-free rate from FRED instead of `rf=0`

## References

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies*, 22(5), 1915–1953.
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *Journal of Portfolio Management*, 30(4), 110–119.
- Michaud, R. O. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" *Financial Analysts Journal*, 45(1), 31–42. 