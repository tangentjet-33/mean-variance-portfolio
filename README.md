# Mean-Variance Portfolio Optimization

Python implementation of Markowitz mean-variance optimization across 8 assets (SPY, GOVT, EEMV, CME, BR, CBOE, ICE, ACN) using monthly returns from 2015-01 through 2024-12.

## What this project does

- Computes arithmetic and geometric expected returns, covariance matrix, and asset volatilities.
- Solves the mean-variance quadratic program under three constraint regimes: short-selling allowed, long-only, and long-only with a minimum-weight floor.
- Computes the tangency (maximum-Sharpe) portfolio and plots the Capital Market Line.
- Runs an out-of-sample backtest comparing the optimized portfolio against an equal-weight benchmark.

## Status

In progress. See `src/` for current modules.

## How to run

```bash
conda env create -f environment.yml
conda activate mvo
python -m src.main
```

## Author

Ilya Sharif — MEng, University of Toronto (Data Analytics & Machine Learning, 2025).
