"""
Data ingestion: download daily adjusted close prices and compute returns.

Caches results to ``config.PRICES_CACHE_PATH`` as parquet; subsequent calls
read from cache unless ``force_refresh`` is True.
"""

import logging
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src import config

load_dotenv()
logger = logging.getLogger(__name__)


def download_prices(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices from Tiingo for one or more tickers.

    Hits the Tiingo daily prices endpoint once per ticker, with a brief
    delay between requests, and returns a single DataFrame with one column
    per ticker.

    Parameters
    ----------
    tickers : list[str] | None
        Tickers to download. Defaults to ``config.TICKERS``.
    start_date : str | None
        Start date (inclusive) in ``YYYY-MM-DD`` format. Defaults to ``config.START_DATE``.
    end_date : str | None
        End date (inclusive) in ``YYYY-MM-DD`` format. Defaults to ``config.END_DATE``.
    force_refresh : bool
        If True, bypass any cached parquet file and re-download from Tiingo.
        Default False (use cache if available).

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices indexed by date (``datetime64[ns]``),
        with one column per ticker.

    Raises
    ------
    RuntimeError
        If the ``TIINGO_API_KEY`` environment variable is not set, or if any
        ticker request returns a non-200 response.
    """
    cache_path = config.PRICES_CACHE_PATH
    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading prices from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    tickers = tickers or config.TICKERS
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE

    frames: list[pd.DataFrame] = []

    token = os.getenv("TIINGO_API_KEY")
    if token is None:
        raise RuntimeError("TIINGO_API_KEY not set. Add it to your .env file.")

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }

    for ticker in tickers:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Tiingo request failed for {ticker}: {response.status_code} {response.text}"
            )
        data = response.json()
        df = pd.DataFrame(data)[["date", "adjClose"]]
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.set_index("date")
        df.index = df.index.astype("datetime64[ns]")
        df = df.rename(columns={"adjClose": ticker})
        frames.append(df)
        time.sleep(0.5)
        logger.info(f"Downloaded {ticker}: {len(df)} rows")

    prices = pd.concat(frames, axis=1)
    nan_counts = prices.isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values present per ticker:\n{nan_counts[nan_counts > 0]}")

    logger.info(f"Combined prices: {prices.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    logger.info(f"Wrote prices to cache: {cache_path}")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end and compute simple monthly returns.

    Uses pandas' month-end resample convention (``"ME"``) which picks the
    last available trading day of each calendar month, then applies
    ``pct_change`` to compute simple returns. The first month is dropped
    since there is no prior month to compute a return against.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices indexed by date (``datetime64[ns]``), with one column
        per ticker. Typically the output of ``download_prices``.

    Returns
    -------
    pd.DataFrame
        Monthly simple returns indexed by month-end date, with one column
        per ticker. Length is one less than the number of months in the
        input window (first month dropped due to no prior baseline).
    """

    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change().dropna()
    logger.info(f"Computed monthly returns: {returns.shape}")
    return returns
