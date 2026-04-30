"""Project-wide constants for the mean-variance portfolio project."""

from pathlib import Path

TICKERS: list[str] = ["SPY", "GOVT", "EEMV", "CME", "BR", "CBOE", "ICE", "ACN"]
START_DATE: str = "2014-12-01"
END_DATE: str = "2024-12-31"

PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "raw"
PRICES_CACHE_PATH: Path = CACHE_DIR / "prices_daily.parquet"
