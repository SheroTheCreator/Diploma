"""
Data preprocessing service for portfolio analysis.
"""

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data by handling missing values.
    Uses forward-fill to handle missing data from holidays,
    then backward-fill, then drops remaining invalid rows and columns.
    """
    cleaned = prices.copy()
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna(axis=1, how="any")  # drop columns (assets) with remaining NaN
    return cleaned.dropna(how="all").sort_index()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple periodic returns and drop any columns with NaN."""
    returns = prices.pct_change().dropna()
    returns = returns.dropna(axis=1, how="any")  # drop assets with incomplete return series
    return returns
