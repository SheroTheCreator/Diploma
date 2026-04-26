"""
Data preprocessing service for portfolio analysis.
"""

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data by handling missing values.
    Uses forward-fill to handle missing data from holidays, 
    then backward-fill, then drops remaining invalid rows.
    """
    cleaned = prices.copy()
    cleaned = cleaned.ffill().bfill()
    return cleaned.dropna(how="all").sort_index()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple periodic returns."""
    return prices.pct_change().dropna()
