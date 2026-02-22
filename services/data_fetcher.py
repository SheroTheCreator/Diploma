"""
Data fetching service for historical market data.
"""

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str], 
    start: str, 
    end: str, 
    interval: str = "1d"
) -> pd.DataFrame:
    """Download historical adjusted closing prices from Yahoo Finance."""
    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {tickers} in period {start} to {end}.")

    # Extract adjusted close prices
    if len(tickers) == 1:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        else:
            prices = data[["Close"]].copy()
        prices.columns = tickers
    else:
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"].copy()
            else:
                prices = data["Close"].copy()
        else:
            prices = data.copy()

    # Ensure column names are strings and sort by date
    prices.columns = [str(col) for col in prices.columns]
    return prices.sort_index()
