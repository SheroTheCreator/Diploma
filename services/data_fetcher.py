"""
Data fetching service for historical market data.

Uses yfinance library to download asset prices.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import yfinance as yf


class YahooFinanceDataFetcher:
    """
    Downloads historical price data from Yahoo Finance.
    """

    def download_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical adjusted closing prices for given tickers.
        
        :param tickers: List of ticker symbols (e.g. ['SPY', 'TLT'])
        :param start: Start date in format 'YYYY-MM-DD'
        :param end: End date in format 'YYYY-MM-DD'
        :param interval: Data frequency ('1d' for daily, '1wk' for weekly, etc.)
        :return: DataFrame with dates as index and tickers as columns
        :raises ValueError: If no data downloaded or tickers invalid
        """
        if not tickers:
            raise ValueError("At least one ticker must be provided")

        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                raise ValueError(
                    f"No data downloaded for tickers {tickers} in period {start} to {end}"
                )

            # Extract adjusted close prices
            if len(tickers) == 1:
                # Single ticker: yfinance returns simple DataFrame
                if "Adj Close" in data.columns:
                    prices = data[["Adj Close"]].copy()
                    prices.columns = tickers
                elif "Close" in data.columns:
                    # Fallback for auto_adjust=True case
                    prices = data[["Close"]].copy()
                    prices.columns = tickers
                else:
                    raise ValueError("No price column found in downloaded data")
            else:
                # Multiple tickers: yfinance returns MultiIndex DataFrame
                if isinstance(data.columns, pd.MultiIndex):
                    if "Adj Close" in data.columns.get_level_values(0):
                        prices = data["Adj Close"].copy()
                    elif "Close" in data.columns.get_level_values(0):
                        prices = data["Close"].copy()
                    else:
                        raise ValueError("No price column found in multi-ticker data")
                else:
                    # Single level columns 
                    prices = data.copy()

            # Ensure column names are strings
            prices.columns = [str(col) for col in prices.columns]

            # Sort by date
            prices = prices.sort_index()

            return prices

        except Exception as e:
            raise RuntimeError(
                f"Failed to download data from Yahoo Finance: {str(e)}"
            ) from e
