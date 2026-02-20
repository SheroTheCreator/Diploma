"""
Data fetching service for historical market data.

Implements data collection from Yahoo Finance (Section 5.1, 7.3).[file:2]
Uses yfinance library as specified in the methodology.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf


class YahooFinanceDataFetcher:
    """
    Downloads historical price data from Yahoo Finance.
    
    Implements data collection step from Section 7.1 (General logic of empirical study)
    using yfinance library (Section 5.1).[file:2]
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
        
        Uses adjusted close prices to reflect total return including dividends
        and stock splits (Section 7.3).[file:2]
        
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
            # Download data using yfinance
            # auto_adjust=False to get explicit 'Adj Close' column
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
                    # Single level columns (shouldn't happen with multiple tickers, but handle it)
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

    def get_latest_price(self, ticker: str) -> float:
        """
        Get the most recent closing price for a ticker.
        
        Useful for current portfolio valuation (not used in historical analysis).
        
        :param ticker: Ticker symbol
        :return: Latest adjusted close price
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                raise ValueError(f"No recent data available for {ticker}")
            
            return float(hist["Close"].iloc[-1])
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to get latest price for {ticker}: {str(e)}"
            ) from e

    def get_ticker_info(self, ticker: str) -> dict:
        """
        Get metadata about a ticker (name, sector, etc.).
        
        Optional: Can be used to display asset information in reports.
        
        :param ticker: Ticker symbol
        :return: Dictionary with ticker metadata
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "USD"),
            }
        
        except Exception as e:
            # Return minimal info if lookup fails
            return {
                "ticker": ticker,
                "name": ticker,
                "sector": "Unknown",
                "industry": "Unknown",
                "currency": "USD",
            }
