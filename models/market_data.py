"""
Data models for portfolio analysis.

Defines core data structures for assets and market datasets (Section 7).[file:2]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class MarketAsset:
    """
    Represents a single tradable asset in the portfolio universe.
    
    Used in Section 7.2 (Selection of market and assets).[file:2]
    """
    ticker: str
    asset_type: Optional[str] = None
    name: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate ticker is not empty."""
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        # Default name to ticker if not provided
        if self.name is None:
            self.name = self.ticker
    
    def __str__(self) -> str:
        """String representation of asset."""
        if self.asset_type:
            return f"{self.ticker} ({self.asset_type})"
        return self.ticker
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"MarketAsset(ticker='{self.ticker}', asset_type='{self.asset_type}', name='{self.name}')"


@dataclass
class MarketDataSet:
    """
    Container for a collection of assets and their historical data.
    
    Implements the investment universe concept from Section 7.1 
    (General logic of empirical study).[file:2]
    
    Stores:
    - List of assets (Section 7.2)
    - Historical prices (Section 7.3)
    - Return series (computed from prices)
    """
    assets: List[MarketAsset]
    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    
    def __post_init__(self) -> None:
        """Validate that at least one asset is provided."""
        if not self.assets:
            raise ValueError("MarketDataSet must contain at least one asset")
    
    def get_tickers(self) -> List[str]:
        """
        Get list of ticker symbols in the dataset.
        
        :return: List of ticker strings
        """
        return [asset.ticker for asset in self.assets]
    
    def get_asset_by_ticker(self, ticker: str) -> Optional[MarketAsset]:
        """
        Retrieve asset object by ticker symbol.
        
        :param ticker: Ticker symbol to search for
        :return: MarketAsset if found, None otherwise
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        return None
    
    def set_prices(self, prices: pd.DataFrame) -> None:
        """
        Set historical price data for the dataset.
        
        Validates that columns match asset tickers (Section 7.3).[file:2]
        
        :param prices: DataFrame with tickers as columns, dates as index
        :raises ValueError: If columns don't match asset tickers
        """
        tickers = self.get_tickers()
        price_tickers = list(prices.columns)
        
        if set(price_tickers) != set(tickers):
            raise ValueError(
                f"Price DataFrame columns {price_tickers} do not match "
                f"dataset tickers {tickers}"
            )
        
        self.prices = prices
    
    def set_returns(self, returns: pd.DataFrame) -> None:
        """
        Set return series for the dataset.
        
        Validates that columns match asset tickers (Section 7.4).[file:2]
        
        :param returns: DataFrame with tickers as columns, dates as index
        :raises ValueError: If columns don't match asset tickers
        """
        tickers = self.get_tickers()
        return_tickers = list(returns.columns)
        
        if set(return_tickers) != set(tickers):
            raise ValueError(
                f"Returns DataFrame columns {return_tickers} do not match "
                f"dataset tickers {tickers}"
            )
        
        self.returns = returns
    
    def has_prices(self) -> bool:
        """Check if price data is loaded."""
        return self.prices is not None and not self.prices.empty
    
    def has_returns(self) -> bool:
        """Check if return data is computed."""
        return self.returns is not None and not self.returns.empty
    
    def get_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the date range covered by the dataset.
        
        :return: Tuple of (start_date, end_date)
        :raises ValueError: If no price or return data loaded
        """
        if self.has_prices():
            return self.prices.index.min(), self.prices.index.max()
        elif self.has_returns():
            return self.returns.index.min(), self.returns.index.max()
        else:
            raise ValueError("No price or return data loaded in dataset")
    
    def n_assets(self) -> int:
        """Get number of assets in the dataset."""
        return len(self.assets)
    
    def n_observations(self) -> int:
        """
        Get number of time periods in the dataset.
        
        :return: Number of rows in price/return data
        :raises ValueError: If no data loaded
        """
        if self.has_returns():
            return len(self.returns)
        elif self.has_prices():
            return len(self.prices)
        else:
            raise ValueError("No data loaded in dataset")
    
    def summary(self) -> dict:
        """
        Generate summary statistics about the dataset.
        
        Useful for validation and reporting (Section 7.4).[file:2]
        
        :return: Dictionary with dataset characteristics
        """
        summary_dict = {
            "n_assets": self.n_assets(),
            "tickers": self.get_tickers(),
            "has_prices": self.has_prices(),
            "has_returns": self.has_returns(),
        }
        
        if self.has_prices() or self.has_returns():
            start_date, end_date = self.get_date_range()
            summary_dict.update({
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "n_observations": self.n_observations(),
            })
        
        return summary_dict
    
    def __str__(self) -> str:
        """String representation of dataset."""
        tickers = ", ".join(self.get_tickers())
        return f"MarketDataSet({self.n_assets()} assets: {tickers})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"MarketDataSet(assets={self.assets}, has_prices={self.has_prices()}, has_returns={self.has_returns()})"
