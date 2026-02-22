"""
Data models for portfolio analysis.

Defines core data structures for assets and market datasets (Section 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class MarketAsset:
    """
    Represents a single tradable asset in the portfolio universe.
    
    Used in Section 7.2 (Selection of market and assets).
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
    (General logic of empirical study).
    
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
    
    def set_prices(self, prices: pd.DataFrame) -> None:
        """
        Set historical price data for the dataset.
        
        Validates that columns match asset tickers (Section 7.3).
        
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
        
        Validates that columns match asset tickers (Section 7.4).
        
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
    
    def __str__(self) -> str:
        """String representation of dataset."""
        tickers = ", ".join(self.get_tickers())
        return f"MarketDataSet({len(self.assets)} assets: {tickers})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"MarketDataSet(assets={self.assets}, has_prices={self.prices is not None}, has_returns={self.returns is not None})"