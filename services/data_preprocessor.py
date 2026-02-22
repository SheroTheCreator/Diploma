"""
Data preprocessing service for portfolio analysis.

Implements data cleaning and transformation (Section 7.3).
Handles missing values and computes periodic returns.
"""

from __future__ import annotations

import pandas as pd


class DataPreprocessor:
    """
    Preprocesses raw price data for portfolio analysis.
    
    Implements data preparation steps from Section 7.3 (Data sources and preprocessing).
    """

    def clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by handling missing values.
        
        Uses forward-fill to handle missing data from holidays or trading halts,
        then drops any remaining rows with all NaN values (Section 7.3).
        
        :param prices: Raw price DataFrame from data fetcher
        :return: Cleaned price DataFrame
        """
        cleaned = prices.copy()
        
        # Forward-fill missing values (carry last known price forward)
        cleaned = cleaned.fillna(method="ffill")
        
        # Backward-fill any remaining leading NaNs
        cleaned = cleaned.fillna(method="bfill")
        
        # Drop rows where all values are still NaN
        cleaned = cleaned.dropna(how="all")
        
        # Sort by date to ensure chronological order
        cleaned = cleaned.sort_index()
        
        return cleaned

    def compute_simple_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple periodic returns from price series.
        
        Simple return formula (used in Section 7.4):
        r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
        
        This is the standard approach for portfolio analysis as it preserves
        the property that portfolio returns are weighted averages of asset returns.
        
        :param prices: Clean price DataFrame
        :return: DataFrame of periodic returns
        """
        returns = prices.pct_change()
        
        # Drop first row (NaN from pct_change)
        returns = returns.dropna()
        
        return returns