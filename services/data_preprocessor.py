"""
Data preprocessing service for portfolio analysis.

Implements data cleaning and transformation (Section 7.3).[file:2]
Handles missing values and computes periodic returns.
"""

from __future__ import annotations

import pandas as pd


class DataPreprocessor:
    """
    Preprocesses raw price data for portfolio analysis.
    
    Implements data preparation steps from Section 7.3 (Data sources and preprocessing).[file:2]
    """

    def clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by handling missing values.
        
        Uses forward-fill to handle missing data from holidays or trading halts,
        then drops any remaining rows with all NaN values (Section 7.3).[file:2]
        
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

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute logarithmic returns from price series.
        
        Log return formula:
        r_t = ln(P_t / P_{t-1})
        
        Log returns are time-additive and more suitable for some statistical models,
        but not used in classical Markowitz framework (Section 2.1).[file:2]
        
        :param prices: Clean price DataFrame
        :return: DataFrame of log returns
        """
        import numpy as np
        
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.dropna()
        
        return log_returns

    def align_data(self, *dataframes: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """
        Align multiple DataFrames to common date range.
        
        Useful when combining data from different sources or assets
        with different trading calendars.
        
        :param dataframes: Variable number of DataFrames to align
        :return: Tuple of aligned DataFrames
        """
        if not dataframes:
            return tuple()
        
        # Find common date range
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)
        
        # Reindex all DataFrames to common dates
        aligned = tuple(df.loc[common_index] for df in dataframes)
        
        return aligned

    def remove_outliers(
        self,
        returns: pd.DataFrame,
        n_std: float = 5.0,
    ) -> pd.DataFrame:
        """
        Remove extreme outliers from return series.
        
        Optional preprocessing step. Note: Section 6.1 discusses that
        financial returns often have fat tails, so aggressive outlier
        removal may distort the true risk characteristics.[file:2]
        
        :param returns: Return DataFrame
        :param n_std: Number of standard deviations for outlier threshold
        :return: Returns with outliers replaced by NaN (can then forward-fill)
        """
        cleaned = returns.copy()
        
        for column in cleaned.columns:
            mean = cleaned[column].mean()
            std = cleaned[column].std()
            
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            # Mark outliers as NaN
            mask = (cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)
            cleaned.loc[mask, column] = pd.NA
        
        return cleaned

    def resample_prices(
        self,
        prices: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Resample price data to different frequency.
        
        Allows analysis at weekly or monthly frequency instead of daily.
        Affects annualization factor (periods_per_year parameter).
        
        :param prices: Daily price DataFrame
        :param frequency: Target frequency ('W' for weekly, 'M' for monthly, etc.)
        :return: Resampled prices (last price in each period)
        """
        resampled = prices.resample(frequency).last()
        return resampled

    def normalize_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prices to start at 100.
        
        Useful for visualizing relative performance when assets
        have very different price levels.
        
        :param prices: Price DataFrame
        :return: Normalized prices (base 100)
        """
        normalized = 100 * prices / prices.iloc[0]
        return normalized
