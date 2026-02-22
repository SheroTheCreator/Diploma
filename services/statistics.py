"""
Risk and return statistics calculator.

Implements statistical measures such as expected returns, 
volatility, covariance, and correlation matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskReturnCalculator:
    """
    Computes risk and return statistics for portfolio analysis.
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        """
        Initialize calculator with annualization parameter.
        
        :param periods_per_year: Number of trading periods per year
        """
        self.periods_per_year = periods_per_year

    def calculate_annualized_return(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized expected returns.
        
        :param returns: DataFrame of periodic returns
        :return: Series of annualized expected returns per asset
        """
        mean_periodic = returns.mean()
        annualized = mean_periodic * self.periods_per_year
        return annualized

    def calculate_annualized_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized volatility (standard deviation).
        
        :param returns: DataFrame of periodic returns
        :return: Series of annualized standard deviations per asset
        """
        std_periodic = returns.std()
        annualized = std_periodic * np.sqrt(self.periods_per_year)
        return annualized

    def calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix of returns.
        
        :param returns: DataFrame of periodic returns
        :return: Covariance matrix
        """
        return returns.cov()

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.
        
        :param returns: DataFrame of periodic returns
        :return: Correlation matrix with values in [-1, 1]
        """
        return returns.corr()
