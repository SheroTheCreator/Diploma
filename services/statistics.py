"""
Risk and return statistics calculator.

Implements statistical measures from Sections 2.2-2.3 and 4.1 of the thesis:
- Expected returns (Formula 1)
- Portfolio variance (Formulas 2, 4)
- Covariance matrix (Formula 3)
- Standard deviation (Formula 5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskReturnCalculator:
    """
    Computes risk and return statistics for portfolio analysis.
    
    Based on Markowitz mean-variance framework (Section 2).
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        """
        Initialize calculator with annualization parameter.
        
        :param periods_per_year: Number of trading periods per year
                                 (252 for daily data, Section 5.2)
        """
        self.periods_per_year = periods_per_year

    def calculate_annualized_return(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized expected returns (Formula 1, Section 2.2).
        
        E[R_p] = Σ w_i * E[R_i]
        
        For individual assets (w_i = 1 for asset i, 0 for others):
        E[R_annual] = mean(r_daily) × periods_per_year
        
        :param returns: DataFrame of periodic (e.g. daily) returns
        :return: Series of annualized expected returns per asset
        """
        mean_periodic = returns.mean()
        annualized = mean_periodic * self.periods_per_year
        return annualized

    def calculate_annualized_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized volatility (standard deviation, Formula 5, Section 4.1).
        
        σ = sqrt( (1/(T-1)) * Σ(R_t - R̄)² )
        
        Annualization: σ_annual = σ_periodic × sqrt(periods_per_year)
        
        :param returns: DataFrame of periodic returns
        :return: Series of annualized standard deviations per asset
        """
        std_periodic = returns.std()
        annualized = std_periodic * np.sqrt(self.periods_per_year)
        return annualized

    def calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix of returns (Formula 3, Section 2.3).
        
        Cov(R_i, R_j) = E[(R_i - E[R_i])(R_j - E[R_j])]
        
        The covariance matrix Σ is used in portfolio variance calculation:
        σ_p² = w^T Σ w (Formula 4)
        
        :param returns: DataFrame of periodic returns
        :return: Covariance matrix (non-annualized, daily if input is daily)
        """
        return returns.cov()

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns (Section 4.4).
        
        ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)
        
        Correlation quantifies the degree of linear association between assets.
        Key for understanding diversification benefits (Section 3.3).
        
        :param returns: DataFrame of periodic returns
        :return: Correlation matrix with values in [-1, 1]
        """
        return returns.corr()