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
    
    Based on Markowitz mean-variance framework (Section 2).[file:2]
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
        Calculate annualized expected returns (Formula 1, Section 2.2).[file:2]
        
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
        Calculate annualized volatility (standard deviation, Formula 5, Section 4.1).[file:2]
        
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
        Calculate covariance matrix of returns (Formula 3, Section 2.3).[file:2]
        
        Cov(R_i, R_j) = E[(R_i - E[R_i])(R_j - E[R_j])]
        
        The covariance matrix Σ is used in portfolio variance calculation:
        σ_p² = w^T Σ w (Formula 4)
        
        :param returns: DataFrame of periodic returns
        :return: Covariance matrix (non-annualized, daily if input is daily)
        """
        return returns.cov()

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns (Section 4.4).[file:2]
        
        ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)
        
        Correlation quantifies the degree of linear association between assets.
        Key for understanding diversification benefits (Section 3.3).
        
        :param returns: DataFrame of periodic returns
        :return: Correlation matrix with values in [-1, 1]
        """
        return returns.corr()

    def calculate_portfolio_return(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
    ) -> float:
        """
        Calculate portfolio expected return (Formula 1, Section 2.2).[file:2]
        
        E[R_p] = Σ w_i * E[R_i]
        
        :param weights: Portfolio weights (must sum to 1)
        :param expected_returns: Expected returns of individual assets
        :return: Portfolio expected return
        """
        if len(weights) != len(expected_returns):
            raise ValueError("Weights and returns must have same length")
        
        return float(np.dot(weights, expected_returns.values))

    def calculate_portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame,
    ) -> float:
        """
        Calculate portfolio volatility (Formulas 2, 4, Section 2.2-2.3).[file:2]
        
        σ_p² = Σ_i Σ_j w_i w_j Cov(R_i, R_j)  (Formula 2)
        
        Or in matrix form:
        σ_p² = w^T Σ w  (Formula 4)
        
        Then: σ_p = sqrt(σ_p²)  (Formula 5)
        
        :param weights: Portfolio weights
        :param cov_matrix: Covariance matrix of asset returns
        :return: Portfolio standard deviation (volatility)
        """
        if len(weights) != len(cov_matrix):
            raise ValueError("Weights and covariance matrix dimensions must match")
        
        # Formula 4: w^T Σ w
        variance = float(weights.T @ cov_matrix.values @ weights)
        
        # Formula 5: σ = sqrt(variance)
        return float(np.sqrt(variance))

    def calculate_portfolio_statistics(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate complete portfolio statistics (annualized).
        
        Convenience method combining return, volatility, and Sharpe calculation.
        
        :param weights: Portfolio weights
        :param returns: DataFrame of periodic returns
        :return: Dictionary with 'return', 'volatility', and other metrics
        """
        # Expected returns and covariance
        annual_returns = self.calculate_annualized_return(returns)
        cov_matrix = self.calculate_covariance_matrix(returns)
        
        # Annualize covariance
        annual_cov = cov_matrix * self.periods_per_year
        
        # Portfolio metrics
        port_return = self.calculate_portfolio_return(weights, annual_returns)
        port_volatility = self.calculate_portfolio_volatility(weights, annual_cov)
        
        return {
            "return": port_return,
            "volatility": port_volatility,
        }
