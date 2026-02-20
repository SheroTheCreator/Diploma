"""
Global configuration for portfolio analysis.

Defines project-wide parameters for the empirical study (Section 7).[file:2]
"""

from __future__ import annotations

from datetime import datetime, timedelta


class ProjectConfig:
    """
    Global configuration parameters for the empirical study.
    
    Defines analysis period, annualization conventions, and risk-free rate
    as discussed in Section 7 (Practical part).[file:2]
    """

    def __init__(self, use_auto_dates: bool = False) -> None:
        """
        Initialize project configuration.
        
        :param use_auto_dates: If True, use dynamic period (current date - 10 years).
                               If False, use fixed period for reproducibility (Section 7.1).[file:2]
        """
        if use_auto_dates:
            # Dynamic: current date minus 10 years
            # Useful for practical application with always-current data
            end = datetime.now()
            start = end - timedelta(days=365 * 10)
            self.start_date: str = start.strftime("%Y-%m-%d")
            self.end_date: str = end.strftime("%Y-%m-%d")
        else:
            # Fixed period for reproducibility in thesis
            # Ensures consistent results when re-running analysis (Section 6.2).[file:2]
            self.start_date: str = "2013-01-01"
            self.end_date: str = "2022-12-31"
        
        # Trading days per year for annualization
        # Standard assumption: 252 trading days in a year (Section 5.2)
        self.periods_per_year: int = 252
        
        # Annual risk-free rate (used in Sharpe ratio, Formula 6)
        # Approximates average US Treasury bill rate over analysis period
        # Can be adjusted based on current market conditions (Section 4.2).[file:2]
        self.risk_free_rate: float = 0.02  # 2% annual
        
        # Directory to store generated figures
        # Charts from Section 5.5 (Visualization of Results)
        self.figures_dir: str = "figures"
        
    def get_annualization_factor(self, frequency: str = "daily") -> int:
        """
        Get annualization factor for different data frequencies.
        
        :param frequency: Data frequency ('daily', 'weekly', 'monthly')
        :return: Periods per year
        """
        factors = {
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
        }
        
        if frequency not in factors:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        return factors[frequency]
    
    def to_dict(self) -> dict:
        """
        Export configuration as dictionary.
        
        Useful for logging and reproducibility documentation.
        """
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "periods_per_year": self.periods_per_year,
            "risk_free_rate": self.risk_free_rate,
            "figures_dir": self.figures_dir,
        }
