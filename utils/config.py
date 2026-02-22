"""
Global configuration for portfolio analysis.

Defines project-wide parameters for the empirical study.
"""

from __future__ import annotations

from datetime import datetime, timedelta


class ProjectConfig:
    """
    Global configuration parameters for the empirical study.
    
    Defines analysis period, annualization conventions, and risk-free rate.
    """

    def __init__(self, use_auto_dates: bool = False) -> None:
        """
        Initialize project configuration.
        
        :param use_auto_dates: If True, use dynamic period (current date - 10 years).
                               If False, use fixed period for reproducibility.
        """
        if use_auto_dates:
            # Dynamic: current date minus 10 years
            end = datetime.now()
            start = end - timedelta(days=365 * 10)
            self.start_date: str = start.strftime("%Y-%m-%d")
            self.end_date: str = end.strftime("%Y-%m-%d")
        else:
            # Fixed period for reproducibility
            self.start_date: str = "2013-01-01"
            self.end_date: str = "2022-12-31"
        
        # Trading days per year for annualization
        self.periods_per_year: int = 252
        
        # Annual risk-free rate
        self.risk_free_rate: float = 0.02  # 2% annual
        
        # Directory to store generated figures
        self.figures_dir: str = "figures"
