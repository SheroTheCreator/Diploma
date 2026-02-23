"""
Global configuration for portfolio analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ProjectConfig:
    """Project-wide configuration parameters."""
    start_date: str
    end_date: str
    periods_per_year: int = 252
    risk_free_rate: float = 0.02
    figures_dir: str = "figures"


def get_config(years: int = 10, use_fixed: bool = False) -> ProjectConfig:
    """Generate configuration based on selected date mode."""
    if not use_fixed:
        # Dynamic period: current date minus N years
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        return ProjectConfig(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    
    # Fixed period for strict reproducibility
    return ProjectConfig("2013-01-01", "2022-12-31")
