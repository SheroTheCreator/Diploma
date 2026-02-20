"""
Visualization service for portfolio analysis results.

Implements graphical output as discussed in Section 5.5 (Visualization of Results).[file:2]
Uses matplotlib library for creating plots of the efficient frontier,
correlation matrices, and price histories.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from services.optimizers import PortfolioResult


class VisualizationService:
    """
    Handles chart generation for portfolio analysis results.
    
    Creates visualizations of:
    - Efficient frontier (Section 2.4, Figure 1)
    - Correlation matrices (Section 4.4)
    - Historical price series (Section 7.3)
    
    Uses matplotlib and seaborn as specified in Section 5.5.[file:2]
    """

    def __init__(self, output_dir: str = "figures") -> None:
        """
        Initialize visualization service with output directory.
        
        :param output_dir: Directory to save generated figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 10

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix",
        filename: str = "correlation_matrix.png",
    ) -> None:
        """
        Plot correlation matrix as a heatmap (Section 4.4).[file:2]
        
        Visualizes the degree of linear association between asset returns.
        Important for understanding diversification benefits.
        
        :param corr_matrix: Correlation matrix from calculate_correlation_matrix()
        :param title: Chart title
        :param filename: Output filename
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation Coefficient"},
            vmin=-1,
            vmax=1,
        )
        
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_efficient_frontier(
        self,
        frontier_points: List[PortfolioResult],
        individual_assets: Optional[pd.DataFrame] = None,
        min_variance_portfolio: Optional[PortfolioResult] = None,
        max_sharpe_portfolio: Optional[PortfolioResult] = None,
        equal_weight_portfolio: Optional[PortfolioResult] = None,
        title: str = "Efficient Frontier",
        filename: str = "efficient_frontier.png",
    ) -> None:
        """
        Plot efficient frontier with optimal portfolios (Section 2.4, Figure 1).[file:2]
        
        The efficient frontier represents the set of portfolios that are not dominated
        by any other portfolio in the risk-return trade-off (Pareto-optimal solutions).
        
        :param frontier_points: List of PortfolioResult from efficient_frontier()
        :param individual_assets: DataFrame with columns ['Return', 'Volatility', 'Ticker']
        :param min_variance_portfolio: Global minimum-variance portfolio
        :param max_sharpe_portfolio: Maximum-Sharpe ratio portfolio
        :param equal_weight_portfolio: Equal-weight (1/n) benchmark
        :param title: Chart title
        :param filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot efficient frontier line
        if frontier_points:
            vols = [p.volatility for p in frontier_points]
            rets = [p.expected_return for p in frontier_points]
            ax.plot(
                vols,
                rets,
                "b-",
                linewidth=2.5,
                label="Efficient Frontier",
                zorder=2,
            )

        # Plot individual assets
        if individual_assets is not None:
            for idx, row in individual_assets.iterrows():
                ax.scatter(
                    row["Volatility"],
                    row["Return"],
                    marker="o",
                    s=120,
                    color="gray",
                    edgecolors="black",
                    linewidths=1.5,
                    alpha=0.7,
                    zorder=3,
                )
                ax.annotate(
                    row["Ticker"],
                    (row["Volatility"], row["Return"]),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                )

        # Highlight min-variance portfolio
        if min_variance_portfolio:
            ax.scatter(
                min_variance_portfolio.volatility,
                min_variance_portfolio.expected_return,
                marker="s",
                s=180,
                color="green",
                edgecolors="black",
                linewidths=2,
                label="Min-Variance",
                zorder=5,
            )

        # Highlight max-Sharpe portfolio
        if max_sharpe_portfolio:
            ax.scatter(
                max_sharpe_portfolio.volatility,
                max_sharpe_portfolio.expected_return,
                marker="*",
                s=350,
                color="gold",
                edgecolors="black",
                linewidths=2,
                label="Max-Sharpe",
                zorder=5,
            )

        # Highlight equal-weight benchmark
        if equal_weight_portfolio:
            ax.scatter(
                equal_weight_portfolio.volatility,
                equal_weight_portfolio.expected_return,
                marker="^",
                s=150,
                color="orange",
                edgecolors="black",
                linewidths=1.5,
                label="Equal-Weight",
                zorder=4,
            )

        ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Annualized Expected Return", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_return_distribution(
        self,
        returns: pd.Series,
        ticker: str,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot histogram of return distribution for a single asset (Section 4.1).[file:2]
        
        Visualizes the empirical distribution of returns. Helps assess whether
        normality assumption (Section 2.1) is reasonable.
        
        :param returns: Series of periodic returns
        :param ticker: Asset ticker symbol
        :param filename: Output filename
        """
        if filename is None:
            filename = f"return_distribution_{ticker}.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(returns, bins=50, color="steelblue", edgecolor="black", alpha=0.7, density=True)
        
        # Add mean line
        mean_return = returns.mean()
        ax.axvline(mean_return, color="red", linestyle="--", linewidth=2, 
                   label=f"Mean: {mean_return:.4f}")
        
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_pdf, 'r-', linewidth=2, alpha=0.6, label='Normal fit')
        
        ax.set_xlabel("Daily Returns", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Return Distribution: {ticker}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_price_history(
        self,
        prices: pd.DataFrame,
        title: str = "Historical Prices",
        filename: str = "price_history.png",
    ) -> None:
        """
        Plot price time series for all assets (Section 7.3).[file:2]
        
        Shows historical price evolution over the analysis period.
        
        :param prices: DataFrame with prices (columns = tickers, index = dates)
        :param title: Chart title
        :param filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for column in prices.columns:
            ax.plot(prices.index, prices[column], label=column, linewidth=1.8)
        
        ax.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate date labels for readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_weights_comparison(
        self,
        portfolios_dict: dict[str, PortfolioResult],
        tickers: List[str],
        title: str = "Portfolio Weights Comparison",
        filename: str = "weights_comparison.png",
    ) -> None:
        """
        Plot side-by-side bar chart comparing portfolio weights.
        
        Useful for visualizing how different optimization objectives
        lead to different asset allocations.
        
        :param portfolios_dict: Dictionary {portfolio_name: PortfolioResult}
        :param tickers: List of asset ticker symbols
        :param title: Chart title
        :param filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_portfolios = len(portfolios_dict)
        n_assets = len(tickers)
        x = np.arange(n_assets)
        width = 0.8 / n_portfolios
        
        for i, (name, portfolio) in enumerate(portfolios_dict.items()):
            offset = (i - n_portfolios / 2) * width + width / 2
            ax.bar(x + offset, portfolio.weights, width, label=name)
        
        ax.set_xlabel("Asset", fontsize=11, fontweight="bold")
        ax.set_ylabel("Weight", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
