"""
Visualization service for portfolio analysis results.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from services.optimizers import PortfolioResult


def _setup_plot(figsize: tuple[int, int] = (10, 6)) -> None:
    """Configure consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = 10


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame, 
    output_dir: str = "figures", 
    filename: str = "correlation_matrix.png"
) -> None:
    """Plot correlation matrix as a heatmap."""
    _setup_plot(figsize=(8, 6))
    plt.figure()
    
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
        square=True, linewidths=1, cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1
    )
    
    plt.title("Asset Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_efficient_frontier(
    frontier_points: list[PortfolioResult],
    individual_assets: pd.DataFrame,
    min_variance_portfolio: PortfolioResult | None = None,
    max_sharpe_portfolio: PortfolioResult | None = None,
    equal_weight_portfolio: PortfolioResult | None = None,
    output_dir: str = "figures",
    filename: str = "efficient_frontier.png",
) -> None:
    """Plot efficient frontier with optimal portfolios and individual assets."""
    _setup_plot(figsize=(10, 7))
    fig, ax = plt.subplots()

    # Efficient frontier line
    if frontier_points:
        vols = [p.volatility for p in frontier_points]
        rets = [p.expected_return for p in frontier_points]
        ax.plot(vols, rets, "b-", linewidth=2.5, label="Efficient Frontier", zorder=2)

    # Individual assets
    for _, row in individual_assets.iterrows():
        ax.scatter(row["Volatility"], row["Return"], marker="o", s=120, color="gray", edgecolors="black", alpha=0.7, zorder=3)
        ax.annotate(row["Ticker"], (row["Volatility"], row["Return"]), xytext=(7, 7), textcoords="offset points", fontweight="bold")

    # Min-Variance portfolio
    if min_variance_portfolio:
        ax.scatter(
            min_variance_portfolio.volatility, min_variance_portfolio.expected_return, 
            marker="s", s=180, color="green", edgecolors="black", label="Min-Variance", zorder=5
        )

    # Max-Sharpe portfolio
    if max_sharpe_portfolio:
        ax.scatter(
            max_sharpe_portfolio.volatility, max_sharpe_portfolio.expected_return, 
            marker="*", s=350, color="gold", edgecolors="black", label="Max-Sharpe", zorder=5
        )

    # Equal-Weight portfolio
    if equal_weight_portfolio:
        ax.scatter(
            equal_weight_portfolio.volatility, equal_weight_portfolio.expected_return, 
            marker="^", s=150, color="orange", edgecolors="black", label="Equal-Weight", zorder=4
        )

    ax.set_xlabel("Annualized Volatility (Risk)", fontweight="bold")
    ax.set_ylabel("Annualized Expected Return", fontweight="bold")
    ax.set_title("Efficient Frontier", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_price_history(
    prices: pd.DataFrame, 
    output_dir: str = "figures", 
    filename: str = "price_history.png"
) -> None:
    """Plot historical price time series."""
    _setup_plot(figsize=(12, 6))
    fig, ax = plt.subplots()
    
    for col in prices.columns:
        ax.plot(prices.index, prices[col], label=col, linewidth=1.8)
    
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Price", fontweight="bold")
    ax.set_title("Historical Prices", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / filename, dpi=150, bbox_inches="tight")
    plt.close()
