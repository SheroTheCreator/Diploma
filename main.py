import pandas as pd
import numpy as np

from utils.cli import parse_cli_arguments
from utils.config import get_config
from services.data_fetcher import download_prices
from services.data_preprocessor import clean_prices, compute_simple_returns
from services.statistics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_covariance_matrix,
    calculate_correlation_matrix
)
from services.risk_metrics import calculate_sharpe_ratio, calculate_value_at_risk
from services.optimizers import MeanVarianceOptimizer, WeightConstraints, PortfolioResult
from services.visualization import plot_correlation_matrix, plot_efficient_frontier, plot_price_history


def calculate_equal_weight_portfolio(
    n_assets: int, 
    annual_returns: pd.Series, 
    annual_cov: pd.DataFrame
) -> PortfolioResult:
    """Calculate equal-weight (1/n) benchmark portfolio."""
    weights = np.ones(n_assets) / n_assets
    port_return = float(np.dot(weights, annual_returns.values))
    port_var = float(weights.T @ annual_cov.values @ weights)
    
    return PortfolioResult(
        weights=weights, 
        expected_return=port_return, 
        volatility=float(np.sqrt(port_var))
    )


def main() -> None:
    """Execute main portfolio analysis workflow."""
    args = parse_cli_arguments()
    config = get_config(use_auto_dates=args.auto_dates)

    print("\n" + "=" * 60)
    print(" PORTFOLIO OPTIMIZATION ANALYSIS")
    print("=" * 60)
    print(f" Period: {config.start_date} to {config.end_date}\n")
    
    # 1. Fetch and Preprocess Data
    prices = download_prices(args.tickers, start=config.start_date, end=config.end_date)
    prices = clean_prices(prices)
    returns = compute_simple_returns(prices)

    # 2. Descriptive Statistics
    annual_returns = calculate_annualized_return(returns, config.periods_per_year)
    annual_volatility = calculate_annualized_volatility(returns, config.periods_per_year)
    cov_matrix = calculate_covariance_matrix(returns)
    corr_matrix = calculate_correlation_matrix(returns)

    # 3. Portfolio Optimization
    annual_cov = cov_matrix * config.periods_per_year
    optimizer = MeanVarianceOptimizer(expected_returns=annual_returns, cov_matrix=annual_cov)
    constraints = WeightConstraints(min_weight=0.0, max_weight=1.0)

    min_var_port = optimizer.min_variance_portfolio(constraints)
    max_sharpe_port = optimizer.max_sharpe_portfolio(config.risk_free_rate, constraints)
    frontier = optimizer.efficient_frontier(n_points=20, constraints)
    eq_port = calculate_equal_weight_portfolio(len(args.tickers), annual_returns, annual_cov)

    # 4. Display Results
    print("--- Returns & Volatility ---")
    for t in args.tickers:
        print(f"  {t:>6}: Return {annual_returns[t]:>7.2%}, Volatility {annual_volatility[t]:>7.2%}")

    print("\n--- Correlation Matrix ---")
    print(corr_matrix.round(3))

    print("\n--- Risk-Adjusted Metrics ---")
    for t in args.tickers:
        sr = calculate_sharpe_ratio(returns[t], config.risk_free_rate, config.periods_per_year)
        var = calculate_value_at_risk(returns[t], level=0.95)
        print(f"  {t:>6}: Sharpe Ratio {sr:>6.2f}, 95% VaR {var:>6.2%}")

    print("\n--- Optimal Portfolios ---")
    portfolios = [
        ("Min-Variance", min_var_port),
        ("Max-Sharpe", max_sharpe_port),
        ("Equal-Weight", eq_port)
    ]
    
    for name, port in portfolios:
        sharpe = (port.expected_return - config.risk_free_rate) / port.volatility
        weights_str = ", ".join(f"{t}: {w:.1%}" for t, w in zip(args.tickers, port.weights) if w > 0.01)
        print(f"\n{name}:")
        print(f"  Return:  {port.expected_return:.2%}")
        print(f"  Risk:    {port.volatility:.2%}")
        print(f"  Sharpe:  {sharpe:.2f}")
        print(f"  Weights: {weights_str}")

    # 5. Visualization
    if not args.disable_plots:
        print("\nGenerating charts...")
        plot_correlation_matrix(corr_matrix, config.figures_dir)
        plot_price_history(prices, config.figures_dir)
        
        assets_df = pd.DataFrame({
            "Ticker": args.tickers, 
            "Return": annual_returns.values, 
            "Volatility": annual_volatility.values
        })
        
        plot_efficient_frontier(
            frontier_points=frontier,
            individual_assets=assets_df,
            min_variance_portfolio=min_var_port,
            max_sharpe_portfolio=max_sharpe_port,
            equal_weight_portfolio=eq_port,
            output_dir=config.figures_dir
        )
        print(f"Charts saved to '{config.figures_dir}' directory.")
        
    print("\nAnalysis complete.\n")


if __name__ == "__main__":
    main()
