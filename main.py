from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from services.data_fetcher import YahooFinanceDataFetcher
from services.data_preprocessor import DataPreprocessor
from services.statistics import RiskReturnCalculator
from services.risk_metrics import MetricsCalculator
from services.optimizers import MeanVarianceOptimizer, WeightConstraints, PortfolioResult
from services.visualization import VisualizationService
from utils.cli import parse_cli_arguments
from utils.config import ProjectConfig


def calculate_equal_weight_portfolio(
    n_assets: int,
    annual_returns: pd.Series,
    annual_cov: pd.DataFrame,
) -> PortfolioResult:
    """Calculate equal-weight (1/n) benchmark portfolio."""
    weights = np.ones(n_assets) / n_assets
    port_return = float(np.dot(weights, annual_returns.values))
    port_var = float(weights.T @ annual_cov.values @ weights)
    port_vol = float(np.sqrt(port_var))
    return PortfolioResult(weights=weights, expected_return=port_return, volatility=port_vol)


def main() -> None:
    """
    Main workflow
    
    Steps:
    1. Data collection and preprocessing
    2. Descriptive statistics and risk metrics
    3. Correlation analysis
    4. Portfolio optimization
    5. Visualization and interpretation
    """
    
    # Parsing of command-line arguments
    cli_args = parse_cli_arguments()
    tickers = cli_args.tickers

    if len(tickers) < 1:
        raise ValueError("At least one ticker must be provided.")

    config = ProjectConfig(use_auto_dates=cli_args.auto_dates)

    # Show analysis period
    print("\n" + "=" * 70)
    print("  PORTFOLIO OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print(f"\n Analysis period: {config.start_date} to {config.end_date}")
    if cli_args.auto_dates:
        print("   (Dynamic: last 10 years from today)")
    else:
        print("   (Fixed period for reproducibility)")
    
    print("\nProcessing data...\n")

    # Download and preprocess data from Yahoo API
    fetcher = YahooFinanceDataFetcher()
    prices = fetcher.download_prices(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
    )

    preprocessor = DataPreprocessor()
    cleaned_prices = preprocessor.clean_prices(prices)
    returns = preprocessor.compute_simple_returns(cleaned_prices)

    # Compute descriptive statistics
    calculator = RiskReturnCalculator(periods_per_year=config.periods_per_year)
    annual_returns = calculator.calculate_annualized_return(returns)
    annual_volatility = calculator.calculate_annualized_volatility(returns)
    cov_matrix = calculator.calculate_covariance_matrix(returns)
    corr_matrix = calculator.calculate_correlation_matrix(returns)

    # Compute risk metrics
    metrics_calculator = MetricsCalculator(periods_per_year=config.periods_per_year)

    # Portfolio optimization 
    annual_cov = cov_matrix * config.periods_per_year
    optimizer = MeanVarianceOptimizer(
        expected_returns=annual_returns,
        cov_matrix=annual_cov,
    )
    constraints = WeightConstraints(min_weight=0.0, max_weight=1.0)

    min_var_port = optimizer.min_variance_portfolio(constraints=constraints)
    max_sharpe_port = optimizer.max_sharpe_portfolio(
        risk_free_rate=config.risk_free_rate,
        constraints=constraints,
    )
    frontier = optimizer.efficient_frontier(n_points=20, constraints=constraints)
    
    # Equal-weight benchmark
    eq_port = calculate_equal_weight_portfolio(
        len(tickers),
        annual_returns,
        annual_cov,
    )

    # Ensure output directory
    figures_path = Path(config.figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    # ========== DISPLAY RESULTS ==========
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # --- Asset Statistics ---
    print("\n ASSET STATISTICS (Annualized)")
    print("-" * 70)
    print("\nExpected Returns")
    print("-> Average yearly return based on historical data")
    print()
    print(annual_returns.to_string())
    
    print("\n\nVolatility")
    print("-> Measure of return variability (higher = riskier)")
    print()
    print(annual_volatility.to_string())

    # --- Correlation Analysis  ---
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)
    print("\n-> Correlation quantifies asset interdependence:")
    print("  +1.0 = perfect positive correlation (assets move together)")
    print("   0.0 = no linear relationship")
    print("  -1.0 = perfect negative correlation (strong diversification)")
    print()
    print(corr_matrix.to_string())
    
    # Interpret correlation for 2-asset case
    if len(tickers) == 2:
        t1, t2 = tickers
        corr_val = corr_matrix.loc[t1, t2]
        print()
        if corr_val < -0.1:
            print(f"{t1} and {t2} show negative correlation ({corr_val:.3f})")
            print("   -> Diversification benefit: combining them reduces portfolio risk")
        elif corr_val > 0.7:
            print(f"{t1} and {t2} are highly correlated ({corr_val:.3f})")
            print("   -> Limited diversification: they tend to move together")
        else:
            print(f"{t1} and {t2} show moderate correlation ({corr_val:.3f})")
            print("   -> Some diversification benefit available")

    # --- Risk-Adjusted Metrics ---
    print("\n" + "=" * 70)
    print("  RISK-ADJUSTED METRICS")
    print("-" * 70)
    
    print("\nSharpe Ratio")
    print("-> Excess return per unit of risk (higher = better)")
    print(f"  Risk-free rate: {config.risk_free_rate:.1%}")
    print()
    for ticker in tickers:
        sr = metrics_calculator.sharpe_ratio(
            returns=returns[ticker],
            risk_free_rate_annual=config.risk_free_rate,
        )
        print(f"  {ticker}: {sr:.4f}")

    var_level = 0.95
    print(f"\n\nValue at Risk (VaR) at {int(var_level * 100)}%")
    print("-> Maximum expected daily loss in worst 5% of scenarios")
    print("  (VaR does not provide info about losses beyond threshold)")
    print()
    for ticker in tickers:
        var_res = metrics_calculator.var_result(
            returns=returns[ticker],
            level=var_level,
            method="historical",
        )
        print(f"  {ticker}: {var_res.var:.2%}")
        print(f"           (5% chance of losing more than this per day)")

    # --- Optimal Portfolios ---
    print("\n" + "=" * 70)
    print("  OPTIMAL PORTFOLIOS (Markowitz Optimization)")
    print("-" * 70)

    print("\n1. Minimum-Variance Portfolio")
    print("   -> Lowest possible risk portfolio on efficient frontier")
    print()
    for ticker, w in zip(tickers, min_var_port.weights):
        if w > 0.001:  # Only show non-zero weights
            print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {min_var_port.expected_return:>6.2%}")
    print(f"   Volatility:      {min_var_port.volatility:>6.2%}")
    min_var_sharpe = (min_var_port.expected_return - config.risk_free_rate) / min_var_port.volatility
    print(f"   Sharpe Ratio:    {min_var_sharpe:>6.4f}")

    print("\n\n2. Maximum-Sharpe Portfolio")
    print("   -> Best risk-adjusted return (tangency portfolio)")
    print()
    for ticker, w in zip(tickers, max_sharpe_port.weights):
        if w > 0.001:
            print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {max_sharpe_port.expected_return:>6.2%}")
    print(f"   Volatility:      {max_sharpe_port.volatility:>6.2%}")
    max_sharpe_ratio = (max_sharpe_port.expected_return - config.risk_free_rate) / max_sharpe_port.volatility
    print(f"   Sharpe Ratio:    {max_sharpe_ratio:>6.4f}")

    print("\n\n3. Equal-Weight Benchmark (1/n):")
    print("   -> Simple naive diversification strategy")
    print()
    for ticker, w in zip(tickers, eq_port.weights):
        print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {eq_port.expected_return:>6.2%}")
    print(f"   Volatility:      {eq_port.volatility:>6.2%}")
    eq_sharpe = (eq_port.expected_return - config.risk_free_rate) / eq_port.volatility
    print(f"   Sharpe Ratio:    {eq_sharpe:>6.4f}")

    # Comparison
    print("\n\n COMPARISON:")
    print("-" * 70)
    comparison_data = {
        "Portfolio": ["Min-Variance", "Max-Sharpe", "Equal-Weight"],
        "Return": [
            f"{min_var_port.expected_return:.2%}",
            f"{max_sharpe_port.expected_return:.2%}",
            f"{eq_port.expected_return:.2%}"
        ],
        "Risk": [
            f"{min_var_port.volatility:.2%}",
            f"{max_sharpe_port.volatility:.2%}",
            f"{eq_port.volatility:.2%}"
        ],
        "Sharpe": [
            f"{min_var_sharpe:.4f}",
            f"{max_sharpe_ratio:.4f}",
            f"{eq_sharpe:.4f}"
        ]
    }
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # ========== Visualization ==========
    if not cli_args.disable_plots:
        print("\n" + "=" * 70)
        print("  GENERATING CHARTS...")
        print("-" * 70)
        
        viz = VisualizationService(output_dir=config.figures_dir)
        
        # Correlation heatmap
        viz.plot_correlation_matrix(corr_matrix, filename="correlation_matrix.png")
        print("  * Correlation heatmap saved")
        
        # Price history
        viz.plot_price_history(cleaned_prices, filename="price_history.png")
        print("  * Price history chart saved")
        
        # Efficient frontier
        assets_df = pd.DataFrame({
            "Ticker": tickers,
            "Return": annual_returns.values,
            "Volatility": annual_volatility.values,
        })
        
        viz.plot_efficient_frontier(
            frontier_points=frontier,
            individual_assets=assets_df,
            min_variance_portfolio=min_var_port,
            max_sharpe_portfolio=max_sharpe_port,
            filename="efficient_frontier.png",
        )
        print("  * Efficient frontier chart saved")

    # ========== Final Summary ==========
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {figures_path.absolute()}")
    if not cli_args.disable_plots:
        print("\nTIP: Check the efficient frontier chart to visualize")
        print("   the risk-return trade-off of different portfolios.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
