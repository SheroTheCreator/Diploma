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

    print("\n" + "=" * 70)
    print("  PORTFOLIO OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print(f"\n Analysis period: {config.start_date} to {config.end_date}")
    
    if args.auto_dates:
        print("   (Dynamic: last 10 years from today)")
    else:
        print("   (Fixed period for reproducibility)")
        
    if args.max_weight < 1.0:
        print(f"   (Constraints: max weight per asset = {args.max_weight:.1%})")
        
    print(f"   (Benchmark: {args.benchmark})")
    
    print("\nProcessing data...\n")
    
    # 1. Fetch and Preprocess Data (Including Benchmark)
    fetch_tickers = list(set(args.tickers + [args.benchmark]))
    prices = download_prices(fetch_tickers, start=config.start_date, end=config.end_date)
    prices = clean_prices(prices)
    returns = compute_simple_returns(prices)

    # Split returns into portfolio assets and benchmark
    port_returns = returns[args.tickers]
    bench_returns = returns[args.benchmark]

    # 2. Descriptive Statistics
    annual_returns = calculate_annualized_return(port_returns, config.periods_per_year)
    annual_volatility = calculate_annualized_volatility(port_returns, config.periods_per_year)
    cov_matrix = calculate_covariance_matrix(port_returns)
    corr_matrix = calculate_correlation_matrix(port_returns)

    # Benchmark statistics
    bench_ann_ret = bench_returns.mean() * config.periods_per_year
    bench_ann_vol = bench_returns.std() * np.sqrt(config.periods_per_year)
    bench_sharpe = calculate_sharpe_ratio(bench_returns, config.risk_free_rate, config.periods_per_year)
    bench_var = calculate_value_at_risk(bench_returns, level=0.95)

    # 3. Portfolio Optimization
    annual_cov = cov_matrix * config.periods_per_year
    optimizer = MeanVarianceOptimizer(expected_returns=annual_returns, cov_matrix=annual_cov)
    constraints = WeightConstraints(min_weight=0.0, max_weight=args.max_weight)

    min_var_port = optimizer.min_variance_portfolio(constraints)
    max_sharpe_port = optimizer.max_sharpe_portfolio(config.risk_free_rate, constraints)
    frontier = optimizer.efficient_frontier(n_points=20, constraints)
    eq_port = calculate_equal_weight_portfolio(len(args.tickers), annual_returns, annual_cov)

    # ========== DISPLAY RESULTS ==========
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # --- Asset Statistics ---
    print("\n ASSET STATISTICS (Annualized)")
    print("-" * 70)
    print(f"\nExpected Returns (Benchmark {args.benchmark}: {bench_ann_ret:.2%})")
    print("-> Average yearly return based on historical data")
    print()
    print(annual_returns.to_string())
    
    print(f"\n\nVolatility (Benchmark {args.benchmark}: {bench_ann_vol:.2%})")
    print("-> Measure of return variability (higher = riskier)")
    print()
    print(annual_volatility.to_string())

    # --- Correlation Analysis  ---
    print("\n\n" + "=" * 70)
    print("  CORRELATION ANALYSIS")
    print("-" * 70)
    print("\n-> Correlation quantifies asset interdependence:")
    print("  +1.0 = perfect positive correlation (assets move together)")
    print("   0.0 = no linear relationship")
    print("  -1.0 = perfect negative correlation (strong diversification)")
    print()
    print(corr_matrix.to_string())
    
    # Interpret correlation for 2-asset case
    if len(args.tickers) == 2:
        t1, t2 = args.tickers
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
    print("\n\n" + "=" * 70)
    print("  RISK-ADJUSTED METRICS")
    print("-" * 70)
    
    print("\nSharpe Ratio")
    print("-> Excess return per unit of risk (higher = better)")
    print(f"  Risk-free rate: {config.risk_free_rate:.1%}")
    print()
    print(f"  [Benchmark] {args.benchmark}: {bench_sharpe:.4f}\n")
    for t in args.tickers:
        sr = calculate_sharpe_ratio(port_returns[t], config.risk_free_rate, config.periods_per_year)
        print(f"  {t}: {sr:.4f}")

    var_level = 0.95
    print(f"\n\nValue at Risk (VaR) at {int(var_level * 100)}%")
    print("-> Maximum expected daily loss in worst 5% of scenarios")
    print()
    print(f"  [Benchmark] {args.benchmark}: {bench_var:.2%} (5% chance of losing more than this per day)\n")
    for t in args.tickers:
        var = calculate_value_at_risk(port_returns[t], level=var_level)
        print(f"  {t}: {var:.2%}")

    # --- Optimal Portfolios ---
    print("\n\n" + "=" * 70)
    print("  OPTIMAL PORTFOLIOS (Markowitz Optimization)")
    print("-" * 70)

    print("\n1. Minimum-Variance Portfolio")
    print("   -> Lowest possible risk portfolio on efficient frontier")
    print()
    for ticker, w in zip(args.tickers, min_var_port.weights):
        if w > 0.001:  # Only show non-zero weights
            print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {min_var_port.expected_return:>6.2%}")
    print(f"   Volatility:      {min_var_port.volatility:>6.2%}")
    min_var_sharpe = (min_var_port.expected_return - config.risk_free_rate) / min_var_port.volatility
    print(f"   Sharpe Ratio:    {min_var_sharpe:>6.4f}")

    print("\n\n2. Maximum-Sharpe Portfolio")
    print("   -> Best risk-adjusted return (tangency portfolio)")
    print()
    for ticker, w in zip(args.tickers, max_sharpe_port.weights):
        if w > 0.001:
            print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {max_sharpe_port.expected_return:>6.2%}")
    print(f"   Volatility:      {max_sharpe_port.volatility:>6.2%}")
    max_sharpe_ratio = (max_sharpe_port.expected_return - config.risk_free_rate) / max_sharpe_port.volatility
    print(f"   Sharpe Ratio:    {max_sharpe_ratio:>6.4f}")

    print("\n\n3. Equal-Weight Benchmark (1/n):")
    print("   -> Simple naive diversification strategy")
    print()
    for ticker, w in zip(args.tickers, eq_port.weights):
        print(f"     {ticker}: {w:>6.2%}")
    print(f"\n   Expected Return: {eq_port.expected_return:>6.2%}")
    print(f"   Volatility:      {eq_port.volatility:>6.2%}")
    eq_sharpe = (eq_port.expected_return - config.risk_free_rate) / eq_port.volatility
    print(f"   Sharpe Ratio:    {eq_sharpe:>6.4f}")

    # Comparison
    print("\n\n COMPARISON:")
    print("-" * 70)
    comparison_data = {
        "Portfolio": ["Min-Variance", "Max-Sharpe", "Equal-Weight", f"Benchmark ({args.benchmark})"],
        "Return": [
            f"{min_var_port.expected_return:.2%}",
            f"{max_sharpe_port.expected_return:.2%}",
            f"{eq_port.expected_return:.2%}",
            f"{bench_ann_ret:.2%}"
        ],
        "Risk": [
            f"{min_var_port.volatility:.2%}",
            f"{max_sharpe_port.volatility:.2%}",
            f"{eq_port.volatility:.2%}",
            f"{bench_ann_vol:.2%}"
        ],
        "Sharpe": [
            f"{min_var_sharpe:.4f}",
            f"{max_sharpe_ratio:.4f}",
            f"{eq_sharpe:.4f}",
            f"{bench_sharpe:.4f}"
        ]
    }
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # 5. Visualization
    if not args.disable_plots:
        print("\n\n" + "=" * 70)
        print("  GENERATING CHARTS...")
        print("-" * 70)
        
        plot_correlation_matrix(corr_matrix, config.figures_dir)
        print("  * Correlation heatmap saved")
        
        plot_price_history(prices, config.figures_dir)
        print("  * Price history chart saved")
        
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
            benchmark_stat={"Ticker": args.benchmark, "Return": bench_ann_ret, "Volatility": bench_ann_vol},
            output_dir=config.figures_dir
        )
        print("  * Efficient frontier chart saved")

    # ========== Final Summary ==========
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to the '{config.figures_dir}' directory.")
    if not args.disable_plots:
        print("\nTIP: Check the efficient frontier chart to visualize")
        print("   the risk-return trade-off of different portfolios vs the Benchmark.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
