import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.config import get_config
from services.data_fetcher import download_prices
from services.data_preprocessor import clean_prices, compute_simple_returns
from services.statistics import (
    calculate_annualized_return, calculate_annualized_volatility,
    calculate_covariance_matrix, calculate_correlation_matrix
)
from services.risk_metrics import calculate_sharpe_ratio, calculate_value_at_risk
from services.optimizers import MeanVarianceOptimizer, WeightConstraints, PortfolioResult
from services.visualization import plot_correlation_matrix, plot_efficient_frontier, plot_price_history


def calculate_equal_weight_portfolio(n_assets, annual_returns, annual_cov):
    weights = np.ones(n_assets) / n_assets
    port_return = float(np.dot(weights, annual_returns.values))
    port_var = float(weights.T @ annual_cov.values @ weights)
    return PortfolioResult(weights=weights, expected_return=port_return, volatility=float(np.sqrt(port_var)))

# --- Page Configuration ---
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

st.title("📈 Portfolio Optimization Web App")
st.markdown("Build an optimal portfolio based on the Markowitz Mean-Variance framework using real historical data.")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Parameters")
tickers_input = st.sidebar.text_input("Tickers (space separated)", "AAPL MSFT GOOGL JPM")
benchmark_input = st.sidebar.text_input("Benchmark", "SPY")
max_weight = st.sidebar.slider("Max Allocation per Asset", 0.05, 1.0, 0.40, 0.05)
auto_dates = st.sidebar.checkbox("Use Dynamic Dates (Last 10 Years)", value=True)

if st.sidebar.button("Run Analysis", type="primary"):
    
    # Process Inputs
    tickers = [t.strip().upper() for t in tickers_input.split() if t.strip()]
    benchmark = benchmark_input.strip().upper()
    
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers for portfolio optimization.")
        st.stop()
        
    min_possible = 1.0 / len(tickers)
    if max_weight < min_possible:
        st.error(f"Max weight ({max_weight}) is mathematically impossible for {len(tickers)} assets. It must be at least {min_possible:.2f}.")
        st.stop()
        
    with st.spinner("Fetching market data and running optimization..."):
        try:
            # 1. Fetch & Preprocess
            config = get_config(use_auto_dates=auto_dates)
            fetch_tickers = list(set(tickers + [benchmark]))
            
            prices = download_prices(fetch_tickers, start=config.start_date, end=config.end_date)
            prices = clean_prices(prices)
            returns = compute_simple_returns(prices)

            port_returns = returns[tickers]
            bench_returns = returns[benchmark]

            # 2. Descriptive Statistics
            annual_returns = calculate_annualized_return(port_returns, config.periods_per_year)
            annual_volatility = calculate_annualized_volatility(port_returns, config.periods_per_year)
            cov_matrix = calculate_covariance_matrix(port_returns)
            corr_matrix = calculate_correlation_matrix(port_returns)

            # Benchmark stats
            bench_ann_ret = bench_returns.mean() * config.periods_per_year
            bench_ann_vol = bench_returns.std() * np.sqrt(config.periods_per_year)
            bench_sharpe = calculate_sharpe_ratio(bench_returns, config.risk_free_rate, config.periods_per_year)

            # 3. Optimizations
            annual_cov = cov_matrix * config.periods_per_year
            optimizer = MeanVarianceOptimizer(expected_returns=annual_returns, cov_matrix=annual_cov)
            constraints = WeightConstraints(min_weight=0.0, max_weight=max_weight)

            min_var_port = optimizer.min_variance_portfolio(constraints)
            max_sharpe_port = optimizer.max_sharpe_portfolio(config.risk_free_rate, constraints)
            frontier = optimizer.efficient_frontier(n_points=20, constraints=constraints)
            eq_port = calculate_equal_weight_portfolio(len(tickers), annual_returns, annual_cov)

            # --- UI Layout: Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Portfolio Results", "📈 Visualizations", "🔍 Raw Data"])
            
            with tab1:
                st.subheader(f"Portfolio Performance vs Benchmark ({benchmark})")
                st.caption(f"Analysis period: **{config.start_date}** to **{config.end_date}**")
                
                # Comparison Dataframe
                comp_data = {
                    "Portfolio": ["Min-Variance", "Max-Sharpe (Tangency)", "Equal-Weight (1/n)", f"Benchmark ({benchmark})"],
                    "Expected Return": [
                        f"{min_var_port.expected_return:.2%}", f"{max_sharpe_port.expected_return:.2%}",
                        f"{eq_port.expected_return:.2%}", f"{bench_ann_ret:.2%}"
                    ],
                    "Volatility (Risk)": [
                        f"{min_var_port.volatility:.2%}", f"{max_sharpe_port.volatility:.2%}",
                        f"{eq_port.volatility:.2%}", f"{bench_ann_vol:.2%}"
                    ],
                    "Sharpe Ratio": [
                        f"{(min_var_port.expected_return - config.risk_free_rate) / min_var_port.volatility:.3f}",
                        f"{(max_sharpe_port.expected_return - config.risk_free_rate) / max_sharpe_port.volatility:.3f}",
                        f"{(eq_port.expected_return - config.risk_free_rate) / eq_port.volatility:.3f}",
                        f"{bench_sharpe:.3f}"
                    ]
                }
                
                # Render table nicely
                df_comp = pd.DataFrame(comp_data).set_index("Portfolio")
                st.dataframe(df_comp, use_container_width=True)
                
                st.subheader("Asset Allocation (Weights)")
                weights_df = pd.DataFrame({
                    "Min-Variance": min_var_port.weights,
                    "Max-Sharpe": max_sharpe_port.weights,
                    "Equal-Weight": eq_port.weights
                }, index=tickers)
                
                # Highlight the max values in the columns
                st.dataframe(
                    weights_df.style.format("{:.2%}").highlight_max(axis=0, color="lightgreen"),
                    use_container_width=True
                )

            with tab2:
                st.subheader("Efficient Frontier")
                st.markdown("Visualizing the optimal risk-return trade-off. Any portfolio below the blue line is sub-optimal.")
                assets_df = pd.DataFrame({"Ticker": tickers, "Return": annual_returns.values, "Volatility": annual_volatility.values})
                bench_stat = {"Ticker": benchmark, "Return": bench_ann_ret, "Volatility": bench_ann_vol}
                
                fig_ef = plot_efficient_frontier(
                    frontier, assets_df, min_var_port, max_sharpe_port, eq_port, bench_stat, save=False
                )
                st.pyplot(fig_ef)
                plt.close(fig_ef)  # Free memory
                
                st.subheader("Price History")
                fig_pr = plot_price_history(prices, save=False)
                st.pyplot(fig_pr)
                plt.close(fig_pr)
                
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Correlation Matrix")
                    st.markdown("Measures how assets move together. **Negative** numbers mean strong diversification.")
                    fig_corr = plot_correlation_matrix(corr_matrix, save=False)
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)
                    
                with col2:
                    st.subheader("Annualized Asset Metrics")
                    st.markdown("Individual risk and return profiles for the selected period.")
                    stats_df = pd.DataFrame({
                        "Expected Return": annual_returns,
                        "Volatility": annual_volatility
                    })
                    st.dataframe(stats_df.style.format("{:.2%}"), use_container_width=True)
                    
                    st.subheader("Value at Risk (95%)")
                    st.markdown("The maximum expected daily loss in the worst 5% of scenarios.")
                    var_data = {t: f"{calculate_value_at_risk(port_returns[t], 0.95):.2%}" for t in tickers}
                    st.dataframe(pd.Series(var_data, name="95% VaR"), use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Check if all ticker symbols are valid and publicly traded.")
else:
    st.info("👈 Enter your parameters in the sidebar and click **Run Analysis** to build portfolios.")
