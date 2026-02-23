# Portfolio Optimization Tool

An empirical implementation of the Markowitz Mean-Variance framework. This tool downloads historical market data, calculates risk/return statistics, and computes optimal portfolio weights using convex optimization, wrapped in a user-friendly Streamlit web interface.

## Features

- **Historical Data Processing:** Automatically fetches adjusted close prices from Yahoo Finance and computes periodic returns.
- **Risk & Return Metrics:** Calculates annualized returns, volatility, Covariance/Correlation matrices, Sharpe Ratio, and historical Value at Risk (VaR).
- **Portfolio Optimization:** Uses `scipy.optimize` to find:
  - Minimum-Variance Portfolio
  - Maximum-Sharpe (Tangency) Portfolio
  - Efficient Frontier (Pareto-optimal portfolios)
- **Custom Constraints:** Set a maximum weight per asset to enforce realistic diversification.
- **Benchmark Comparison:** Automatically fetches a market index (e.g., S&P 500) and compares your optimal portfolios against it.
- **Data Visualization:** Generates correlation heatmaps, price history charts, and the Efficient Frontier plot.

## 🚀 Quick Start (Easiest Way to Run)

You don't need to manually install dependencies or type commands if you use the provided launch scripts.

**On Windows:**
1. Simply double-click the **`run.bat`** file.
2. It will automatically set up Python, install everything needed, and open the app in your browser!

**On Mac / Linux:**
1. Open your terminal in the project folder.
2. Run the bash script:
   ```bash
   bash run.sh
   ```

*Note: The first launch might take a minute as it downloads the required libraries.*

---

## Manual Installation & Usage

If you prefer to run things manually:

1. Clone the repository and navigate to the folder:
   ```bash
   git clone https://github.com/SheroTheCreator/Diploma.git
   cd Diploma
   ```
2. Install the required dependencies (requires Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
Diploma/
├── app.py              # Streamlit Web Application (Entry point)
├── run.bat             # Auto-launcher for Windows
├── run.sh              # Auto-launcher for Mac/Linux
├── requirements.txt    # Python dependencies
├── utils/
│   └── config.py       # Global config (dates, risk-free rate)
└── services/
    ├── data_fetcher.py      # Downloads data via yfinance
    ├── data_preprocessor.py # Cleans prices and computes returns
    ├── statistics.py        # Returns, volatility, correlation
    ├── risk_metrics.py      # Sharpe Ratio, VaR
    ├── optimizers.py        # Markowitz optimization engine
    └── visualization.py     # Matplotlib/Seaborn charts
```

## Output

The application interface presents:
1. **Portfolio Results:** A summary comparison table comparing optimal portfolios against the market benchmark, along with exact capital allocation weights.
2. **Visualizations:** The Efficient Frontier plot visualizing Pareto-optimal risk-return combinations, and historical price charts.
3. **Raw Data:** The asset Correlation Matrix and individual Risk-Adjusted Metrics (Sharpe Ratio, 95% Historical VaR).