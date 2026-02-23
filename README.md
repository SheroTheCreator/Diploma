# Portfolio Optimization Tool

An empirical implementation of the Markowitz Mean-Variance framework. This tool downloads historical market data, calculates risk/return statistics, and computes optimal portfolio weights using convex optimization.

The project supports **two interfaces**:
- **Web App** (`app.py`) — an interactive browser-based UI powered by Streamlit.
- **CLI** (`main.py`) — a classic command-line interface for terminal usage.

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SheroTheCreator/Diploma.git
   cd Diploma
   ```
2. Install the required dependencies (requires Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App (Recommended)

This is the easiest and most visual way to use the tool. Run the following command and the app will open automatically in your browser:

```bash
streamlit run app.py
```

The sidebar lets you:
- Enter ticker symbols (space-separated)
- Set a custom benchmark (e.g., `SPY`, `QQQ`)
- Adjust the maximum weight per asset with a slider
- Toggle between fixed and dynamic (last 10 years) date ranges

Results are shown across three tabs:
- **Portfolio Results** — comparison table and allocation weights.
- **Visualizations** — interactive Efficient Frontier and Price History charts.
- **Raw Data** — Correlation matrix, individual asset metrics, and VaR.

## Running from Command Line

### Basic Usage
```bash
python main.py --tickers AAPL MSFT GOOGL JPM
```

### Advanced Usage (Constraints and Benchmark)
```bash
python main.py --tickers AAPL MSFT GOOGL JPM --max-weight 0.4 --benchmark SPY
```

### Available CLI Arguments
- `--tickers` : List of asset tickers (e.g., AAPL MSFT). If omitted, the program will prompt you interactively.
- `--disable-plots` : Disables graphical output.
- `--auto-dates` : Uses a dynamic date range (last 10 years from today).
- `--max-weight` : Maximum allocation per asset (e.g., `0.35` for 35%). Default is `1.0`.
- `--benchmark` : Market benchmark ticker. Default is `SPY`.

## Project Structure

```
Diploma/
├── app.py              # Streamlit Web UI
├── main.py             # CLI entry point
├── requirements.txt    # Python dependencies
├── utils/
│   ├── cli.py          # CLI argument parser
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

The analysis produces:
1. Asset Statistics (Annualized Expected Return, Volatility).
2. Correlation Matrix with diversification interpretation.
3. Risk-Adjusted Metrics (Sharpe Ratio, 95% Historical VaR) vs benchmark.
4. Optimal Portfolio Weights (Min-Variance, Max-Sharpe, Equal-Weight).
5. Summary comparison table including the market benchmark.

In CLI mode, charts are saved to the `figures/` directory.