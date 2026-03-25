# 📈 Portfolio Optimization Tool

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6?logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Bachelor's Thesis Project** — Czech University of Life Sciences Prague, 2026
> *Analysis of Historical Market Data to Justify Portfolio Decisions* — Roman Shevchenko

An empirical implementation of the **Markowitz Mean-Variance framework** for portfolio optimization. The tool downloads historical market data, computes risk/return statistics, and finds optimal portfolio weights using convex optimization — all wrapped in an interactive **Streamlit** web interface.

---

## 🚀 Live Demo

**Try the app instantly — no installation required:**

🔗 **[thesisshevchenko.streamlit.app](https://thesisshevchenko.streamlit.app)**

---

## 📊 Key Results (Empirical Study 2013–2022)

The algorithm was validated on a 5-asset, 10-year dataset across different asset classes:

| Portfolio | Annual Volatility | Sharpe Ratio | vs. Benchmark (S&P 500) |
|---|---|---|---|
| Equal-weight (naive) | ~14% | ~0.31 | Baseline |
| **Minimum-Variance** | **8.81%** | 0.49 | ✅ Lower risk |
| **Maximum-Sharpe** | ~11% | **0.651** | ✅ +75% Sharpe vs benchmark (0.370) |

> Mathematical optimization significantly outperformed naive equal-weight diversification across all risk-adjusted metrics.

---

## ✨ Features

- **Historical Data Pipeline** — Fetches adjusted close prices from Yahoo Finance via `yfinance` and computes periodic returns
- **Risk & Return Metrics** — Annualized returns, volatility, Covariance/Correlation matrices, Sharpe Ratio, and historical Value at Risk (VaR at 95%)
- **Portfolio Optimization** — Uses `scipy.optimize` (SLSQP) to compute:
  - 📌 Minimum-Variance Portfolio
  - 📌 Maximum-Sharpe (Tangency) Portfolio
  - 📌 Efficient Frontier (full Pareto-optimal curve)
- **Custom Constraints** — Set a maximum weight per asset to enforce realistic diversification
- **Benchmark Comparison** — Compares optimal portfolios against a market index (e.g., S&P 500)
- **Interactive Visualizations** — Efficient Frontier plot, correlation heatmap, price history charts

---

## 🧮 Methodology

The project implements the classical **Markowitz Mean-Variance** optimization model (1952):

**Portfolio variance** is minimized using the quadratic form:
```
σ²_p = wᵀ · Σ · w
```
where `w` is the vector of asset weights and `Σ` is the annualized covariance matrix.

**Sharpe Ratio** (risk-adjusted return):
```
S = (E[Rp] - Rf) / σp
```

**Optimization** is performed via Sequential Least Squares Programming (SLSQP) with constraints:
- Sum of weights = 1.0
- Each weight ∈ [0, max_weight]

The **Efficient Frontier** is constructed by solving the optimization problem across a range of target returns, producing the full set of Pareto-optimal portfolios.

---

## 🛠️ Technologies

| Category | Stack |
|---|---|
| Language | Python 3.9+ |
| Data Acquisition | `yfinance` |
| Data Processing | `pandas`, `numpy` |
| Optimization | `scipy.optimize` (SLSQP) |
| Visualization | `matplotlib`, `seaborn` |
| Web Interface | `Streamlit` |

---

## 🚀 Quick Start

**On Windows** — double-click `run.bat`

**On Mac / Linux:**
```bash
bash run.sh
```

*The script auto-installs all dependencies and opens the app in your browser.*

### Manual Installation

```bash
git clone https://github.com/SheroTheCreator/Diploma.git
cd Diploma
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.9+

---

## 📁 Project Structure

```
Diploma/
├── app.py                    # Streamlit Web Application (entry point)
├── run.bat                   # Auto-launcher for Windows
├── run.sh                    # Auto-launcher for Mac/Linux
├── requirements.txt          # Python dependencies
├── utils/
│   └── config.py             # Global config (dates, risk-free rate)
└── services/
    ├── data_fetcher.py        # Downloads data via yfinance
    ├── data_preprocessor.py  # Cleans prices, computes returns
    ├── statistics.py         # Annualized returns, volatility, correlation
    ├── risk_metrics.py       # Sharpe Ratio, historical VaR
    ├── optimizers.py         # Markowitz optimization engine (SLSQP)
    └── visualization.py      # Efficient Frontier & heatmap charts
```

---

## 🔮 Future Improvements

- [ ] Black-Litterman model integration
- [ ] Monte Carlo simulation for return forecasting
- [ ] Multi-period rebalancing strategy
- [ ] Export results to PDF/Excel report
- [ ] Docker containerization for one-click deployment

---

## 📄 License

This project is licensed under the MIT License.
