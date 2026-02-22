"""
Portfolio optimization using Markowitz mean-variance framework.

Implements the optimization problems from the Markowitz model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class WeightConstraints:
    """Box constraints on portfolio weights."""
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioResult:
    """Container for optimized portfolio characteristics."""
    weights: np.ndarray
    expected_return: float
    volatility: float


class MeanVarianceOptimizer:
    """
    Mean-variance portfolio optimizer (Markowitz model).
    
    Solves quadratic optimization problems to find optimal portfolio weights
    that balance expected return and risk (variance).
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> None:
        """Initialize optimizer with annualized return and covariance estimates."""
        if not isinstance(expected_returns, pd.Series):
            raise TypeError("expected_returns must be a pandas Series.")
        if not isinstance(cov_matrix, pd.DataFrame):
            raise TypeError("cov_matrix must be a pandas DataFrame.")

        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.tickers: List[str] = list(expected_returns.index)
        self.n_assets: int = len(self.tickers)

    # ---------- Core helpers ----------

    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio expected return and volatility."""
        mu = self.expected_returns.values  # shape (n_assets,)
        cov = self.cov_matrix.values       # shape (n_assets, n_assets)

        port_return = float(np.dot(weights, mu))
        port_var = float(weights.T @ cov @ weights)
        port_vol = float(np.sqrt(port_var))
        
        return port_return, port_vol

    def _build_bounds(self, constraints: WeightConstraints) -> Tuple[Tuple[float, float], ...]:
        """Create bounds tuple for scipy.optimize from WeightConstraints."""
        return tuple((constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets))

    def _weight_sum_constraint(self) -> dict:
        """Constraint enforcing sum of weights equal to 1."""
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # ---------- Public optimization methods ----------

    def min_variance_portfolio(
        self,
        constraints: WeightConstraints,
        initial_weights: np.ndarray | None = None,
    ) -> PortfolioResult:
        """
        Compute global minimum-variance portfolio.
        
        This portfolio has the lowest possible risk on the efficient frontier.
        """
        if initial_weights is None:
            initial_weights = np.repeat(1.0 / self.n_assets, self.n_assets)

        cov = self.cov_matrix.values

        def objective(w: np.ndarray) -> float:
            # Minimize portfolio variance
            return float(w.T @ cov @ w)

        bounds = self._build_bounds(constraints)
        cons = [self._weight_sum_constraint()]

        result = minimize(
            fun=objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            raise RuntimeError(f"Min-variance optimization failed: {result.message}")

        w_opt = result.x
        ret, vol = self._portfolio_performance(w_opt)
        return PortfolioResult(weights=w_opt, expected_return=ret, volatility=vol)

    def max_sharpe_portfolio(
        self,
        risk_free_rate: float,
        constraints: WeightConstraints,
        initial_weights: np.ndarray | None = None,
    ) -> PortfolioResult:
        """
        Compute maximum Sharpe ratio portfolio.
        
        This portfolio achieves the best risk-adjusted return (tangency portfolio).
        """
        if initial_weights is None:
            initial_weights = np.repeat(1.0 / self.n_assets, self.n_assets)

        mu = self.expected_returns.values
        cov = self.cov_matrix.values

        def negative_sharpe(w: np.ndarray) -> float:
            port_ret = float(np.dot(w, mu))
            port_var = float(w.T @ cov @ w)
            port_vol = float(np.sqrt(port_var))
            
            if port_vol == 0.0:
                return 1e6  # Penalize zero volatility
            
            sharpe = (port_ret - risk_free_rate) / port_vol
            return float(-sharpe)

        bounds = self._build_bounds(constraints)
        cons = [self._weight_sum_constraint()]

        result = minimize(
            fun=negative_sharpe,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")

        w_opt = result.x
        ret, vol = self._portfolio_performance(w_opt)
        return PortfolioResult(weights=w_opt, expected_return=ret, volatility=vol)

    def efficient_frontier(
        self,
        n_points: int,
        constraints: WeightConstraints,
    ) -> List[PortfolioResult]:
        """
        Compute efficient frontier.
        
        Generates a set of portfolios with varying target returns,
        each minimizing risk for that return level.
        """
        min_var_port = self.min_variance_portfolio(constraints)
        min_ret = min_var_port.expected_return
        max_ret = self.expected_returns.max()

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_portfolios: List[PortfolioResult] = []

        for target_ret in target_returns:
            try:
                port = self._efficient_portfolio_at_return(target_ret, constraints)
                frontier_portfolios.append(port)
            except RuntimeError:
                # Skip infeasible target returns
                continue

        return frontier_portfolios

    def _efficient_portfolio_at_return(
        self,
        target_return: float,
        constraints: WeightConstraints,
    ) -> PortfolioResult:
        """Find minimum-variance portfolio with specified target return."""
        initial_weights = np.repeat(1.0 / self.n_assets, self.n_assets)
        mu = self.expected_returns.values
        cov = self.cov_matrix.values

        def objective(w: np.ndarray) -> float:
            return float(w.T @ cov @ w)

        def return_constraint(w: np.ndarray) -> float:
            return float(np.dot(w, mu) - target_return)

        bounds = self._build_bounds(constraints)
        cons = [
            self._weight_sum_constraint(),
            {"type": "eq", "fun": return_constraint}
        ]

        result = minimize(
            fun=objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed for target return {target_return:.4f}")

        w_opt = result.x
        ret, vol = self._portfolio_performance(w_opt)
        return PortfolioResult(weights=w_opt, expected_return=ret, volatility=vol)
