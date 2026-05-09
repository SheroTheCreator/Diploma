"""
Markowitz mean-variance portfolio optimization.
"""

from dataclasses import dataclass
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
    """Mean-variance portfolio optimizer (Markowitz model)."""

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        """Initialize optimizer with annualized return and covariance estimates."""
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_returns)

    def _portfolio_performance(self, weights: np.ndarray) -> tuple[float, float]:
        """Calculate portfolio expected return and volatility."""
        port_return = float(np.dot(weights, self.expected_returns.values))
        port_var = float(weights.T @ self.cov_matrix.values @ weights)
        return port_return, float(np.sqrt(max(port_var, 0.0)))

    def _get_bounds(self, constraints: WeightConstraints) -> tuple:
        """Create bounds tuple for scipy.optimize."""
        return tuple((constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets))

    def _validate_constraints(self, constraints: WeightConstraints) -> None:
        """Check that weight bounds are feasible given the number of assets."""
        n = self.n_assets
        if constraints.min_weight * n > 1.0 + 1e-9:
            raise ValueError(
                f"min_weight={constraints.min_weight} is incompatible: "
                f"minimum possible sum = {constraints.min_weight * n:.4f} > 1.0 "
                f"for {n} assets."
            )
        if constraints.max_weight * n < 1.0 - 1e-9:
            raise ValueError(
                f"max_weight={constraints.max_weight} is incompatible: "
                f"maximum possible sum = {constraints.max_weight * n:.4f} < 1.0 "
                f"for {n} assets."
            )

    def min_variance_portfolio(self, constraints: WeightConstraints) -> PortfolioResult:
        """Compute global minimum-variance portfolio."""
        self._validate_constraints(constraints)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        cov = self.cov_matrix.values

        res = minimize(
            fun=lambda w: float(w.T @ cov @ w),
            x0=initial_weights,
            method="SLSQP",
            bounds=self._get_bounds(constraints),
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            tol=1e-10,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not res.success:
            raise RuntimeError(f"Min-variance optimization failed: {res.message}")

        ret, vol = self._portfolio_performance(res.x)
        return PortfolioResult(weights=res.x, expected_return=ret, volatility=vol)

    def max_sharpe_portfolio(self, risk_free_rate: float, constraints: WeightConstraints) -> PortfolioResult:
        """Compute maximum Sharpe ratio portfolio (tangency portfolio)."""
        self._validate_constraints(constraints)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        mu = self.expected_returns.values
        cov = self.cov_matrix.values

        def negative_sharpe(w: np.ndarray) -> float:
            p_ret = float(np.dot(w, mu))
            p_vol = float(np.sqrt(max(w.T @ cov @ w, 0.0)))
            return 1e6 if p_vol == 0.0 else -((p_ret - risk_free_rate) / p_vol)

        res = minimize(
            fun=negative_sharpe,
            x0=initial_weights,
            method="SLSQP",
            bounds=self._get_bounds(constraints),
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            tol=1e-10,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not res.success:
            raise RuntimeError(f"Max-Sharpe optimization failed: {res.message}")

        ret, vol = self._portfolio_performance(res.x)
        return PortfolioResult(weights=res.x, expected_return=ret, volatility=vol)

    def efficient_frontier(self, n_points: int, constraints: WeightConstraints) -> list[PortfolioResult]:
        """Generate a set of optimal portfolios along the efficient frontier."""
        self._validate_constraints(constraints)
        min_var = self.min_variance_portfolio(constraints)
        target_returns = np.linspace(min_var.expected_return, self.expected_returns.max(), n_points)

        frontier = []
        mu = self.expected_returns.values
        cov = self.cov_matrix.values
        bounds = self._get_bounds(constraints)

        for target in target_returns:
            res = minimize(
                fun=lambda w: float(w.T @ cov @ w),
                x0=np.ones(self.n_assets) / self.n_assets,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                    {"type": "eq", "fun": lambda w, t=target: float(np.dot(w, mu) - t)},
                ],
                tol=1e-10,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

            if res.success:
                ret, vol = self._portfolio_performance(res.x)
                frontier.append(PortfolioResult(weights=res.x, expected_return=ret, volatility=vol))

        return frontier
