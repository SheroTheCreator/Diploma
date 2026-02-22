"""
Handles user input and argument parsing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class CliArguments:
    """Container for arguments."""
    tickers: List[str]
    disable_plots: bool
    auto_dates: bool


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for the portfolio analysis CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Portfolio Optimization Tool\n\n"
            "Empirical implementation of Markowitz mean-variance framework "
            "using historical market data. Computes optimal portfolios, "
            "risk metrics, and efficient frontier.\n\n"
            "Based on Modern Portfolio Theory with data from Yahoo Finance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=False,
        help="List of asset tickers as used on Yahoo Finance (e.g., SPY TLT GLD).",
    )

    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Disable graphical output (only numeric results will be produced).",
    )

    parser.add_argument(
        "--auto-dates",
        action="store_true",
        help="Use dynamic date range (current date minus 10 years) instead of fixed period.",
    )

    return parser


def parse_cli_arguments() -> CliArguments:
    """Parse CLI arguments and handle interactive fallback if needed."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tickers: List[str] = args.tickers if args.tickers is not None else []
    disable_plots: bool = bool(args.disable_plots)
    auto_dates: bool = bool(args.auto_dates)

    # Interactive input if no tickers provided
    if not tickers:
        print("=" * 70)
        print("  Portfolio Optimization Tool")
        print("  Markowitz Mean-Variance Analysis")
        print("=" * 70)
        print("\nEnter ticker symbols separated by spaces.")
        print("Examples:")
        print("  - Conservative: SPY TLT GLD")
        print("  - Growth: SPY QQQ EEM")
        print("  - Two-asset: SPY TLT")
        print("\n(Tickers must be available on Yahoo Finance)")
        print()
        
        raw = input("Your tickers: ").strip()
        if not raw:
            print("\nError: At least one ticker must be provided.")
            raise SystemExit(1)
        
        tickers = raw.split()
        print(f"\n* Selected {len(tickers)} asset(s): {', '.join(tickers)}")

    return CliArguments(
        tickers=tickers,
        disable_plots=disable_plots,
        auto_dates=auto_dates,
    )
