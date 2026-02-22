"""
Handles user input and argument parsing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CliArguments:
    """
    Container for arguments.
    """
    tickers: List[str]
    asset_types: Optional[List[str]]
    disable_plots: bool
    auto_dates: bool


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the portfolio analysis CLI.
    
    Defines available command-line options for running the empirical study.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Portfolio Optimization Tool\n\n"
            "Empirical implementation of Markowitz mean-variance framework "
            "using historical market data. Computes optimal portfolios, "
            "risk metrics, and efficient frontier.\n\n"
            "Based on Modern Portfolio Theory (Section 2) with data from "
            "Yahoo Finance (Section 5.1)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=False,
        help=(
            "List of asset tickers as used on Yahoo Finance "
            "(e.g., SPY TLT GLD). If omitted, the program will ask interactively. "
            "Section 7.2: Selection of market and assets."
        ),
    )

    parser.add_argument(
        "--types",
        nargs="+",
        required=False,
        help=(
            "Optional list of asset types corresponding to tickers "
            "(e.g., equity bond commodity). "
            "Must have same length as --tickers if provided. "
            "Used for categorical analysis and reporting."
        ),
    )

    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help=(
            "Disable graphical output (only numeric results will be produced). "
            "Useful for automated runs or environments without display. "
            "Default: plots enabled (Section 5.5)."
        ),
    )

    parser.add_argument(
        "--auto-dates",
        action="store_true",
        help=(
            "Use dynamic date range (current date minus 10 years) instead of "
            "fixed period 2013-2022. Useful for keeping analysis always current, "
            "but reduces reproducibility (Section 6.2, Section 7.1)."
        ),
    )

    return parser


def parse_cli_arguments() -> CliArguments:

    parser = build_arg_parser()
    args = parser.parse_args()

    tickers: List[str] = args.tickers if args.tickers is not None else []
    asset_types: Optional[List[str]] = args.types
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

    # Validate asset_types length if provided
    if asset_types is not None and len(asset_types) != len(tickers):
        parser.error(
            f"--types length ({len(asset_types)}) must match "
            f"--tickers length ({len(tickers)})"
        )

    return CliArguments(
        tickers=tickers,
        asset_types=asset_types,
        disable_plots=disable_plots,
        auto_dates=auto_dates,
    )
