"""
Parse command-line arguments for portfolio analysis.
"""

import argparse


def parse_cli_arguments() -> argparse.Namespace:
    """Parse and return CLI arguments. Prompts interactively if tickers are missing."""
    parser = argparse.ArgumentParser(description="Portfolio Optimization Tool")
    
    parser.add_argument(
        "--tickers", 
        nargs="+", 
        help="List of asset tickers (e.g., SPY TLT GLD)."
    )
    
    parser.add_argument(
        "--disable-plots", 
        action="store_true", 
        help="Disable graphical output."
    )
    
    parser.add_argument(
        "--auto-dates", 
        action="store_true", 
        help="Use dynamic date range (last 10 years) instead of fixed period."
    )

    args = parser.parse_args()

    # Interactive fallback
    if not args.tickers:
        print("=" * 60)
        print(" PORTFOLIO OPTIMIZATION TOOL")
        print("=" * 60)
        print("\nEnter ticker symbols separated by spaces (e.g., SPY TLT GLD).")
        
        raw = input("Tickers: ").strip()
        if not raw:
            raise SystemExit("Error: At least one ticker must be provided.")
            
        args.tickers = raw.split()
        print(f"\n* Selected {len(args.tickers)} asset(s): {', '.join(args.tickers)}")

    return args
