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
        help="List of asset tickers (e.g., AAPL MSFT GOOGL)."
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
    
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="Max weight per asset (e.g. 0.4). Prevents 100% allocation to one asset."
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark ticker for comparison (default: SPY)."
    )

    args = parser.parse_args()

    # Interactive fallback
    if not args.tickers:
        print("=" * 60)
        print(" PORTFOLIO OPTIMIZATION TOOL")
        print("=" * 60)
        print("\nEnter ticker symbols separated by spaces (e.g., AAPL MSFT GOOGL).")
        
        raw = input("Tickers: ").strip()
        if not raw:
            raise SystemExit("Error: At least one ticker must be provided.")
            
        args.tickers = raw.split()
        print(f"\n* Selected {len(args.tickers)} asset(s): {', '.join(args.tickers)}")

    # Validation for constraints
    if args.max_weight <= 0 or args.max_weight > 1.0:
        raise SystemExit(f"Error: --max-weight ({args.max_weight}) must be between 0.0 and 1.0")
    
    min_possible_weight = 1.0 / len(args.tickers)
    if args.max_weight < min_possible_weight:
        raise SystemExit(
            f"Error: --max-weight ({args.max_weight}) is mathematically impossible for {len(args.tickers)} assets. "
            f"It must be at least {min_possible_weight:.3f} to sum up to 1.0"
        )

    return args
