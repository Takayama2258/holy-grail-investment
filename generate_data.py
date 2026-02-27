"""
Generate static scan results for GitHub Pages deployment.

Runs the Holy Grail scanner and writes results to data.json,
which the static frontend loads on GitHub Pages.

Usage:
    python generate_data.py
    python generate_data.py --interval 1h
"""

import argparse
import json
from datetime import datetime, timezone

from scanner import scan_holy_grail


def main():
    parser = argparse.ArgumentParser(description="Generate Holy Grail scan data")
    parser.add_argument(
        "--interval", default="1d", choices=["1d", "1h"],
        help="Candle interval (default: 1d)",
    )
    parser.add_argument(
        "--output", default="data.json",
        help="Output file path (default: data.json)",
    )
    args = parser.parse_args()

    print(f"Running Holy Grail scan (interval={args.interval})...")
    result = scan_holy_grail(interval=args.interval)

    payload = {
        **result,
        "interval": args.interval,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    n = len(result["results"])
    print(f"Done — {n} setups found, {result['errors']} errors. Written to {args.output}")


if __name__ == "__main__":
    main()
