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
import sys
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

    print(f"Running Holy Grail scan (interval={args.interval})...", flush=True)
    result = scan_holy_grail(interval=args.interval)

    payload = {
        **result,
        "interval": args.interval,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    n = len(result["results"])
    print(f"Done — {n} setups found, {result['errors']} errors. Written to {args.output}", flush=True)

    if n > 0:
        print("Matches:", flush=True)
        for r in result["results"]:
            print(f"  {r['ticker']:6s}  ADX={r['adx14']:.1f}  EMA={r['ema20']:.2f}  "
                  f"Price={r['price']:.2f}  {r['direction']}", flush=True)


if __name__ == "__main__":
    main()
