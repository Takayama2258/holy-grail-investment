"""
Holy Grail Scanner — Linda Raschke's trend-continuation pullback strategy.

Conditions (daily bars, per ticker):
  1. ADX(14) > 30              → strong trend
  2. Price pulls back to 20-EMA (touch or within 1%)
  3. ADX trending up recently  (ADX now > ADX 5 bars ago)
  4. Direction: bullish if close > EMA, bearish otherwise
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta  # noqa: F401 — imported for df.ta accessor
except ImportError:
    raise ImportError("pandas_ta is required. Install with: pip install pandas_ta")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S&P 500 default universe (top ~100 liquid names)
# ---------------------------------------------------------------------------
SP500_TICKERS: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
    "AMAT", "AMD", "AMGN", "AMZN", "ANET", "APH", "AVGO", "AXP", "BA", "BAC",
    "BDX", "BK", "BKNG", "BLK", "BMY", "BSX", "BX", "C", "CAT", "CCI",
    "CDNS", "CI", "CL", "CMCSA", "CME", "COF", "COP", "COST", "CRM", "CSCO",
    "CTAS", "CVX", "D", "DE", "DHR", "DIS", "DUK", "ECL", "EL", "EMR",
    "EQIX", "ETN", "EW", "F", "FDX", "FI", "GD", "GE", "GILD", "GM",
    "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "ICE", "INTC", "INTU", "ISRG",
    "ITW", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MCHP", "MCO", "MDLZ", "MDT", "MET", "META", "MMC", "MO", "MRK", "MS",
    "MSFT", "MSI", "MU", "NEE", "NFLX", "NKE", "NOW", "NSC", "NVDA", "ORCL",
    "PEP", "PFE", "PG", "PGR", "PM", "PNC", "PYPL", "QCOM", "REGN", "RTX",
    "SBUX", "SCHW", "SHW", "SLB", "SNPS", "SO", "SPG", "SYK", "T", "TDG",
    "TGT", "TJX", "TMO", "TMUS", "TRV", "TSLA", "TXN", "UNH", "UNP", "UPS",
    "USB", "V", "VLO", "VRTX", "VZ", "WBA", "WFC", "WM", "WMT", "XOM", "ZTS",
]


@dataclass
class HolyGrailMatch:
    ticker: str
    company_name: str
    price: float
    ema20: float
    adx14: float
    direction: str  # "Bullish" or "Bearish"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_company_name(ticker: str) -> str:
    """Return the long/short company name for a ticker, or the ticker itself."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _check_ticker(ticker: str, interval: str = "1d") -> Optional[HolyGrailMatch]:
    """Download data for *ticker* and test the Holy Grail conditions.

    Returns a ``HolyGrailMatch`` if conditions are met, else ``None``.
    """
    try:
        df: pd.DataFrame = yf.download(
            ticker, period="6mo", interval=interval, auto_adjust=True, progress=False
        )
        if df.empty or len(df) < 30:
            return None

        # Flatten MultiIndex columns that yfinance sometimes creates
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # --- indicators ---
        df.ta.ema(length=20, append=True)
        df.ta.adx(length=14, append=True)

        ema_col = "EMA_20"
        adx_col = "ADX_14"

        if ema_col not in df.columns or adx_col not in df.columns:
            return None

        df.dropna(subset=[ema_col, adx_col], inplace=True)
        if len(df) < 6:
            return None

        latest = df.iloc[-1]
        adx_now = float(latest[adx_col])
        ema_now = float(latest[ema_col])
        close = float(latest["Close"])
        high = float(latest["High"])
        low = float(latest["Low"])

        # Condition 1: ADX > 30
        if adx_now <= 30:
            return None

        # Condition 2: price touches or is within 1% of 20-EMA
        touch = low <= ema_now <= high
        within_pct = abs(close - ema_now) / ema_now < 0.01
        if not (touch or within_pct):
            return None

        # Condition 3: ADX trending up (now > 5 bars ago)
        adx_prev = float(df.iloc[-6][adx_col])
        if adx_now <= adx_prev:
            return None

        # Direction
        direction = "Bullish" if close > ema_now else "Bearish"

        company_name = _get_company_name(ticker)

        return HolyGrailMatch(
            ticker=ticker,
            company_name=company_name,
            price=round(close, 2),
            ema20=round(ema_now, 2),
            adx14=round(adx_now, 2),
            direction=direction,
        )
    except Exception:
        logger.exception("Error scanning %s", ticker)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_holy_grail(
    tickers: list[str] | None = None,
    interval: str = "1d",
    max_workers: int = 15,
) -> dict:
    """Scan *tickers* for the Holy Grail setup.

    Returns a dict with keys:
        results  — list of matching dicts
        scanned  — total tickers attempted
        errors   — count of tickers that raised exceptions
    """
    if tickers is None:
        tickers = SP500_TICKERS

    results: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check_ticker, t, interval): t for t in tickers}
        for future in as_completed(futures):
            try:
                match = future.result()
                if match is not None:
                    results.append(asdict(match))
            except Exception:
                errors += 1

    # Sort alphabetically by ticker
    results.sort(key=lambda r: r["ticker"])

    return {
        "results": results,
        "scanned": len(tickers),
        "errors": errors,
    }
