"""
FastAPI server for the Holy Grail stock scanner.

Run with:
    python server.py
    # or: uvicorn server:app --reload
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from scanner import scan_holy_grail, SP500_TICKERS

app = FastAPI(title="Holy Grail Scanner")

# ---------------------------------------------------------------------------
# Simple in-memory cache (15-minute TTL)
# ---------------------------------------------------------------------------
_cache: dict[str, dict] = {}
CACHE_TTL = 15 * 60  # seconds


def _cache_key(interval: str, tickers_hash: int) -> str:
    return f"{interval}:{tickers_hash}"


def _get_cached(key: str) -> Optional[dict]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(key: str, data: dict) -> None:
    _cache[key] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the single-page frontend."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/scan")
async def scan(
    timeframe: str = Query("1d", regex="^(1d|1h)$"),
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
):
    """Run the Holy Grail scan.

    Query params:
        timeframe — "1d" (daily) or "1h" (hourly)
        tickers   — optional comma-separated list; defaults to S&P 500 subset
    """
    ticker_list = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else SP500_TICKERS
    )

    key = _cache_key(timeframe, hash(tuple(ticker_list)))
    cached = _get_cached(key)
    if cached:
        return JSONResponse(content=cached)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: scan_holy_grail(ticker_list, interval=timeframe)
    )

    payload = {
        **result,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    _set_cache(key, payload)
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
