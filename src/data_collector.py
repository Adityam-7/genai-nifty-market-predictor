"""
data_collector.py
-----------------
Yahoo Finance data collection & preprocessing pipeline for
Nifty50 (^NSEI) and Bank Nifty (^NSEBANK).

Features:
  - Real-time & historical OHLCV data via yfinance
  - Automatic data cleaning (missing values, outliers)
  - Multi-index (Nifty50 + BankNifty) parallel fetch
  - Caching to avoid redundant API calls
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Index tickers on NSE ──────────────────────────────────────────────────────
TICKERS = {
    "Nifty50":    "^NSEI",
    "BankNifty":  "^NSEBANK",
    "Nifty_IT":   "^CNXIT",
    "Nifty_Auto": "^CNXAUTO",
    "USD_INR":    "USDINR=X",
}

CACHE_DIR = Path("./data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class MarketDataCollector:
    """Collects, cleans, and structures market data for downstream pipelines."""

    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

    # ── Core Fetch ────────────────────────────────────────────────────────────
    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a ticker, with local cache.

        Args:
            ticker:   Yahoo Finance symbol, e.g. "^NSEI"
            period:   "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"
            interval: "1m","2m","5m","15m","30m","60m","1d","1wk","1mo"
            start:    "YYYY-MM-DD"  (overrides period if provided)
            end:      "YYYY-MM-DD"

        Returns:
            Cleaned OHLCV DataFrame with DatetimeIndex.
        """
        cache_key = self._cache_key(ticker, period, interval, start, end)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"[Cache HIT] {ticker}")
            return cached

        logger.info(f"[Fetching] {ticker} | period={period} interval={interval}")
        try:
            tkr = yf.Ticker(ticker)
            if start:
                raw = tkr.history(start=start, end=end, interval=interval)
            else:
                raw = tkr.history(period=period, interval=interval)

            if raw.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            df = self._clean(raw, ticker)
            self._save_cache(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: Optional[dict] = None,
        period: str = "6mo",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple indices simultaneously.

        Returns:
            Dict mapping friendly name → DataFrame.
        """
        tickers = tickers or TICKERS
        results = {}
        for name, symbol in tickers.items():
            df = self.fetch(symbol, period=period, interval=interval)
            if not df.empty:
                results[name] = df
                logger.info(f"  ✓ {name:12s} — {len(df)} rows")
            else:
                logger.warning(f"  ✗ {name:12s} — no data")
        return results

    def get_real_time_quote(self, ticker: str) -> dict:
        """Fetch latest quote data (price, change, volume)."""
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.fast_info
            hist = tkr.history(period="2d", interval="1d")

            prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else None
            last_close = hist["Close"].iloc[-1] if len(hist) >= 1 else None

            change = last_close - prev_close if (last_close and prev_close) else None
            pct_change = (change / prev_close * 100) if change else None

            return {
                "ticker":        ticker,
                "last_price":    round(last_close, 2) if last_close else None,
                "previous_close":round(prev_close, 2) if prev_close else None,
                "change":        round(change, 2) if change else None,
                "pct_change":    round(pct_change, 2) if pct_change else None,
                "volume":        getattr(info, "three_month_average_volume", None),
                "52w_high":      getattr(info, "year_high", None),
                "52w_low":       getattr(info, "year_low", None),
                "timestamp":     datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Real-time quote failed for {ticker}: {e}")
            return {}

    # ── Cleaning ──────────────────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Standardise and clean raw yfinance DataFrame.

        Steps:
          1. Rename columns → lowercase snake_case
          2. Drop unnecessary columns (Dividends, Stock Splits)
          3. Forward-fill then back-fill NaN OHLCV values
          4. Remove rows where Close == 0 (exchange-closed artefacts)
          5. Clip extreme outliers using IQR method on Close
          6. Compute log-returns, daily range, and volume ratio
        """
        df = df.copy()

        # 1. Standardise column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        core_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in core_cols if c in df.columns]]

        # 2. Remove zero/negative prices
        df = df[df["close"] > 0]

        # 3. Forward + backward fill missing values
        df = df.ffill().bfill()

        # 4. Outlier clipping on close price (IQR × 3)
        if len(df) > 20:
            q1, q3 = df["close"].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[
                (df["close"] >= q1 - 3 * iqr) &
                (df["close"] <= q3 + 3 * iqr)
            ]

        # 5. Derived features
        df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
        df["daily_range"]   = (df["high"] - df["low"]) / df["close"] * 100
        df["volume_ratio"]  = df["volume"] / df["volume"].rolling(20).mean()
        df["ticker"]        = ticker

        return df.dropna(subset=["log_return"])

    # ── Cache Helpers ─────────────────────────────────────────────────────────
    def _cache_key(self, *args) -> str:
        raw = "_".join(str(a) for a in args if a)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self.cache_ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    collector = MarketDataCollector()

    print("\n=== Fetching Nifty50 (6 months, daily) ===")
    nifty = collector.fetch("^NSEI", period="6mo", interval="1d")
    print(nifty.tail(5).to_string())

    print("\n=== Real-time Quote ===")
    quote = collector.get_real_time_quote("^NSEI")
    for k, v in quote.items():
        print(f"  {k:<20}: {v}")
