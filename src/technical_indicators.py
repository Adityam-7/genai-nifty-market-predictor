"""
technical_indicators.py
-----------------------
Computes a rich set of technical indicators used as features
for the GenAI market predictor.

Indicator groups:
  • Trend      — SMA, EMA, MACD, ADX, Supertrend
  • Momentum   — RSI, Stochastic, Williams %R, CCI, ROC
  • Volatility — Bollinger Bands, ATR, Keltner Channel
  • Volume     — OBV, VWAP, MFI, CMF, Volume OSC
  • Pattern    — Pivot Points, Support/Resistance levels
  • Composite  — Buy/Sell signal score (−1 to +1)
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class TechnicalIndicators:
    """
    Stateless indicator engine.

    Usage:
        ti = TechnicalIndicators()
        enriched_df = ti.compute_all(df)
        summary = ti.get_signal_summary(enriched_df)
    """

    # ── Public API ────────────────────────────────────────────────────────────
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every indicator group and return enriched DataFrame."""
        df = df.copy()
        df = self._trend(df)
        df = self._momentum(df)
        df = self._volatility(df)
        df = self._volume(df)
        df = self._pivots(df)
        df = self._composite_signal(df)
        return df

    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """
        Return a human-readable dict of the latest indicator values
        and an overall signal (-1 Bearish → 0 Neutral → +1 Bullish).
        """
        if df.empty:
            return {}

        last = df.iloc[-1]

        def _safe(col, round_digits=2):
            v = last.get(col, np.nan)
            return round(float(v), round_digits) if pd.notna(v) else None

        # Determine RSI zone
        rsi = _safe("rsi")
        rsi_signal = "Overbought" if rsi and rsi > 70 else ("Oversold" if rsi and rsi < 30 else "Neutral")

        # Bollinger band position
        bb_pos = None
        close = _safe("close")
        bb_upper = _safe("bb_upper")
        bb_lower = _safe("bb_lower")
        if close and bb_upper and bb_lower:
            bb_range = bb_upper - bb_lower
            bb_pos = round((close - bb_lower) / bb_range * 100, 1) if bb_range else None

        return {
            "close":          close,
            "rsi":            rsi,
            "rsi_signal":     rsi_signal,
            "macd":           _safe("macd"),
            "macd_signal":    _safe("macd_signal"),
            "macd_hist":      _safe("macd_hist"),
            "adx":            _safe("adx"),
            "cci":            _safe("cci"),
            "stoch_k":        _safe("stoch_k"),
            "stoch_d":        _safe("stoch_d"),
            "bb_upper":       bb_upper,
            "bb_lower":       bb_lower,
            "bb_mid":         _safe("bb_mid"),
            "bb_position_pct":bb_pos,
            "atr":            _safe("atr"),
            "obv":            _safe("obv", 0),
            "mfi":            _safe("mfi"),
            "sma_20":         _safe("sma_20"),
            "sma_50":         _safe("sma_50"),
            "sma_200":        _safe("sma_200"),
            "ema_9":          _safe("ema_9"),
            "ema_21":         _safe("ema_21"),
            "vwap":           _safe("vwap"),
            "pivot":          _safe("pivot"),
            "r1":             _safe("r1"),
            "r2":             _safe("r2"),
            "s1":             _safe("s1"),
            "s2":             _safe("s2"),
            "composite_score":_safe("composite_score"),
            "signal":         last.get("signal", "NEUTRAL"),
        }

    # ── Trend Indicators ──────────────────────────────────────────────────────
    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h, l = df["high"], df["low"]

        # Moving Averages
        for n in [9, 20, 50, 200]:
            df[f"sma_{n}"] = c.rolling(n).mean()
        for n in [9, 21, 50]:
            df[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()

        # MACD (12, 26, 9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ADX
        df = self._adx(df)

        # Supertrend
        df = self._supertrend(df, period=10, multiplier=3.0)

        # Golden / Death Cross flag
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        return df

    # ── Momentum Indicators ───────────────────────────────────────────────────
    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l = df["close"], df["high"], df["low"]

        # RSI (14)
        df["rsi"] = self._rsi(c, period=14)

        # Stochastic (14, 3, 3)
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        k = (c - low14) / (high14 - low14 + 1e-9) * 100
        df["stoch_k"] = k.rolling(3).mean()
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R (14)
        df["williams_r"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

        # CCI (20)
        tp = (h + l + c) / 3
        df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # ROC (10)
        df["roc"] = c.pct_change(10) * 100

        return df

    # ── Volatility Indicators ─────────────────────────────────────────────────
    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l = df["close"], df["high"], df["low"]

        # Bollinger Bands (20, 2)
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_mid"]   = sma20
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20 * 100

        # ATR (14)
        df["atr"] = self._atr(df, period=14)

        # Keltner Channel (20, 1.5)
        df["kc_mid"]   = c.ewm(span=20, adjust=False).mean()
        df["kc_upper"] = df["kc_mid"] + 1.5 * df["atr"]
        df["kc_lower"] = df["kc_mid"] - 1.5 * df["atr"]

        # Historical Volatility (20-day annualised)
        log_ret = np.log(c / c.shift(1))
        df["hist_vol"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

        return df

    # ── Volume Indicators ─────────────────────────────────────────────────────
    def _volume(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # OBV
        obv = [0]
        for i in range(1, len(c)):
            if c.iloc[i] > c.iloc[i - 1]:
                obv.append(obv[-1] + v.iloc[i])
            elif c.iloc[i] < c.iloc[i - 1]:
                obv.append(obv[-1] - v.iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # VWAP (rolling 20-period approximation)
        tp = (h + l + c) / 3
        df["vwap"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()

        # MFI (14)
        mf_raw = tp * v
        pos_mf = mf_raw.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf_raw.where(tp < tp.shift(1), 0).rolling(14).sum()
        df["mfi"] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))

        # CMF (20)
        mfm = ((c - l) - (h - c)) / (h - l + 1e-9)
        df["cmf"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum()

        # Volume Oscillator (5-day vs 20-day EMA)
        df["vol_osc"] = (
            v.ewm(span=5, adjust=False).mean() /
            v.ewm(span=20, adjust=False).mean() - 1
        ) * 100

        return df

    # ── Pivot Points ──────────────────────────────────────────────────────────
    def _pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classic daily pivot points using previous bar's HLC."""
        ph = df["high"].shift(1)
        pl = df["low"].shift(1)
        pc = df["close"].shift(1)

        df["pivot"] = (ph + pl + pc) / 3
        df["r1"]    = 2 * df["pivot"] - pl
        df["s1"]    = 2 * df["pivot"] - ph
        df["r2"]    = df["pivot"] + (ph - pl)
        df["s2"]    = df["pivot"] - (ph - pl)
        df["r3"]    = ph + 2 * (df["pivot"] - pl)
        df["s3"]    = pl - 2 * (ph - df["pivot"])

        return df

    # ── Composite Signal ──────────────────────────────────────────────────────
    def _composite_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted aggregation of indicator signals into a single score.
        Score range: -1.0 (strong sell) → +1.0 (strong buy)
        """
        scores = pd.DataFrame(index=df.index)

        # RSI: >70 → -1, <30 → +1, else scaled
        scores["rsi_s"] = df["rsi"].apply(
            lambda x: -1 if x > 70 else (1 if x < 30 else (50 - x) / 50)
            if pd.notna(x) else 0
        )

        # MACD histogram direction
        scores["macd_s"] = df["macd_hist"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            if pd.notna(x) else 0
        )

        # Price vs SMA50 and SMA200
        scores["sma_s"] = (
            (df["close"] > df["sma_50"]).astype(float) +
            (df["close"] > df["sma_200"]).astype(float)
        ) / 2 * 2 - 1   # maps to [-1, +1]

        # Stochastic
        scores["stoch_s"] = df["stoch_k"].apply(
            lambda x: -1 if x > 80 else (1 if x < 20 else 0)
            if pd.notna(x) else 0
        )

        # ADX trend strength (only enhances directional signals)
        adx_weight = df["adx"].fillna(0).clip(0, 50) / 50

        # Bollinger Band squeeze
        c = df["close"]
        scores["bb_s"] = np.where(c > df["bb_upper"], -0.5,
                         np.where(c < df["bb_lower"],  0.5, 0))

        weights = {"rsi_s": 0.25, "macd_s": 0.30, "sma_s": 0.20,
                   "stoch_s": 0.15, "bb_s": 0.10}

        composite = sum(scores[k] * w for k, w in weights.items())
        # Amplify by ADX strength
        df["composite_score"] = (composite * (0.5 + 0.5 * adx_weight)).clip(-1, 1).round(3)

        df["signal"] = df["composite_score"].apply(
            lambda x: "STRONG BUY"  if x >  0.5  else
                      "BUY"         if x >  0.2  else
                      "NEUTRAL"     if x >= -0.2 else
                      "SELL"        if x >= -0.5 else
                      "STRONG SELL"
        )
        return df

    # ── Helper: RSI ───────────────────────────────────────────────────────────
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    # ── Helper: ATR ───────────────────────────────────────────────────────────
    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    # ── Helper: ADX ───────────────────────────────────────────────────────────
    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        h, l, c = df["high"], df["low"], df["close"]
        atr = self._atr(df, period)

        up   = h - h.shift(1)
        down = l.shift(1) - l

        plus_dm  = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)

        plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df["adx"]       = dx.ewm(span=period, adjust=False).mean()
        df["plus_di"]   = plus_di
        df["minus_di"]  = minus_di

        return df

    # ── Helper: Supertrend ────────────────────────────────────────────────────
    def _supertrend(
        self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        hl2  = (df["high"] + df["low"]) / 2
        atr  = self._atr(df, period)
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = [np.nan] * len(df)
        direction  = [1] * len(df)

        for i in range(1, len(df)):
            prev_close = df["close"].iloc[i - 1]
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            supertrend[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]

        df["supertrend"]   = supertrend
        df["st_direction"] = direction
        return df
