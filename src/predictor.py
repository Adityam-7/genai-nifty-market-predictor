"""
predictor.py
------------
Core GenAI prediction engine for Nifty50 and Bank Nifty.

Pipeline:
  1. Fetch OHLCV data              (DataCollector)
  2. Compute technical indicators  (TechnicalIndicators)
  3. Retrieve market news context  (FinancialNewsRAG)
  4. Build structured prompt       (PromptBuilder)
  5. Call GPT-4o for prediction    (OpenAI)
  6. Parse & return PredictionResult

Prediction output:
  • Direction     : BULLISH / BEARISH / NEUTRAL
  • Price targets  : Next 1-day, 3-day, 5-day
  • Confidence     : 0.0 – 1.0
  • Key reasons    : Ranked list of supporting factors
  • Risk factors   : Potential invalidation scenarios
  • Trade setup    : Entry, Stop Loss, Target (optional)
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Structured output from a single market prediction call."""
    index:            str
    direction:        str                     # BULLISH | BEARISH | NEUTRAL
    confidence:       float                   # 0.0 - 1.0
    price_1d:         Optional[float] = None  # 1-day predicted CLOSE
    price_3d:         Optional[float] = None
    price_5d:         Optional[float] = None
    day_high_1d:      Optional[float] = None  # predicted intraday HIGH tomorrow
    day_low_1d:       Optional[float] = None  # predicted intraday LOW tomorrow
    day_high_3d:      Optional[float] = None
    day_low_3d:       Optional[float] = None
    day_high_5d:      Optional[float] = None
    day_low_5d:       Optional[float] = None
    support_levels:   list[float] = field(default_factory=list)
    resistance_levels:list[float] = field(default_factory=list)
    key_reasons:      list[str]   = field(default_factory=list)
    risk_factors:     list[str]   = field(default_factory=list)
    trade_setup:      dict        = field(default_factory=dict)
    sentiment:        str         = "NEUTRAL"  # POSITIVE | NEGATIVE | NEUTRAL
    technical_score:  float       = 0.0        # composite indicator score
    model_used:       str         = "gpt-4o"
    generated_at:     str         = field(default_factory=lambda: datetime.now().isoformat())
    raw_analysis:     str         = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        direction_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(self.direction, "❓")
        conf_pct = round(self.confidence * 100, 1)
        lines = [
            f"{direction_emoji} **{self.index}** — {self.direction} ({conf_pct}% confidence)",
            f"",
            f"**Price Targets:**",
            f"  1-Day: ₹{self.price_1d:,.0f}" if self.price_1d else "  1-Day: N/A",
            f"  3-Day: ₹{self.price_3d:,.0f}" if self.price_3d else "  3-Day: N/A",
            f"  5-Day: ₹{self.price_5d:,.0f}" if self.price_5d else "  5-Day: N/A",
        ]
        if self.trade_setup:
            lines += [
                f"",
                f"**Trade Setup:**",
                f"  Entry : ₹{self.trade_setup.get('entry', 'N/A')}",
                f"  Target: ₹{self.trade_setup.get('target', 'N/A')}",
                f"  SL    : ₹{self.trade_setup.get('stop_loss', 'N/A')}",
            ]
        if self.key_reasons:
            lines += ["", "**Key Reasons:**"] + [f"  • {r}" for r in self.key_reasons[:5]]
        if self.risk_factors:
            lines += ["", "**Risk Factors:**"] + [f"  ⚠️ {r}" for r in self.risk_factors[:3]]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
class MarketPredictor:
    """
    End-to-end GenAI prediction pipeline for Nifty50 and Bank Nifty.
    """

    SYSTEM_PROMPT = """You are an expert quantitative analyst and AI trading strategist 
specialising in Indian equity index markets (NSE Nifty50 and Bank Nifty).

You have deep expertise in:
- Technical analysis (moving averages, RSI, MACD, Bollinger Bands, ADX, Supertrend)
- Market microstructure (support/resistance, pivot points, volume analysis)
- Macro fundamentals (RBI policy, FII/DII flows, global cues, USD/INR)
- Options market (PCR, max pain, IV percentile)
- Sentiment analysis from financial news

You must provide precise, data-driven market predictions with clear reasoning.
Always respond in valid JSON format as specified.
"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model   = model
        self._client = None

        self._init_openai()

    def _init_openai(self) -> None:
        if not self.api_key:
            logger.warning("No OpenAI API key found. Predictions will use demo mode.")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialised — model: {self.model}")
        except ImportError:
            logger.warning("openai package not installed.")

    # ── Main Prediction Method ────────────────────────────────────────────────
    def predict(
        self,
        index_name: str,
        indicator_summary: dict,
        news_context: str,
        current_price: float,
        lookback_stats: Optional[dict] = None,
    ) -> PredictionResult:
        """
        Generate a market prediction for the given index.

        Args:
            index_name:        "Nifty50" or "BankNifty"
            indicator_summary: Output from TechnicalIndicators.get_signal_summary()
            news_context:      RAG-retrieved financial news synthesis
            current_price:     Latest closing price
            lookback_stats:    Optional dict with historical stats (vol, returns)

        Returns:
            PredictionResult dataclass
        """
        prompt = self._build_prompt(
            index_name, indicator_summary, news_context,
            current_price, lookback_stats
        )

        if self._client is None:
            logger.info("Using demo prediction (no OpenAI key)")
            return self._demo_prediction(index_name, indicator_summary, current_price)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.15,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            return self._parse_response(raw_text, index_name, indicator_summary)

        except Exception as e:
            logger.error(f"OpenAI prediction failed: {e}")
            return self._demo_prediction(index_name, indicator_summary, current_price)

    # ── Prompt Builder ────────────────────────────────────────────────────────
    def _build_prompt(
        self,
        index_name: str,
        ind: dict,
        news: str,
        price: float,
        stats: Optional[dict],
    ) -> str:

        tech_block = f"""
TECHNICAL INDICATORS (Latest Values):
  Price Action:
    Current Price : ₹{price:,.2f}
    RSI (14)      : {ind.get('rsi', 'N/A')} — {ind.get('rsi_signal', '')}
    MACD          : {ind.get('macd', 'N/A')} | Signal: {ind.get('macd_signal', 'N/A')} | Hist: {ind.get('macd_hist', 'N/A')}
    ADX (14)      : {ind.get('adx', 'N/A')} (>25 = strong trend)
    CCI (20)      : {ind.get('cci', 'N/A')}
    Stochastic    : %K={ind.get('stoch_k', 'N/A')} | %D={ind.get('stoch_d', 'N/A')}
    Williams %R   : {ind.get('williams_r', 'N/A')}

  Trend:
    SMA 20/50/200 : {ind.get('sma_20','N/A')} / {ind.get('sma_50','N/A')} / {ind.get('sma_200','N/A')}
    EMA 9/21      : {ind.get('ema_9','N/A')} / {ind.get('ema_21','N/A')}

  Volatility:
    BB Upper/Mid/Lower : {ind.get('bb_upper','N/A')} / {ind.get('bb_mid','N/A')} / {ind.get('bb_lower','N/A')}
    BB Position        : {ind.get('bb_position_pct','N/A')}%
    ATR (14)           : {ind.get('atr','N/A')}

  Volume:
    VWAP (20)    : {ind.get('vwap','N/A')}
    MFI (14)     : {ind.get('mfi','N/A')}
    OBV Trend    : {ind.get('obv','N/A')}

  Key Levels:
    Pivot Point  : {ind.get('pivot','N/A')}
    R1 / R2      : {ind.get('r1','N/A')} / {ind.get('r2','N/A')}
    S1 / S2      : {ind.get('s1','N/A')} / {ind.get('s2','N/A')}

  Composite Signal Score : {ind.get('composite_score','N/A')} → {ind.get('signal','N/A')}
"""

        stats_block = ""
        if stats:
            stats_block = f"""
HISTORICAL STATISTICS:
  20-Day Historical Volatility : {stats.get('hist_vol','N/A')}%
  30-Day Return                : {stats.get('return_30d','N/A')}%
  Beta vs Nifty                : {stats.get('beta','N/A')}
  Average Volume Ratio         : {stats.get('avg_vol_ratio','N/A')}
"""

        return f"""
Analyse the following data for {index_name} and provide a detailed market prediction.

{tech_block}
{stats_block}

MARKET NEWS CONTEXT (RAG Retrieved):
{news}

Based on ALL the above data, provide your prediction in this EXACT JSON format:
{{
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "price_targets": {{
    "1_day_close": <predicted closing price as float>,
    "1_day_high":  <predicted intraday high as float>,
    "1_day_low":   <predicted intraday low as float>,
    "3_day_close": <predicted closing price as float>,
    "3_day_high":  <predicted intraday high as float>,
    "3_day_low":   <predicted intraday low as float>,
    "5_day_close": <predicted closing price as float>,
    "5_day_high":  <predicted intraday high as float>,
    "5_day_low":   <predicted intraday low as float>
  }},
  "support_levels": [<float>, <float>],
  "resistance_levels": [<float>, <float>],
  "trade_setup": {{
    "entry":     <price as float>,
    "target":    <price as float>,
    "stop_loss": <price as float>,
    "risk_reward_ratio": <float>
  }},
  "key_reasons": [
    "Reason 1 (specific, data-backed)",
    "Reason 2",
    "Reason 3",
    "Reason 4",
    "Reason 5"
  ],
  "risk_factors": [
    "Risk 1",
    "Risk 2",
    "Risk 3"
  ],
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "analysis": "<2-3 paragraph detailed analysis>"
}}

Be specific about price levels. Use the actual current price of ₹{price:,.2f} as base.
"""

    # ── Response Parser ───────────────────────────────────────────────────────
    def _parse_response(
        self, raw: str, index_name: str, ind: dict
    ) -> PredictionResult:
        try:
            data = json.loads(raw)
            pt = data.get("price_targets", {})
            ts = data.get("trade_setup", {})

            def _f(val):
                try:
                    v = float(val)
                    return v if v > 0 else None
                except (TypeError, ValueError):
                    return None

            return PredictionResult(
                index=index_name,
                direction=data.get("direction", "NEUTRAL"),
                confidence=float(data.get("confidence", 0.5)),
                price_1d=_f(pt.get("1_day_close")),
                price_3d=_f(pt.get("3_day_close")),
                price_5d=_f(pt.get("5_day_close")),
                day_high_1d=_f(pt.get("1_day_high")),
                day_low_1d=_f(pt.get("1_day_low")),
                day_high_3d=_f(pt.get("3_day_high")),
                day_low_3d=_f(pt.get("3_day_low")),
                day_high_5d=_f(pt.get("5_day_high")),
                day_low_5d=_f(pt.get("5_day_low")),
                support_levels=data.get("support_levels", []),
                resistance_levels=data.get("resistance_levels", []),
                key_reasons=data.get("key_reasons", []),
                risk_factors=data.get("risk_factors", []),
                trade_setup={
                    "entry":       ts.get("entry"),
                    "target":      ts.get("target"),
                    "stop_loss":   ts.get("stop_loss"),
                    "risk_reward": ts.get("risk_reward_ratio"),
                },
                sentiment=data.get("sentiment", "NEUTRAL"),
                technical_score=float(ind.get("composite_score", 0)),
                model_used=self.model,
                raw_analysis=data.get("analysis", ""),
            )
        except Exception as e:
            logger.error(f"Failed to parse prediction response: {e}\n{raw[:300]}")
            return PredictionResult(
                index=index_name,
                direction="NEUTRAL",
                confidence=0.5,
                raw_analysis=raw,
            )

    # ── Demo Mode (no API key) ────────────────────────────────────────────────
    def _demo_prediction(
        self, index_name: str, ind: dict, price: float
    ) -> PredictionResult:
        """
        Rule-based demo prediction for when OpenAI is not available.
        Uses composite_score from technical indicators.
        """
        score = ind.get("composite_score", 0) or 0
        rsi   = ind.get("rsi", 50) or 50
        macd_hist = ind.get("macd_hist", 0) or 0

        # Determine direction
        if score > 0.3:
            direction = "BULLISH"
            confidence = 0.55 + min(abs(score) * 0.3, 0.25)
        elif score < -0.3:
            direction = "BEARISH"
            confidence = 0.55 + min(abs(score) * 0.3, 0.25)
        else:
            direction = "NEUTRAL"
            confidence = 0.45 + (1 - abs(score)) * 0.1

        atr = ind.get("atr", price * 0.01) or price * 0.01
        multiplier = 1 if direction == "BULLISH" else -1

        # Predicted closing prices
        price_1d = round(price + multiplier * atr * 0.5, 0)
        price_3d = round(price + multiplier * atr * 1.2, 0)
        price_5d = round(price + multiplier * atr * 2.0, 0)

        # Predicted intraday high/low (close +/- fraction of ATR)
        day_high_1d = round(price_1d + atr * 0.6, 0)
        day_low_1d  = round(price_1d - atr * 0.6, 0)
        day_high_3d = round(price_3d + atr * 0.7, 0)
        day_low_3d  = round(price_3d - atr * 0.7, 0)
        day_high_5d = round(price_5d + atr * 0.8, 0)
        day_low_5d  = round(price_5d - atr * 0.8, 0)

        reasons = []
        if rsi > 60:
            reasons.append(f"RSI at {rsi:.1f} — showing bullish momentum")
        elif rsi < 40:
            reasons.append(f"RSI at {rsi:.1f} — bearish pressure continues")
        if macd_hist > 0:
            reasons.append("MACD histogram positive — upward momentum")
        elif macd_hist < 0:
            reasons.append("MACD histogram negative — downward momentum")
        if ind.get("sma_50") and price > ind["sma_50"]:
            reasons.append(f"Price above 50-SMA (₹{ind['sma_50']:,.0f}) — trend intact")
        reasons.append(
            f"Composite signal score: {score:.2f} → {ind.get('signal','N/A')}"
        )
        if ind.get("bb_position_pct"):
            reasons.append(f"BB position at {ind['bb_position_pct']}% of band range")

        return PredictionResult(
            index=index_name,
            direction=direction,
            confidence=round(confidence, 2),
            price_1d=price_1d,
            price_3d=price_3d,
            price_5d=price_5d,
            day_high_1d=day_high_1d,
            day_low_1d=day_low_1d,
            day_high_3d=day_high_3d,
            day_low_3d=day_low_3d,
            day_high_5d=day_high_5d,
            day_low_5d=day_low_5d,
            support_levels=[
                round(ind.get("s1", price * 0.98), 0),
                round(ind.get("s2", price * 0.96), 0),
            ],
            resistance_levels=[
                round(ind.get("r1", price * 1.02), 0),
                round(ind.get("r2", price * 1.04), 0),
            ],
            trade_setup={
                "entry":       round(price, 0),
                "target":      round(price_3d, 0),
                "stop_loss":   round(price - multiplier * atr * 1.5, 0),
                "risk_reward": round(abs(price_3d - price) / (atr * 1.5), 2),
            },
            sentiment="POSITIVE" if direction == "BULLISH" else
                      "NEGATIVE" if direction == "BEARISH" else "NEUTRAL",
            technical_score=score,
            model_used="demo-rule-based",
            key_reasons=reasons,
            risk_factors=[
                "Global volatility could override technical signals",
                "Watch for RBI/Fed announcements this week",
                "FII activity may shift direction rapidly",
            ],
            raw_analysis=(
                f"Demo mode: Prediction for {index_name} based purely on "
                f"technical indicators. Composite score: {score:.3f}. "
                f"Provide an OpenAI API key for full GenAI analysis."
            ),
        )
