# 📊 GenAI-Driven Nifty50 & Bank Nifty Market Predictor

> A stock market prediction system that forecasts Nifty50 and Bank Nifty 
> movements using Yahoo Finance, LangChain RAG, and OpenAI GPT-4o.
> Built as a personal project to combine technical analysis with GenAI.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GenAI Market Predictor                        │
├──────────────┬──────────────────┬─────────────┬────────────────┤
│  Data Layer  │  Feature Layer   │  RAG Layer  │  GenAI Layer   │
│              │                  │             │                │
│  Yahoo       │  TechnicalInd.   │  RSS Feeds  │  OpenAI        │
│  Finance     │  ─────────────   │  ─────────  │  GPT-4o        │
│  API         │  • RSI           │  ET Markets │                │
│  (yfinance)  │  • MACD          │  MoneyCtrl  │  Structured    │
│              │  • Bollinger     │  LiveMint   │  JSON Output:  │
│  OHLCV Data  │  • ADX           │  ─────────  │  • Direction   │
│  Real-time   │  • Stochastic    │  LangChain  │  • Confidence  │
│  Historical  │  • Supertrend    │  ─────────  │  • Targets     │
│              │  • ATR / OBV     │  FAISS      │  • Trade Setup │
│  Pandas/     │  • Pivot Points  │  MMR Search │  • Reasons     │
│  NumPy       │  • 15+ more…     │  GPT Synth  │  • Risks       │
│  Pipeline    │                  │             │                │
└──────────────┴──────────────────┴─────────────┴────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Streamlit Dashboard   │
              │  • Candlestick Chart    │
              │  • Indicator Overlays   │
              │  • AI Prediction Card   │
              │  • News Headlines       │
              │  • Trade Setup          │
              └─────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Adityam-7/nifty-predictor.git
cd nifty-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-...
```

### 3. Run

**CLI (quick prediction):**
```bash
python main.py --index Nifty50 --period 6mo
python main.py --all                         # Both indices
python main.py --demo                        # No API key needed
```

**Streamlit Dashboard:**
```bash
streamlit run app.py
# or
python main.py --ui
```

---

## 📁 Project Structure

```
nifty-predictor/
├── app.py                    # Streamlit web dashboard
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_collector.py     # Yahoo Finance pipeline (yfinance)
│   ├── technical_indicators.py  # 20+ indicators (RSI, MACD, BB…)
│   ├── rag_pipeline.py       # LangChain + FAISS news RAG
│   └── predictor.py          # GPT-4o prediction engine
│
├── data/
│   ├── cache/                # Parquet cache for market data
│   └── chroma_db/            # ChromaDB vector store (optional)
│
└── logs/
    └── predictor.log
```

---

## 🧩 Components

### 1. `data_collector.py` — Yahoo Finance Pipeline

```python
from src.data_collector import MarketDataCollector

collector = MarketDataCollector(cache_ttl_minutes=15)

# Fetch historical OHLCV
df = collector.fetch("^NSEI", period="6mo", interval="1d")

# Fetch both indices
data = collector.fetch_multiple()

# Real-time quote
quote = collector.get_real_time_quote("^NSEI")
# → {'last_price': 24823.45, 'pct_change': -0.42, ...}
```

**Data cleaning pipeline:**
- Forward/backward fill NaN values
- Remove zero-price artefacts (exchange closures)
- IQR-based outlier clipping (3× IQR)
- Parquet-based caching with TTL

---

### 2. `technical_indicators.py` — Feature Engineering

| Group | Indicators |
|-------|-----------|
| **Trend** | SMA (20/50/200), EMA (9/21/50), MACD, ADX, Supertrend, Golden Cross |
| **Momentum** | RSI (14), Stochastic (%K/%D), Williams %R, CCI, ROC |
| **Volatility** | Bollinger Bands, ATR, Keltner Channel, Historical Volatility |
| **Volume** | OBV, VWAP, MFI, CMF, Volume Oscillator |
| **Levels** | Classic Pivot Points (P, R1, R2, R3, S1, S2, S3) |
| **Composite** | Weighted signal score (−1.0 → +1.0) |

```python
from src.technical_indicators import TechnicalIndicators

ti = TechnicalIndicators()
enriched_df = ti.compute_all(df)
summary = ti.get_signal_summary(enriched_df)
# → {'rsi': 58.3, 'macd': 24.5, 'signal': 'BUY', 'composite_score': 0.42, ...}
```

---

### 3. `rag_pipeline.py` — LangChain RAG

```python
from src.rag_pipeline import FinancialNewsRAG

rag = FinancialNewsRAG(openai_api_key="sk-...")

# Get synthesised market context for prediction
context = rag.get_market_context("Nifty50")

# Get raw headlines
headlines = rag.get_headlines(10)
```

**RAG Architecture:**
**How the RAG pipeline works:**
- Pulls latest financial news from 5 RSS feeds (no API key needed)
- Chunks and embeds using OpenAI's text-embedding-3-small
- FAISS vector store with MMR retrieval to avoid redundant results
- GPT-4o-mini synthesises retrieved chunks into a market context summary

---

### 4. `predictor.py` — GPT-4o Prediction Engine

```python
from src.predictor import MarketPredictor

predictor = MarketPredictor(openai_api_key="sk-...")
result = predictor.predict(
    index_name="Nifty50",
    indicator_summary=summary,
    news_context=context,
    current_price=24823.0,
)

print(result.direction)    # "BULLISH"
print(result.confidence)   # 0.73
print(result.price_5d)     # 25200.0
print(result.trade_setup)  # {'entry': 24823, 'target': 25200, 'stop_loss': 24500}
print(result.key_reasons)  # ['RSI at 58 showing momentum', ...]
```

**Prompt Engineering:**
- System prompt: Expert quantitative analyst persona
- Technical block: All 20+ indicator values structured clearly
- News block: RAG-retrieved and synthesised context
- JSON response format enforced via `response_format={"type": "json_object"}`
- Temperature: 0.15 (deterministic financial reasoning)

---

## 📊 Streamlit Dashboard

The dashboard provides 4 tabs:

| Tab | Contents |
|-----|----------|
| **Chart & Indicators** | Interactive candlestick with BB, EMA, Supertrend overlays; RSI/MACD subplot; Volume; Key levels table |
| **AI Prediction** | Direction badge, confidence, 1/3/5-day price targets, trade setup, key reasons, risk factors, full analysis |
| **News & Context** | RAG-synthesised summary + live headlines from 5 financial sources |
| **Technical Details** | Full indicator readings grouped by category; raw OHLCV table |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for GPT-4o predictions |
| `OPENAI_MODEL` | `gpt-4o` | LLM model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `PREDICTION_HORIZON` | `5` | Days ahead |
| `LOOKBACK_PERIOD` | `60` | Historical days for context |
| `CONFIDENCE_THRESHOLD` | `0.65` | Min confidence for signals |

---

## 🔬 Technical Implementation Details

### Data Pipeline (Pandas/NumPy)
```
yfinance.Ticker.history()
    → raw OHLCV DataFrame
    → standardise columns
    → remove zeros & outliers (IQR × 3)
    → ffill/bfill NaN
    → compute log_return, daily_range, volume_ratio
    → cache to Parquet
```

### Composite Signal Score Formula
```python
score = (
    RSI_signal   × 0.25 +   # overbought/sold/scaled
    MACD_signal  × 0.30 +   # histogram direction
    SMA_signal   × 0.20 +   # price vs 50/200 SMA
    Stoch_signal × 0.15 +   # overbought/sold
    BB_signal    × 0.10      # band penetration
) × (0.5 + 0.5 × ADX/50)   # trend-strength amplifier
```

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**. It does **not** constitute financial advice. Stock market predictions are inherently uncertain. Always do your own research and consult a registered financial advisor before making investment decisions. Past performance is not indicative of future results.

---

## 📄 License

MIT License — see `LICENSE` for details.
