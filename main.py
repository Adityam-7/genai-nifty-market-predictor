"""
main.py  —  CLI entry point for GenAI Nifty Predictor
======================================================
Usage:
    python main.py --index Nifty50 --period 6mo --api-key sk-...
    python main.py --index BankNifty --demo
    python main.py --all
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.data_collector import MarketDataCollector, TICKERS
from src.technical_indicators import TechnicalIndicators
from src.rag_pipeline import FinancialNewsRAG
from src.predictor import MarketPredictor


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/predictor.log", mode="a"),
        ],
    )


def run_prediction(index_name: str, ticker: str, period: str, api_key: str | None):
    """Run the full prediction pipeline for a single index."""
    print(f"\n{'='*60}")
    print(f"  GenAI Market Predictor — {index_name}")
    print(f"{'='*60}")

    # 1. Collect Data
    print("\n[1/4] 📡 Fetching market data from Yahoo Finance…")
    collector = MarketDataCollector()
    df = collector.fetch(ticker, period=period, interval="1d")

    if df.empty:
        print(f"  ✗ No data available for {ticker}")
        return

    quote = collector.get_real_time_quote(ticker)
    current_price = quote.get("last_price") or float(df["close"].iloc[-1])
    print(f"  ✓ {len(df)} daily candles loaded")
    print(f"  ✓ Current Price: ₹{current_price:,.2f}")
    if quote.get("pct_change"):
        arrow = "▲" if quote["pct_change"] >= 0 else "▼"
        print(f"  ✓ Change: {arrow} {abs(quote['pct_change']):.2f}%")

    # 2. Technical Indicators
    print("\n[2/4] ⚙️  Computing technical indicators…")
    ti = TechnicalIndicators()
    ind_df = ti.compute_all(df)
    summary = ti.get_signal_summary(ind_df)
    print(f"  ✓ RSI: {summary.get('rsi', 'N/A'):.1f} — {summary.get('rsi_signal','')}")
    print(f"  ✓ MACD Hist: {summary.get('macd_hist', 'N/A'):.3f}")
    print(f"  ✓ ADX: {summary.get('adx', 'N/A'):.1f}")
    print(f"  ✓ Signal Score: {summary.get('composite_score','N/A')} → {summary.get('signal','')}")

    # 3. RAG — News Context
    print("\n[3/4] 📰 Retrieving market news via RAG pipeline…")
    rag = FinancialNewsRAG(openai_api_key=api_key)
    news_context = rag.get_market_context(index_name)
    headlines = rag.get_headlines(5)
    print(f"  ✓ Retrieved {len(headlines)} news articles")
    for h in headlines[:3]:
        print(f"    • [{h['source']}] {h['title'][:70]}…")

    # 4. GenAI Prediction
    print(f"\n[4/4] 🤖 Generating GenAI prediction {'(GPT-4o)' if api_key else '(Demo Mode)'}…")
    predictor = MarketPredictor(openai_api_key=api_key)
    result = predictor.predict(
        index_name=index_name,
        indicator_summary=summary,
        news_context=news_context,
        current_price=current_price,
    )

    # ── Display Result ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PREDICTION RESULT — {index_name}")
    print(f"{'─'*60}")
    direction_icon = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(
        result.direction, "❓"
    )
    print(f"\n  {direction_icon} Direction  : {result.direction}")
    print(f"  🎯 Confidence : {result.confidence*100:.1f}%")
    print(f"  💹 Model      : {result.model_used}")
    print(f"\n  Price Targets:")
    if result.price_1d: print(f"    1-Day : ₹{result.price_1d:,.0f}")
    if result.price_3d: print(f"    3-Day : ₹{result.price_3d:,.0f}")
    if result.price_5d: print(f"    5-Day : ₹{result.price_5d:,.0f}")

    if result.trade_setup and result.trade_setup.get("entry"):
        ts = result.trade_setup
        print(f"\n  Trade Setup:")
        print(f"    Entry     : ₹{ts.get('entry','N/A')}")
        print(f"    Target    : ₹{ts.get('target','N/A')}")
        print(f"    Stop Loss : ₹{ts.get('stop_loss','N/A')}")
        rr = ts.get('risk_reward')
        print(f"    R:R Ratio : 1:{rr}" if rr else "")

    if result.key_reasons:
        print(f"\n  Key Reasons:")
        for i, r in enumerate(result.key_reasons[:5], 1):
            print(f"    {i}. {r}")

    if result.risk_factors:
        print(f"\n  Risk Factors:")
        for r in result.risk_factors[:3]:
            print(f"    ⚠️  {r}")

    if result.raw_analysis:
        print(f"\n  Analysis:")
        print(f"  {result.raw_analysis[:500]}…" if len(result.raw_analysis) > 500 else f"  {result.raw_analysis}")

    print(f"\n  Generated at: {result.generated_at}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GenAI-Driven Nifty50 & Bank Nifty Market Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index Nifty50 --period 6mo
  python main.py --index BankNifty --api-key sk-...
  python main.py --all --period 3mo
  python main.py --ui         # Launch Streamlit dashboard
        """,
    )
    parser.add_argument("--index",   choices=["Nifty50", "BankNifty"], default="Nifty50")
    parser.add_argument("--period",  default="6mo",
                        choices=["1mo", "3mo", "6mo", "1y", "2y"])
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--all",     action="store_true",
                        help="Run prediction for both Nifty50 and Bank Nifty")
    parser.add_argument("--demo",    action="store_true",
                        help="Run in demo mode (no API key)")
    parser.add_argument("--ui",      action="store_true",
                        help="Launch the Streamlit web dashboard")
    parser.add_argument("--log",     default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Ensure logs dir exists
    Path("./logs").mkdir(exist_ok=True)
    setup_logging(args.log)

    # API Key resolution
    api_key = None
    if not args.demo:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OpenAI API key found. Running in demo mode.")
            print("   Set OPENAI_API_KEY or use --api-key or --demo flag.\n")

    # Launch Streamlit UI
    if args.ui:
        import subprocess
        print("🚀 Launching Streamlit dashboard…")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return

    # CLI Predictions
    ticker_map = {"Nifty50": "^NSEI", "BankNifty": "^NSEBANK"}

    if args.all:
        indices = [("Nifty50", "^NSEI"), ("BankNifty", "^NSEBANK")]
    else:
        indices = [(args.index, ticker_map[args.index])]

    for index_name, ticker in indices:
        run_prediction(index_name, ticker, args.period, api_key)

    print(f"\n{'='*60}")
    print("  ✅ Prediction pipeline complete.")
    print("  💡 Run with --ui to open the interactive dashboard.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
