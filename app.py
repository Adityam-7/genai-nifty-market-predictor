"""
app.py  —  GenAI-Driven Nifty50 & Bank Nifty Market Predictor
============================================================
Streamlit dashboard bringing together:
  • Live market data (Yahoo Finance)
  • 20+ technical indicators (RSI, MACD, BB, ADX, Supertrend …)
  • RAG-enhanced news context (LangChain + FAISS)
  • GPT-4o market prediction with price targets & trade setup
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path Setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.data_collector import MarketDataCollector, TICKERS
from src.technical_indicators import TechnicalIndicators
from src.rag_pipeline import FinancialNewsRAG
from src.predictor import MarketPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GenAI Nifty Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .stMetric { background: #1c2333; border-radius: 10px; padding: 12px; }
  .stMetric label { color: #8b9cbf !important; font-size: 0.75rem; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
  .signal-pill {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; margin: 4px;
  }
  .bullish  { background: #0d3320; color: #00e676; border: 1px solid #00e676; }
  .bearish  { background: #3d0d0d; color: #ff5252; border: 1px solid #ff5252; }
  .neutral  { background: #1a2233; color: #90caf9; border: 1px solid #90caf9; }
  .card {
    background: #1c2333; border-radius: 12px; padding: 20px;
    margin: 8px 0; border: 1px solid #2d3748;
  }
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #90caf9;
    border-bottom: 2px solid #2d3748; padding-bottom: 8px; margin-bottom: 16px;
  }
  .news-item { border-left: 3px solid #3b82f6; padding: 8px 12px; margin: 6px 0;
               background: #141922; border-radius: 0 8px 8px 0; }
  .news-source { font-size: 0.7rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("⚙️ Configuration")
    st.divider()

    # API Key
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enables GPT-4o predictions and news synthesis (RAG). Leave blank for demo mode.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✓ API Key set")

    st.divider()

    # Index Selection
    index_choice = st.selectbox(
        "📈 Select Index",
        ["Nifty50", "BankNifty"],
        index=0,
    )
    ticker_map = {"Nifty50": "^NSEI", "BankNifty": "^NSEBANK"}
    ticker = ticker_map[index_choice]

    # Time Period
    period = st.select_slider(
        "📅 Historical Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )

    # Prediction Horizon
    horizon = st.select_slider(
        "🔮 Prediction Horizon",
        options=["1 Day", "3 Days", "5 Days"],
        value="5 Days",
    )

    st.divider()

    # Indicator Settings
    st.subheader("📊 Chart Indicators")
    show_bb   = st.checkbox("Bollinger Bands",   True)
    show_ema  = st.checkbox("EMA (9, 21, 50)",   True)
    show_vol  = st.checkbox("Volume",             True)
    show_sptr = st.checkbox("Supertrend",         False)

    st.divider()
    refresh_btn = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("⚡ Auto-refresh (60s)", value=False)
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    # ── Live Price Widget ───────────────────────────────────────────
    st.divider()
    st.markdown("### 📡 Live Price")
    live_placeholder = st.empty()

    st.divider()
    st.markdown("**About**")
    st.caption(
        "GenAI Market Predictor uses Yahoo Finance data, "
        "20+ technical indicators, LangChain RAG (news retrieval), "
        "and GPT-4o for contextual market forecasts."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(ticker: str, period: str):
    collector = MarketDataCollector(cache_ttl_minutes=15)
    df = collector.fetch(ticker, period=period, interval="1d")
    return df

@st.cache_data(ttl=900, show_spinner=False)
def compute_indicators(df_json: str):
    df = pd.read_json(df_json, orient="split")
    ti = TechnicalIndicators()
    enriched = ti.compute_all(df)
    summary  = ti.get_signal_summary(enriched)
    return enriched.to_json(orient="split"), summary

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_context(index_name: str, api_key_hash: str):
    rag = FinancialNewsRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))
    context   = rag.get_market_context(index_name)
    headlines = rag.get_headlines(12)
    return context, headlines

@st.cache_data(ttl=900, show_spinner=False)
def run_prediction(
    index_name: str,
    ind_summary_json: str,
    news_context: str,
    current_price: float,
    api_key_hash: str,
):
    import json
    ind_summary = json.loads(ind_summary_json)
    predictor = MarketPredictor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )
    result = predictor.predict(
        index_name=index_name,
        indicator_summary=ind_summary,
        news_context=news_context,
        current_price=current_price,
    )
    return result

def get_real_time_quote(ticker: str):
    collector = MarketDataCollector()
    return collector.get_real_time_quote(ticker)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_chart(df: pd.DataFrame, ind_df: pd.DataFrame, options: dict) -> go.Figure:
    rows = 3 if options["show_vol"] else 2
    row_heights = [0.55, 0.25, 0.20] if options["show_vol"] else [0.65, 0.35]
    specs = [[{"secondary_y": False}]] * rows
    subplot_titles = [f"Price", "RSI  |  MACD", "Volume"] if options["show_vol"] else ["Price", "RSI  |  MACD"]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=ind_df.index, open=ind_df["open"], high=ind_df["high"],
        low=ind_df["low"],  close=ind_df["close"],
        name="Price",
        increasing_line_color="#00e676",
        decreasing_line_color="#ff5252",
        increasing_fillcolor="rgba(0,230,118,0.5)",
        decreasing_fillcolor="rgba(255,82,82,0.5)",
    ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────────────────
    if options["show_ema"]:
        for n, color in [(9, "#ffd54f"), (21, "#81d4fa"), (50, "#ce93d8")]:
            col_name = f"ema_{n}"
            if col_name in ind_df.columns:
                fig.add_trace(go.Scatter(
                    x=ind_df.index, y=ind_df[col_name], name=f"EMA {n}",
                    line=dict(color=color, width=1.2), opacity=0.85,
                ), row=1, col=1)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    if options["show_bb"] and "bb_upper" in ind_df.columns:
        fig.add_trace(go.Scatter(
            x=ind_df.index, y=ind_df["bb_upper"], name="BB Upper",
            line=dict(color="#546e7a", width=1, dash="dash"), opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ind_df.index, y=ind_df["bb_lower"], name="BB Lower",
            line=dict(color="#546e7a", width=1, dash="dash"), opacity=0.6,
            fill="tonexty", fillcolor="rgba(84,110,122,0.08)",
        ), row=1, col=1)

    # ── Supertrend ─────────────────────────────────────────────────────────
    if options["show_sptr"] and "supertrend" in ind_df.columns:
        bull = ind_df["supertrend"].where(ind_df["st_direction"] == 1)
        bear = ind_df["supertrend"].where(ind_df["st_direction"] == -1)
        fig.add_trace(go.Scatter(
            x=ind_df.index, y=bull, name="Supertrend ↑",
            line=dict(color="#00e676", width=2), mode="lines",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ind_df.index, y=bear, name="Supertrend ↓",
            line=dict(color="#ff5252", width=2), mode="lines",
        ), row=1, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    if "rsi" in ind_df.columns:
        fig.add_trace(go.Scatter(
            x=ind_df.index, y=ind_df["rsi"], name="RSI",
            line=dict(color="#ba68c8", width=1.5),
        ), row=2, col=1)
        for level, color in [(70, "#ff5252"), (30, "#00e676")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          opacity=0.5, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(100,100,200,0.05)",
                      line_width=0, row=2, col=1)

    # ── MACD Histogram ─────────────────────────────────────────────────────
    if "macd_hist" in ind_df.columns:
        colors = ind_df["macd_hist"].apply(
            lambda x: "rgba(0,230,118,0.5)" if x >= 0 else "rgba(255,82,82,0.5)"
        )
        fig.add_trace(go.Bar(
            x=ind_df.index, y=ind_df["macd_hist"], name="MACD Hist",
            marker_color=colors, showlegend=False,
        ), row=2, col=1)

    # ── Volume ─────────────────────────────────────────────────────────────
    if options["show_vol"] and "volume" in ind_df.columns:
        vol_colors = [
            "rgba(0,230,118,0.4)" if c >= o else "rgba(255,82,82,0.4)"
            for c, o in zip(ind_df["close"], ind_df["open"])
        ]
        fig.add_trace(go.Bar(
            x=ind_df.index, y=ind_df["volume"], name="Volume",
            marker_color=vol_colors, showlegend=False,
        ), row=3, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=680,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        xaxis3_title="Date" if options["show_vol"] else "",
    )
    fig.update_yaxes(gridcolor="#1e2d40", zerolinecolor="#1e2d40")
    fig.update_xaxes(gridcolor="#1e2d40", showgrid=False)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 4px 0;'>
      <h1 style='font-size:2.2rem; margin:0;'>
        📊 GenAI Market Predictor
      </h1>
      <p style='color:#8b9cbf; margin-top:4px; font-size:0.95rem;'>
        Nifty50 &amp; Bank Nifty — Powered by Yahoo Finance · LangChain RAG · GPT-4o
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {index_choice} data…"):
        df = load_market_data(ticker, period)

    if df.empty:
        st.error("⚠️ Failed to load market data. Please check your connection and try again.")
        st.stop()

    # ── Indicators ────────────────────────────────────────────────────────
    with st.spinner("⚙️ Computing technical indicators…"):
        ind_df_json, ind_summary = compute_indicators(df.to_json(orient="split"))
        ind_df = pd.read_json(ind_df_json, orient="split")

    # ── Real-time Quote ───────────────────────────────────────────────────
    quote = get_real_time_quote(ticker)
    current_price = quote.get("last_price") or float(df["close"].iloc[-1])
    chg     = quote.get("change", 0) or 0
    pct_chg = quote.get("pct_change", 0) or 0

    # ── Render Live Price in Sidebar ──────────────────────────────────────
    color   = "#00e676" if chg >= 0 else "#ff5252"
    arrow   = "▲" if chg >= 0 else "▼"
    live_placeholder.markdown(f"""
    <div style="background:#1c2333; border-radius:12px; padding:14px; border:1px solid {color}44; text-align:center;">
      <div style="font-size:0.75rem; color:#8b9cbf; margin-bottom:4px;">{index_choice}</div>
      <div style="font-size:1.8rem; font-weight:800; color:{color};">₹{current_price:,.2f}</div>
      <div style="font-size:0.85rem; color:{color}; margin-top:4px;">
        {arrow} {abs(chg):,.2f} &nbsp;|&nbsp; {arrow} {abs(pct_chg):.2f}%
      </div>
      <div style="font-size:0.7rem; color:#4a5568; margin-top:6px;">
        52W H: ₹{quote.get('52w_high') or 0:,.0f} &nbsp;·&nbsp; 52W L: ₹{quote.get('52w_low') or 0:,.0f}
      </div>
      <div style="font-size:0.65rem; color:#4a5568; margin-top:4px;">
        Updated {datetime.now().strftime('%H:%M:%S')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh every 60s
    if auto_refresh:
        import time
        st.markdown(
            f'<meta http-equiv="refresh" content="60">',
            unsafe_allow_html=True,
        )

    # ══ METRICS ROW ═══════════════════════════════════════════════════════
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    delta_arrow = "▲" if chg >= 0 else "▼"

    with m1:
        st.metric("Last Price", f"₹{current_price:,.2f}",
                  f"{delta_arrow} {abs(chg):.2f} ({abs(pct_chg):.2f}%)")
    with m2:
        rsi = ind_summary.get("rsi")
        rsi_str = f"{rsi:.1f}" if rsi else "N/A"
        st.metric("RSI (14)", rsi_str, ind_summary.get("rsi_signal",""))
    with m3:
        adx = ind_summary.get("adx")
        st.metric("ADX (14)", f"{adx:.1f}" if adx else "N/A",
                  "Strong Trend" if adx and adx > 25 else "Weak Trend")
    with m4:
        st.metric("MACD Hist", f"{ind_summary.get('macd_hist',0):.2f}",
                  "Positive ↑" if (ind_summary.get("macd_hist") or 0) > 0 else "Negative ↓")
    with m5:
        score = ind_summary.get("composite_score", 0) or 0
        st.metric("Signal Score", f"{score:.3f}",
                  ind_summary.get("signal", "NEUTRAL"))
    with m6:
        hi52 = quote.get("52w_high")
        lo52 = quote.get("52w_low")
        if hi52 and lo52:
            pct_from_hi = (current_price - hi52) / hi52 * 100
            st.metric("52W High", f"₹{hi52:,.0f}", f"{pct_from_hi:.1f}% from high")
        else:
            st.metric("52W High", "N/A")

    st.divider()

    # ══ TABS ══════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Chart & Indicators",
        "🤖 AI Prediction",
        "📰 News & Context",
        "🔬 Technical Details",
    ])

    # ── TAB 1: CHART ──────────────────────────────────────────────────────
    with tab1:
        chart_opts = {
            "show_bb":   show_bb,
            "show_ema":  show_ema,
            "show_vol":  show_vol,
            "show_sptr": show_sptr,
        }
        fig = build_chart(df, ind_df, chart_opts)
        st.plotly_chart(fig, use_container_width=True)

        # Key Levels Table
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">🎯 Key Support & Resistance</div>', unsafe_allow_html=True)
            levels_data = {
                "Level": ["R2", "R1", "Pivot", "S1", "S2"],
                "Price":  [
                    ind_summary.get("r2"), ind_summary.get("r1"),
                    ind_summary.get("pivot"),
                    ind_summary.get("s1"),  ind_summary.get("s2"),
                ],
                "Type":   ["Resistance", "Resistance", "Pivot", "Support", "Support"],
            }
            lvl_df = pd.DataFrame(levels_data)
            lvl_df["Price"] = lvl_df["Price"].apply(
                lambda x: f"₹{x:,.2f}" if x else "N/A"
            )
            st.dataframe(lvl_df, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">📊 Volatility</div>', unsafe_allow_html=True)
            vd = {
                "Indicator": ["ATR (14)", "BB Width", "BB Upper", "BB Lower", "Hist. Vol (20d)"],
                "Value": [
                    f"₹{ind_summary.get('atr',0):,.2f}" if ind_summary.get("atr") else "N/A",
                    f"{ind_df['bb_width'].iloc[-1]:.2f}%" if "bb_width" in ind_df else "N/A",
                    f"₹{ind_summary.get('bb_upper',0):,.2f}" if ind_summary.get("bb_upper") else "N/A",
                    f"₹{ind_summary.get('bb_lower',0):,.2f}" if ind_summary.get("bb_lower") else "N/A",
                    f"{ind_df['hist_vol'].iloc[-1]:.2f}%" if "hist_vol" in ind_df else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(vd), hide_index=True, use_container_width=True)

    # ── TAB 2: AI PREDICTION ──────────────────────────────────────────────
    with tab2:
        st.markdown("### 🤖 GenAI Market Prediction")

        if not api_key:
            st.info(
                "💡 **Demo Mode Active** — Enter your OpenAI API key in the sidebar "
                "to unlock GPT-4o predictions with RAG-enhanced news context. "
                "Demo predictions use rule-based technical analysis."
            )

        predict_btn = st.button(
            "⚡ Generate Prediction",
            type="primary",
            use_container_width=True,
        )

        if predict_btn or "prediction" not in st.session_state:
            with st.spinner("🧠 Running GenAI prediction pipeline…"):
                # Get news context
                api_key_hash = str(hash(api_key or "demo"))
                news_ctx, headlines = get_news_context(index_choice, api_key_hash)

                # Run prediction
                import json
                result = run_prediction(
                    index_name=index_choice,
                    ind_summary_json=json.dumps(ind_summary),
                    news_context=news_ctx,
                    current_price=current_price,
                    api_key_hash=api_key_hash,
                )
                st.session_state["prediction"] = result
                st.session_state["news_ctx"]   = news_ctx
                st.session_state["headlines"]  = headlines

        result = st.session_state.get("prediction")
        if result:
            # Direction Badge
            dir_class = {
                "BULLISH": "bullish", "BEARISH": "bearish"
            }.get(result.direction, "neutral")
            dir_emoji = {"BULLISH": "📈 BULLISH", "BEARISH": "📉 BEARISH"}.get(
                result.direction, "➡️ NEUTRAL"
            )
            conf_pct = round(result.confidence * 100, 1)

            st.markdown(f"""
            <div style="text-align:center; margin: 16px 0;">
              <span class="signal-pill {dir_class}" style="font-size:1.4rem; padding: 10px 32px;">
                {dir_emoji}
              </span>
              <div style="margin-top:12px; font-size:1.1rem; color:#8b9cbf;">
                Confidence: <strong style="color:#90caf9">{conf_pct}%</strong>
                &nbsp;|&nbsp; Model: <strong style="color:#90caf9">{result.model_used}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Price Targets
            st.markdown('<div class="section-title">🎯 Price Targets (Closing)</div>', unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                delta_1d = f"{'▲' if result.price_1d and result.price_1d > current_price else '▼'} {abs(result.price_1d - current_price):,.0f}" if result.price_1d else ""
                st.metric("1-Day Close", f"₹{result.price_1d:,.0f}" if result.price_1d else "N/A", delta_1d)
            with tc2:
                delta_3d = f"{'▲' if result.price_3d and result.price_3d > current_price else '▼'} {abs(result.price_3d - current_price):,.0f}" if result.price_3d else ""
                st.metric("3-Day Close", f"₹{result.price_3d:,.0f}" if result.price_3d else "N/A", delta_3d)
            with tc3:
                delta_5d = f"{'▲' if result.price_5d and result.price_5d > current_price else '▼'} {abs(result.price_5d - current_price):,.0f}" if result.price_5d else ""
                st.metric("5-Day Close", f"₹{result.price_5d:,.0f}" if result.price_5d else "N/A", delta_5d)

            # Day High / Low Table
            st.markdown('<div class="section-title" style="margin-top:16px">📊 Predicted Day High / Low</div>', unsafe_allow_html=True)
            hl_data = {
                "Horizon": ["1 Day", "3 Days", "5 Days"],
                "Predicted Close": [
                    f"₹{result.price_1d:,.0f}" if result.price_1d else "N/A",
                    f"₹{result.price_3d:,.0f}" if result.price_3d else "N/A",
                    f"₹{result.price_5d:,.0f}" if result.price_5d else "N/A",
                ],
                "Day High 🔺": [
                    f"₹{result.day_high_1d:,.0f}" if result.day_high_1d else "N/A",
                    f"₹{result.day_high_3d:,.0f}" if result.day_high_3d else "N/A",
                    f"₹{result.day_high_5d:,.0f}" if result.day_high_5d else "N/A",
                ],
                "Day Low 🔻": [
                    f"₹{result.day_low_1d:,.0f}" if result.day_low_1d else "N/A",
                    f"₹{result.day_low_3d:,.0f}" if result.day_low_3d else "N/A",
                    f"₹{result.day_low_5d:,.0f}" if result.day_low_5d else "N/A",
                ],
                "Expected Range": [
                    f"₹{abs((result.day_high_1d or 0) - (result.day_low_1d or 0)):,.0f}" if result.day_high_1d and result.day_low_1d else "N/A",
                    f"₹{abs((result.day_high_3d or 0) - (result.day_low_3d or 0)):,.0f}" if result.day_high_3d and result.day_low_3d else "N/A",
                    f"₹{abs((result.day_high_5d or 0) - (result.day_low_5d or 0)):,.0f}" if result.day_high_5d and result.day_low_5d else "N/A",
                ],
            }
            st.dataframe(
                pd.DataFrame(hl_data),
                hide_index=True,
                use_container_width=True,
            )

            # Trade Setup
            if result.trade_setup and result.trade_setup.get("entry"):
                st.markdown('<div class="section-title" style="margin-top:16px">💼 Trade Setup</div>', unsafe_allow_html=True)
                ts = result.trade_setup
                t1, t2, t3, t4 = st.columns(4)
                with t1: st.metric("Entry",       f"₹{ts.get('entry','N/A'):,.0f}" if isinstance(ts.get('entry'), (int,float)) else "N/A")
                with t2: st.metric("Target",      f"₹{ts.get('target','N/A'):,.0f}" if isinstance(ts.get('target'), (int,float)) else "N/A")
                with t3: st.metric("Stop Loss",   f"₹{ts.get('stop_loss','N/A'):,.0f}" if isinstance(ts.get('stop_loss'), (int,float)) else "N/A")
                with t4: st.metric("Risk:Reward", f"1:{ts.get('risk_reward', 'N/A')}" if ts.get('risk_reward') else "N/A")

            # Key Reasons & Risks
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown('<div class="section-title" style="margin-top:16px">✅ Key Reasons</div>', unsafe_allow_html=True)
                for i, reason in enumerate(result.key_reasons[:6], 1):
                    st.markdown(f"**{i}.** {reason}")
            with col_r2:
                st.markdown('<div class="section-title" style="margin-top:16px">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                for risk in result.risk_factors[:4]:
                    st.markdown(f"⚠️ {risk}")

            # Full Analysis
            if result.raw_analysis:
                with st.expander("📝 Full AI Analysis", expanded=False):
                    st.markdown(result.raw_analysis)

            st.caption(f"Generated at: {result.generated_at}")

    # ── TAB 3: NEWS ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📰 Market News & RAG Context")

        if "news_ctx" in st.session_state:
            with st.expander("🤖 RAG-Synthesised Market Context", expanded=True):
                st.markdown(st.session_state["news_ctx"])

        if "headlines" in st.session_state:
            st.markdown('<div class="section-title" style="margin-top:16px">📋 Latest Headlines</div>', unsafe_allow_html=True)
            for art in st.session_state["headlines"]:
                title = art.get("title", "")
                source = art.get("source", "")
                url  = art.get("url", "#")
                pub  = art.get("published", "")
                st.markdown(f"""
                <div class="news-item">
                  <div class="news-source">{source} · {pub[:16]}</div>
                  <a href="{url}" target="_blank" style="color:#90caf9; text-decoration:none; font-size:0.9rem;">
                    {title}
                  </a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run a prediction first to load news context.")

    # ── TAB 4: TECHNICAL DETAILS ──────────────────────────────────────────
    with tab4:
        st.markdown("### 🔬 Full Technical Indicator Readings")
        if ind_summary:
            # Group indicators
            sections = {
                "📈 Trend": ["sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
                              "macd", "macd_signal", "macd_hist", "adx"],
                "⚡ Momentum": ["rsi", "rsi_signal", "stoch_k", "stoch_d",
                                "williams_r", "cci"],
                "🌊 Volatility": ["bb_upper", "bb_mid", "bb_lower", "bb_position_pct",
                                   "atr", "hist_vol"],
                "📦 Volume": ["vwap", "mfi", "cmf", "obv"],
                "🎯 Key Levels": ["pivot", "r1", "r2", "s1", "s2"],
                "🧮 Composite": ["composite_score", "signal"],
            }

            for section_name, keys in sections.items():
                with st.expander(section_name, expanded=(section_name == "🧮 Composite")):
                    rows = []
                    for k in keys:
                        v = ind_summary.get(k)
                        if v is not None:
                            rows.append({"Indicator": k.upper().replace("_", " "), "Value": str(v)})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Historical data table
        with st.expander("📋 Raw OHLCV Data (Last 30 Days)"):
            display_df = df.tail(30)[["open", "high", "low", "close", "volume"]].copy()
            display_df.columns = ["Open", "High", "Low", "Close", "Volume"]
            display_df = display_df.round(2)
            st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()
