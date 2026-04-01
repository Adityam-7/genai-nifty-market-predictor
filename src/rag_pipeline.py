"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline for financial news
and market context retrieval.

Architecture:
  1. News ingestion  — RSS feeds (MoneyControl, ET Markets, NSE)
  2. Chunking        — RecursiveCharacterTextSplitter
  3. Embedding       — OpenAI text-embedding-3-small
  4. Vector store    — FAISS in-memory (Chroma for persistence)
  5. Retrieval       — MMR (maximal marginal relevance) search
  6. Synthesis       — LangChain LLM chain → context string

The retrieved context is injected into the GPT-4o prediction prompt.
"""

import os
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional
import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  News Sources (RSS Feeds — no API key required)
# ─────────────────────────────────────────────────────────────────────────────
FINANCIAL_RSS_FEEDS = {
    "ET Markets":         "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "MoneyControl":       "https://www.moneycontrol.com/rss/latestnews.xml",
    "LiveMint Markets":   "https://www.livemint.com/rss/markets",
    "Business Standard":  "https://www.business-standard.com/rss/markets-106.rss",
    "CNBC TV18":          "https://www.cnbctv18.com/commonfeeds/v1/ind/rss/market.xml",
}

QUERY_TOPICS = [
    "Nifty 50 stock market India",
    "Bank Nifty index prediction",
    "RBI monetary policy",
    "FII DII trading activity India",
    "India inflation GDP growth",
    "US Fed interest rate impact India",
    "NSE BSE market sentiment",
    "crude oil rupee dollar impact",
]


# ─────────────────────────────────────────────────────────────────────────────
class FinancialNewsRAG:
    """
    RAG pipeline that retrieves and synthesises financial news
    relevant to Nifty50 and Bank Nifty market context.
    """

    def __init__(self, openai_api_key: Optional[str] = None, max_articles: int = 30):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.max_articles = max_articles
        self._article_cache: list[dict] = []
        self._cache_time: Optional[datetime] = None
        self._vector_store = None
        self._embeddings = None
        self._retriever = None

        self._init_langchain()

    # ── Initialise LangChain Components ───────────────────────────────────────
    def _init_langchain(self) -> None:
        """Lazy-import and initialise LangChain components."""
        try:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            from langchain_community.vectorstores import FAISS
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate

            self._OpenAIEmbeddings = OpenAIEmbeddings
            self._ChatOpenAI       = ChatOpenAI
            self._FAISS            = FAISS
            self._TextSplitter     = RecursiveCharacterTextSplitter
            self._Document         = Document
            self._RetrievalQA      = RetrievalQA
            self._PromptTemplate   = PromptTemplate
            self._lc_ready = True
            logger.info("LangChain components loaded successfully.")
        except ImportError as e:
            logger.warning(f"LangChain not available: {e}. Using fallback mode.")
            self._lc_ready = False

    # ── Public: Retrieve Context ──────────────────────────────────────────────
    def get_market_context(
        self,
        index_name: str = "Nifty50",
        refresh: bool = False,
    ) -> str:
        """
        Main entry point. Returns a synthesised market context string
        suitable for injection into the LLM prediction prompt.

        Args:
            index_name: "Nifty50" | "BankNifty"
            refresh:    Force re-fetch even if cache is warm

        Returns:
            Multi-paragraph string summarising recent financial news context.
        """
        articles = self._get_articles(refresh=refresh)

        if not articles:
            return self._fallback_context(index_name)

        if self._lc_ready and self.api_key:
            try:
                return self._rag_retrieve(articles, index_name)
            except Exception as e:
                logger.warning(f"RAG pipeline error: {e}. Falling back to keyword filter.")

        return self._keyword_filter_context(articles, index_name)

    # ── Article Ingestion ─────────────────────────────────────────────────────
    def _get_articles(self, refresh: bool = False) -> list[dict]:
        """Fetch articles from RSS feeds with 30-minute cache."""
        now = datetime.now()
        if (
            not refresh
            and self._cache_time
            and (now - self._cache_time) < timedelta(minutes=30)
            and self._article_cache
        ):
            logger.info(f"[News cache HIT] {len(self._article_cache)} articles")
            return self._article_cache

        articles = []
        for source, url in FINANCIAL_RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]:
                    text = self._extract_text(entry)
                    if len(text) > 80:
                        articles.append({
                            "source":    source,
                            "title":     entry.get("title", ""),
                            "content":   text,
                            "url":       entry.get("link", ""),
                            "published": entry.get("published", str(now.date())),
                        })
            except Exception as e:
                logger.warning(f"Feed error ({source}): {e}")

        # Limit to most recent
        articles = articles[: self.max_articles]
        logger.info(f"[News] Fetched {len(articles)} articles from {len(FINANCIAL_RSS_FEEDS)} feeds")

        self._article_cache = articles
        self._cache_time = now
        return articles

    @staticmethod
    def _extract_text(entry) -> str:
        """Extract clean text from an RSS feed entry."""
        raw = entry.get("summary", "") or entry.get("description", "")
        if "<" in raw:
            soup = BeautifulSoup(raw, "html.parser")
            raw = soup.get_text(separator=" ")
        title = entry.get("title", "")
        return f"{title}. {raw}".strip()

    # ── RAG Retrieval via LangChain + FAISS ───────────────────────────────────
    def _rag_retrieve(self, articles: list[dict], index_name: str) -> str:
        """Build FAISS vector store and retrieve top-k relevant chunks."""
        splitter = self._TextSplitter(chunk_size=600, chunk_overlap=80)
        docs = []
        for art in articles:
            chunks = splitter.split_text(art["content"])
            for chunk in chunks:
                docs.append(
                    self._Document(
                        page_content=chunk,
                        metadata={
                            "source":    art["source"],
                            "title":     art["title"],
                            "url":       art["url"],
                            "published": art["published"],
                        },
                    )
                )

        if not docs:
            return self._fallback_context(index_name)

        embeddings = self._OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model="text-embedding-3-small",
        )

        vector_store = self._FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
        )

        # Build relevant query
        query = (
            f"{index_name} Indian stock market forecast, Nifty Bank Nifty trend, "
            f"RBI policy, FII activity, rupee, crude oil, global cues"
        )
        relevant_docs = retriever.get_relevant_documents(query)

        # Synthesise into a context paragraph via LLM
        context_text = "\n\n---\n\n".join(
            f"[{d.metadata['source']}] {d.page_content}" for d in relevant_docs
        )

        llm = self._ChatOpenAI(
            openai_api_key=self.api_key,
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=800,
        )

        synthesis_prompt = f"""
You are a senior financial analyst specialising in Indian equity markets.
Based on the following recent news articles, synthesise a concise 3-paragraph
market context summary focused on factors that could influence {index_name}
in the near term.

News Articles:
{context_text}

Provide:
1. Key macro/global factors
2. Domestic triggers (RBI, FII/DII flows, sector trends)
3. Sentiment and risk factors

Be factual, concise, and data-focused.
"""
        from langchain.schema import HumanMessage
        response = llm([HumanMessage(content=synthesis_prompt)])
        return response.content

    # ── Fallback: Keyword Filter ──────────────────────────────────────────────
    def _keyword_filter_context(self, articles: list[dict], index_name: str) -> str:
        """
        When LangChain is unavailable or API key is missing,
        filter articles by relevance keywords and concatenate.
        """
        keywords = {
            "Nifty50":   ["nifty", "nse", "sensex", "stock", "market", "equity",
                          "fii", "dii", "rbi", "inflation", "rate", "rupee"],
            "BankNifty": ["bank nifty", "banking", "nifty bank", "rbi", "rate",
                          "nbfc", "credit", "npa", "hdfc", "sbi", "icici", "axis"],
        }
        relevant_kw = keywords.get(index_name, keywords["Nifty50"])

        relevant = []
        for art in articles:
            text_lower = (art["title"] + " " + art["content"]).lower()
            score = sum(1 for kw in relevant_kw if kw in text_lower)
            if score > 0:
                relevant.append((score, art))

        relevant.sort(key=lambda x: -x[0])
        top = [a for _, a in relevant[:5]]

        if not top:
            return self._fallback_context(index_name)

        lines = [f"[{a['source']}] {a['title']}: {a['content'][:300]}..." for a in top]
        return (
            f"Recent Market News Context for {index_name}:\n\n"
            + "\n\n".join(lines)
        )

    @staticmethod
    def _fallback_context(index_name: str) -> str:
        return (
            f"No live news data available. {index_name} analysis based on "
            f"technical indicators and historical price patterns only. "
            f"Consider monitoring RBI policy decisions, FII/DII flows, "
            f"global cues (US Fed, crude oil, USD/INR) for directional bias."
        )

    # ── News Headlines (for UI display) ───────────────────────────────────────
    def get_headlines(self, n: int = 10) -> list[dict]:
        """Return latest n headlines from cached articles."""
        articles = self._get_articles()
        return [
            {
                "source":    a["source"],
                "title":     a["title"],
                "url":       a["url"],
                "published": a["published"],
            }
            for a in articles[:n]
        ]
