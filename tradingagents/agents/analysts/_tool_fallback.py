"""Fallback tool execution for analysts when LLM tool calling fails.

Some LLM providers (e.g., Google Gemini) may not reliably produce
tool_calls via bind_tools(). When this happens, the analyst node
manually invokes tools and returns the formatted raw data directly
(no extra LLM call) to avoid exceeding API rate limits.

Downstream agents (Bull/Bear Researchers, Trader, etc.) will analyze
the data — the analyst fallback only needs to provide it.
"""

import logging
from datetime import datetime, timedelta

from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_income_statement,
    get_news,
    get_global_news,
)

logger = logging.getLogger(__name__)

_DEFAULT_INDICATORS = [
    "rsi", "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "boll_ub", "boll_lb", "atr",
]


def _lookback_date(current_date: str, days: int = 60) -> str:
    """Calculate a lookback date from current_date string (yyyy-mm-dd)."""
    dt = datetime.strptime(current_date, "%Y-%m-%d")
    return (dt - timedelta(days=days)).strftime("%Y-%m-%d")


def _news_lookback_date(current_date: str, days: int = 7) -> str:
    return _lookback_date(current_date, days)


def fallback_market(llm, ticker: str, current_date: str, system_message: str) -> str:
    """Manually fetch stock data + indicators and return formatted raw data."""
    logger.warning("Market Analyst: tool calling failed, using manual fallback for %s", ticker)

    start_date = _lookback_date(current_date, 60)

    try:
        stock_data = get_stock_data.invoke({
            "symbol": ticker,
            "start_date": start_date,
            "end_date": current_date,
        })
    except Exception as e:
        logger.error("Fallback get_stock_data failed: %s", e)
        stock_data = f"Error fetching stock data: {e}"

    indicator_reports = []
    for ind in _DEFAULT_INDICATORS:
        try:
            report = get_indicators.invoke({
                "symbol": ticker,
                "indicator": ind,
                "curr_date": current_date,
                "look_back_days": 30,
            })
            indicator_reports.append(f"### {ind}\n{report}")
        except Exception as e:
            logger.error("Fallback get_indicators(%s) failed: %s", ind, e)

    indicators_text = "\n\n".join(indicator_reports) if indicator_reports else "No indicator data available."

    return f"""# Market Technical Analysis Data — {ticker} (as of {current_date})

## Stock Price Data (OHLCV)
{stock_data}

## Technical Indicators
{indicators_text}"""


def fallback_fundamentals(llm, ticker: str, current_date: str, system_message: str) -> str:
    """Manually fetch fundamental data and return formatted raw data."""
    logger.warning("Fundamentals Analyst: tool calling failed, using manual fallback for %s", ticker)

    data_parts = []
    for tool_fn, label in [
        (get_fundamentals, "Company Fundamentals"),
        (get_balance_sheet, "Balance Sheet"),
        (get_income_statement, "Income Statement"),
    ]:
        try:
            if tool_fn == get_fundamentals:
                result = tool_fn.invoke({"ticker": ticker, "curr_date": current_date})
            else:
                result = tool_fn.invoke({"ticker": ticker, "freq": "quarterly", "curr_date": current_date})
            data_parts.append(f"### {label}\n{result}")
        except Exception as e:
            logger.error("Fallback %s failed: %s", label, e)

    all_data = "\n\n".join(data_parts) if data_parts else "No fundamental data available."

    return f"""# Fundamentals Analysis Data — {ticker} (as of {current_date})

{all_data}"""


def fallback_news(llm, ticker: str, current_date: str, system_message: str) -> str:
    """Manually fetch news data and return formatted raw data."""
    logger.warning("News Analyst: tool calling failed, using manual fallback for %s", ticker)

    start_date = _news_lookback_date(current_date, 7)
    parts = []

    try:
        company_news = get_news.invoke({
            "ticker": ticker,
            "start_date": start_date,
            "end_date": current_date,
        })
        parts.append(f"### Company News ({ticker})\n{company_news}")
    except Exception as e:
        logger.error("Fallback get_news failed: %s", e)

    try:
        global_news = get_global_news.invoke({
            "curr_date": current_date,
            "look_back_days": 7,
            "limit": 5,
        })
        parts.append(f"### Global/Macro News\n{global_news}")
    except Exception as e:
        logger.error("Fallback get_global_news failed: %s", e)

    all_news = "\n\n".join(parts) if parts else "No news data available."

    return f"""# News Analysis Data — {ticker} (as of {current_date})

{all_news}"""


def fallback_social(llm, ticker: str, current_date: str, system_message: str) -> str:
    """Manually fetch social/sentiment data and return formatted raw data."""
    logger.warning("Social Media Analyst: tool calling failed, using manual fallback for %s", ticker)

    start_date = _news_lookback_date(current_date, 7)

    try:
        news_data = get_news.invoke({
            "ticker": ticker,
            "start_date": start_date,
            "end_date": current_date,
        })
    except Exception as e:
        logger.error("Fallback get_news (social) failed: %s", e)
        news_data = f"Error fetching news: {e}"

    return f"""# Social Media & Sentiment Data — {ticker} (as of {current_date})

## News and Social Media Discussions
{news_data}"""
