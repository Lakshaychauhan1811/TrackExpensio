"""
LangGraph-inspired chat orchestration without LangChain.

Uses Groq function-calling to decide which MCP tools to call, executes them via
the in-process FastMCP tool registry, and returns a natural language reply.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any, Dict, List, Optional

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import dateparser
from dateparser.search import search_dates

from groq import Groq

from main import TOOL_REGISTRY

GROQ_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))


def _now_local() -> datetime:
    """Return current datetime in the app's local timezone."""
    return datetime.now(APP_TZ)


def _today_str() -> str:
    return _now_local().strftime("%Y-%m-%d")


def _build_system_prompt() -> str:
    today = _now_local()
    today_iso = today.strftime("%Y-%m-%d")
    today_display = today.strftime("%d %b %Y")
    return SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=today_iso,
        today_display=today_display,
        year=today.year,
    )


SYSTEM_PROMPT_TEMPLATE = """You are TrackExpensio, a helpful AI finance assistant. Your role is to help users manage their finances through natural conversation.

CURRENT DATE CONTEXT (always use this — never guess):
- Today is {today_display} ({today_iso})
- Current year is {year}

CORE PRINCIPLES:
1. Always use tools to perform actions (add expenses, check budgets, fetch stock data, etc.)
2. When user mentions MULTIPLE tasks in one message, you MUST call ALL relevant tools IN PARALLEL (same tool_call batch)
   - Example: "What's ITC share price and I spent 500 in food this week" → call both yahoo_finance AND add_expense together
   - Example: "Add 2000 for travel and set savings goal of 30000" → call add_expense AND set_savings_goal together
3. When user mentions MULTIPLE expenses in one message, you MUST call add_expense SEPARATELY for EACH expense (but all in parallel)
4. Default currency is INR unless user explicitly mentions another (dollar, euro, $, €, etc.)
5. Extract dates from user's message — DEFAULT to {today_iso} unless user specifically mentions a different date
6. Be conversational and helpful - answer questions naturally when appropriate

EXPENSE ADDITION:
- Extract for EACH expense: amount, category, date, merchant (if mentioned), currency
- Categories: Food, Travel, Bills, Shopping, Entertainment, Utilities, Rent, Health, Education, Other
- If user lists multiple expenses, make separate add_expense calls for each (all in parallel)
- Always use proper date format (YYYY-MM-DD) - parse natural dates like "23 nov" or "november 23" correctly
- If user says "today" or no date mentioned, use {today_iso} exactly
- NEVER invent or guess dates — always use {today_iso} for "today" and relative phrases

EXPENSE QUERIES:
- When user asks to see expenses (e.g., "show my expenses", "tell me this month expenses", "where did I spend"), use list_expenses tool
- Always include date range (start_date and end_date) when calling list_expenses
- For "this month" queries, use current month's first day to today
- For "last week" queries, use 7 days ago to today
- When showing expenses, format them clearly with: date, amount, category, merchant (where spent)
- If user asks "where I spent" or "where did I spend", make sure to show the merchant/location for each expense

GMAIL / EMAIL EXPENSE IMPORT:
- When user asks to fetch, sync, scan, or import from Gmail/email/inbox, you MUST call sync_gmail_bills FIRST
- After sync_gmail_bills succeeds, call list_expenses for the last 6 months to show updated expense track
- If sync_gmail_bills returns an error about Gmail permission, tell user to click "Sync with Google" in the app header and reconnect to grant Gmail access
- Do NOT only call list_expenses when user explicitly asked to fetch Gmail — that shows old data only

EXPENSE DELETION:
- When user says "remove this expense", "delete this expense", "delete the last expense", "remove expense", or similar:
  1. FIRST call list_expenses to get recent expenses (last 30 days or this month)
  2. Find the matching expense based on user's context:
     - If user says "last expense" or "recent expense", use the most recent expense
     - If user mentions specific details (amount, merchant, date, category), find the matching expense
     - If user says "this expense" referring to a previous message, use the most recent expense
  3. THEN call delete_expense with the expense_id from step 2
- Always confirm deletion with a friendly message

INCOME ADDITION:
- Use add_income tool when user mentions income
- Store in Income section automatically

SAVINGS GOALS:
- Use set_savings_goal tool when user wants to save money or set a savings target
- Store in Savings Goals section automatically

INVESTMENTS:
- Use add_investment tool when user mentions investments
- Use yahoo_finance for stock prices/queries (doesn't add to investments, just queries)

DEBTS:
- Use add_debt tool when user mentions debts or loans
- Store in Debt section automatically

STOCK QUERIES (any Yahoo Finance listed symbol worldwide):
- For technical analysis (RSI, MACD, Bollinger, buy/sell/hold signal, chart, projection), use smart_stock_analyze
- For price/returns/IPO history questions, call BOTH:
  1. yahoo_finance — current price, market cap, sector, 52-week range, short-term returns
  2. get_stock_returns with from_listing=true — listing/IPO history and year-by-year growth
- Accept company names OR tickers: "Netflix", "Palantir", "AAPL", "INFY.NS", "Tesla", etc.
- NEVER give buy/sell/hold advice or say "good time to buy" or "you should invest"
- When user asks "should I buy X", respond with factual historical data only, then add:
  "I cannot provide investment advice. Please review this data and consult a licensed financial advisor."
- Present: current price, market cap, sector, listing/IPO date, 1D/1W/1M/1Y/3Y returns, since-listing return, year-by-year breakdown, 52-week range, P/E if available
- Do NOT recommend stocks; only report data retrieved from Yahoo Finance tools
- **CRITICAL: When user specifies a price range (e.g., "stocks between ₹100-500"), check if price falls in range — do not recommend, only report whether it matches**

RESPONSES:
- Be natural and conversational
- Never show JSON, status codes, or backend details
- Summarize results clearly
- If user asks a question, use appropriate tools to answer accurately
- When multiple tools are called, summarize all results together"""


# Groq-compatible tool schemas (must include "type": "function")
CHAT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "quick_add_expense",
            "description": "Quickly add an expense when only amount/description is provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Expense amount"},
                    "description": {"type": "string", "description": "Short note"},
                    "merchant": {"type": "string", "description": "Store or merchant", "default": ""},
                },
                "required": ["amount", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": "Add a new expense transaction with automatic AI category suggestion. Records expense with date, amount, category, merchant, and optional note. DEFAULT CURRENCY IS INR. Only include metadata.currency if the user explicitly mentions a different currency (e.g., 'dollar', 'euro', '$', '€'). If user doesn't mention currency, use INR (don't include metadata).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD. Use the CURRENT DATE from system context when user says 'today' or gives no date."},
                    "amount": {"type": "number"},
                    "category": {"type": "string"},
                    "merchant": {"type": "string", "default": ""},
                    "note": {"type": "string", "default": ""},
                    "metadata": {
                        "type": "object",
                        "description": "Only include if currency is NOT INR. Contains currency field for non-INR expenses.",
                        "properties": {
                            "currency": {"type": "string", "description": "Currency code (USD, EUR, GBP, JPY). Only use if user explicitly mentions it. Default is INR."}
                        }
                    }
                },
                "required": ["date", "amount", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_budget",
            "description": "Set or update a monthly budget limit for a specific expense category. Creates a budget that will be tracked monthly with alerts when approaching or exceeding limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["category", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_budget_status",
            "description": "Check current month's budget status for all categories. Returns remaining budget, spending percentage, alerts for overspending, and AI-powered tips for better financial management.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Get a summary of expenses grouped by category for a date range. Provides total spending per category, overall total, and count of transactions. Useful for understanding spending patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ai_insights",
            "description": "Generate AI-powered spending insights for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_financial_report",
            "description": "Produce a comprehensive financial report for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_savings_goal",
            "description": "Create or update a savings goal with target amount and deadline. Helps track progress toward financial goals like vacation, emergency fund, down payment, or any savings target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_name": {"type": "string"},
                    "target_amount": {"type": "number"},
                    "target_date": {"type": "string"},
                    "current_amount": {"type": "number", "default": 0},
                },
                "required": ["goal_name", "target_amount", "target_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_bill_reminder",
            "description": "Add a recurring bill reminder for utilities, subscriptions, rent, or other regular payments. Creates reminders to help you never miss a payment deadline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_name": {"type": "string"},
                    "amount": {"type": "number"},
                    "due_date": {"type": "string"},
                    "frequency": {"type": "string", "default": "monthly"},
                    "category": {"type": "string", "default": "Bills"},
                },
                "required": ["bill_name", "amount", "due_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_bills",
            "description": "List bills due within the next N days.",
            "parameters": {
                "type": "object",
                "properties": {"days_ahead": {"type": "integer", "default": 30}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_income",
            "description": "Record an income transaction such as salary, freelance work, dividends, or other earnings. Tracks all sources of income to help calculate net worth and financial health.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                    "source": {"type": "string"},
                    "category": {"type": "string", "default": "Salary"},
                    "note": {"type": "string", "default": ""},
                },
                "required": ["date", "amount", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sync_gmail_bills",
            "description": "Scan Gmail for payment receipts, order bills, and bank transaction emails and import them as expenses. Use when user asks to fetch/sync/import expenses from Gmail or email. Requires Google login with Gmail permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_messages": {"type": "integer", "default": 40, "description": "Max emails to scan (default 40)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smart_stock_analyze",
            "description": "Technical stock analysis with RSI, MACD, Bollinger Bands, buy/sell/hold signal engine, support/resistance, 30-day price projection, and news sentiment. Use when user asks to analyze a stock, wants technical indicators, buy/sell signals, or chart analysis. Ticker examples: AAPL, TSLA, RELIANCE.NS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, INFY.NS)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yahoo_finance",
            "description": "Get real-time market data for ANY Yahoo Finance symbol. Accepts company names (Netflix, Tesla) or tickers (AAPL, INFY.NS). Returns price, market cap, listing date, 52-week range, P/E, dividend yield, sector, and returns (1D/1W/1M/3M/6M/1Y/3Y). Always pair with get_stock_returns for full history.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_return_one_year",
            "description": "Calculate the 1-year return percentage for a given stock symbol. Fetches historical price data for the past year and calculates the percentage return from start price to current price. Returns symbol, start_price, end_price, and one_year_return_percent.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_returns",
            "description": "Calculate stock returns for flexible time periods with year-by-year breakdown. Works for ANY stock symbol from ANY exchange worldwide (US, India, UK, Japan, Europe, etc.). Can calculate returns for specific years (e.g., last 7 years), from listing date to today, or all available historical data. Returns overall return and detailed year-by-year returns. Use this when user asks for 'last X years return', 'return from when listed', 'return from listing date', or 'every year returns' for any stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol from any exchange (e.g., AAPL for US, ITC.NS for India, VOD.L for UK, 7203.T for Japan, SAP.DE for Germany, etc.). Works for any stock available on Yahoo Finance."},
                    "years": {"type": "integer", "description": "Number of years to look back (e.g., 7 for last 7 years). Leave null if user wants from listing date."},
                    "from_listing": {"type": "boolean", "description": "If true, calculate returns from the stock's listing date to today. Use when user asks 'from when it listed', 'from listing date', or 'since IPO'."}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "document_expense_from_rag",
            "description": "Parse a bill, receipt, or invoice document (PDF/image) and automatically extract expense details to add to the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_base64": {"type": "string", "description": "Base64-encoded document content"},
                    "filename": {"type": "string", "description": "Original filename", "default": "receipt.pdf"},
                },
                "required": ["document_base64"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_expenses",
            "description": "List expenses for a time period. Use this to find expense IDs when user wants to delete or update a specific expense. Returns list of expenses with IDs, dates, amounts, categories, and merchants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_expense",
            "description": "Delete an expense by its ID. When user says 'remove this expense', 'delete this expense', 'delete the last expense', or 'remove expense', you MUST first call list_expenses to find the expense ID, then call delete_expense with that ID. If user says 'last expense' or 'recent expense', get the most recent expenses and delete the first one. If user mentions specific details (amount, merchant, date), use list_expenses to find matching expense, then delete it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expense_id": {"type": "string", "description": "The expense ID to delete. You must first call list_expenses to find the expense ID if user doesn't provide it directly."},
                },
                "required": ["expense_id"],
            },
        },
    },
]


def _ensure_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


async def _call_groq(messages: List[Dict[str, Any]], use_tools: bool = True):
    client = _ensure_client()
    kwargs: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more consistent, accurate responses
        "max_tokens": 2000,  # Allow longer responses when needed
    }
    if use_tools:
        kwargs["tools"] = CHAT_TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"
    return await asyncio.to_thread(client.chat.completions.create, **kwargs)


async def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    session_id: Optional[str],
    api_key: Optional[str],
    user_message: Optional[str] = None,
) -> Dict[str, Any]:
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return {"status": "error", "message": f"Tool {name} not available."}

    name, arguments = _maybe_upgrade_quick_add(name, arguments)
    kwargs = _normalize_expense_tool_args(user_message, name, dict(arguments or {}))

    fn = getattr(tool, "fn", tool)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    if session_id and sig and "session_id" in sig.parameters and "session_id" not in kwargs:
        kwargs["session_id"] = session_id
    if api_key and sig and "api_key" in sig.parameters and "api_key" not in kwargs:
        kwargs["api_key"] = api_key

    result = await tool(**kwargs)
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return result
    return {"status": "success", "data": result}


def _coerce_tool_payload(result: Any) -> Any:
    """Normalize MCP tool results (some callers wrap lists in {status, data})."""
    if isinstance(result, dict):
        if result.get("status") == "error":
            return result
        if result.get("status") == "success" and "data" in result:
            return result["data"]
    return result


STOCK_QUERY_MARKERS = (
    "stock", "share", "ticker", "ipo", "listing", "listed", "market cap",
    "share price", "stock price", "should i buy", "good time to buy", "good to buy",
    "worth buying", "worth investing", "invest in", "buy ", "sell ",
    "returns", "return ", "p/e", "pe ratio", "dividend", "52-week", "52 week",
    "yahoo", "nse", "bse", "nasdaq", "nyse", "etf", "mutual fund",
)

INVESTMENT_ADVICE_MARKERS = (
    "should i buy", "should i sell", "good time to buy", "good tiem to buy",
    "good to buy", "worth buying", "worth investing", "recommend", "advice",
    "buy it", "buy now", "invest now", "is it a good", "is it good",
    "time to buy", "tiem to buy", "should i invest", "worth it to buy",
)

TECHNICAL_STOCK_MARKERS = (
    "analyze", "analysis", "rsi", "macd", "bollinger", "candlestick",
    "buy sell", "sell signal", "buy signal", "hold signal", "technical",
    "support", "resistance", "projection", "forecast", "smart stock",
    "chart", "indicator", "overbought", "oversold",
)


def _is_stock_query(message: str) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in STOCK_QUERY_MARKERS)


def _is_investment_advice_query(message: str) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in INVESTMENT_ADVICE_MARKERS)


def _is_technical_stock_analysis_query(message: str) -> bool:
    lower = (message or "").lower()
    if not any(marker in lower for marker in TECHNICAL_STOCK_MARKERS):
        return False
    return _extract_symbol_from_message(message) is not None or any(
        word in lower for word in ("stock", "share", "ticker", "equity")
    )


def _format_stock_analysis_reply(data: Dict[str, Any]) -> str:
    sym = data.get("currency_symbol") or _currency_symbol_for_ticker(data.get("ticker", ""))
    sig = data.get("signal") or {}
    proj = data.get("projection") or {}
    ticker = data.get("ticker", "")

    lines = [f"Smart Stock Analysis — {data.get('name', ticker)} ({ticker})"]
    if data.get("price") is not None:
        lines.append(f"Price: {sym}{float(data['price']):,.2f}")
    if data.get("change_pct") is not None:
        sign = "+" if (data.get("change") or 0) >= 0 else ""
        lines.append(f"Day change: {sign}{data.get('change_pct')}%")

    lines.append(f"Signal: {sig.get('signal', 'HOLD')} ({sig.get('confidence', 0)}% confidence)")
    if data.get("rsi") is not None:
        lines.append(f"RSI: {data['rsi']} | MACD: {data.get('macd', 'N/A')}")
    if data.get("support") is not None or data.get("resistance") is not None:
        lines.append(
            f"Support: {sym}{data.get('support', 'N/A')} | "
            f"Resistance: {sym}{data.get('resistance', 'N/A')}"
        )

    projected = proj.get("projected") or []
    if projected:
        lines.append(
            f"30-day projection: {sym}{projected[-1]:,.2f} ({proj.get('trend', 'neutral')} trend, "
            f"R²={proj.get('r_squared', 'N/A')})"
        )

    reasons = sig.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Key signals:")
        for reason in reasons[:5]:
            lines.append(f"  • {reason}")

    lines.append("")
    lines.append(f"Full interactive chart: /stocks/page?ticker={ticker}")
    lines.append("Educational analysis only — not financial advice.")
    return "\n".join(lines)


async def _maybe_handle_technical_stock_analysis(
    message: str, session_id: Optional[str], api_key: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Route RSI/MACD/signal/chart requests to smart_stock_analyze."""
    if not _is_technical_stock_analysis_query(message):
        return None

    symbol_query = _extract_symbol_from_message(message)
    if not symbol_query:
        return None

    from stock_analyzer import resolve_ticker

    symbol_query = resolve_ticker(symbol_query)

    result = await _execute_tool(
        "smart_stock_analyze",
        {"ticker": symbol_query},
        session_id,
        api_key,
        user_message=message,
    )
    if result.get("status") != "success":
        return {
            "reply": result.get("message", f"Could not analyze '{symbol_query}'."),
            "tool_results": [{"tool": "smart_stock_analyze", "result": result}],
        }

    return {
        "reply": _format_stock_analysis_reply(result),
        "tool_results": [{"tool": "smart_stock_analyze", "result": result}],
    }


def _extract_symbol_from_message(message: str) -> Optional[str]:
    """Extract company name or ticker from a stock-related message."""
    import re

    text = (message or "").strip()
    if not text:
        return None

    dotted = re.search(r"\b([A-Z][A-Z0-9]{1,11}\.[A-Z]{1,4})\b", text)
    if dotted:
        return dotted.group(1)

    ticker = re.search(r"\b([A-Z]{2,6}(?:-[A-Z])?)\b", text)
    if ticker:
        candidate = ticker.group(1)
        skip = {"IT", "US", "UK", "AI", "API", "NS", "BO", "ETF", "IPO", "PE", "IS"}
        if candidate not in skip:
            return candidate

    patterns = [
        r"(?:^|\b)([a-zA-Z][a-zA-Z0-9&.'\-]{2,40})\s+(?:share|stock)\s+price",
        r"(?:should i buy|is it good|is it a good|worth buying)\s+([a-zA-Z][a-zA-Z0-9\s&.'\-]{2,30}?)(?:\s+stock|\s+share|\?|$|,|\s+or\b)",
        r"(?:price of|share price of|stock price of|returns? (?:for|of)|data (?:for|on))\s+([a-zA-Z][a-zA-Z0-9\s&.'\-]{1,40}?)(?:\s+share|\s+stock|\?|$|,|\s+and|\s+is\b)",
        r"(?:tell me|show me|about)\s+([a-zA-Z][a-zA-Z0-9\s&.'\-]{1,40}?)(?:\s+share|\s+stock|\s+price|\?|$|,|\s+and|\s+is\b)",
        r"([a-zA-Z][a-zA-Z0-9&.'\-]{2,30})\s+stock\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = match.group(1).strip(" .,'\"")
            noise = {"the", "this", "that", "it", "me", "my", "a", "an", "please", "tell", "show"}
            words = [w for w in query.split() if w.lower() not in noise]
            cleaned = " ".join(words).strip()
            if len(cleaned) >= 2:
                return cleaned

    return None


def _currency_symbol_for_ticker(symbol: str) -> str:
    symbol_upper = (symbol or "").upper()
    if ".NS" in symbol_upper or ".BO" in symbol_upper:
        return "₹"
    if ".L" in symbol_upper:
        return "£"
    if ".T" in symbol_upper or ".TWO" in symbol_upper or ".SS" in symbol_upper or ".SZ" in symbol_upper:
        return "¥"
    if ".PA" in symbol_upper or ".DE" in symbol_upper or ".F" in symbol_upper:
        return "€"
    if ".HK" in symbol_upper:
        return "HK$"
    return "$"


def _format_market_cap(value: float, currency_symbol: str) -> str:
    if value is None:
        return ""
    if value >= 1e12:
        return f"{currency_symbol}{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{currency_symbol}{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{currency_symbol}{value / 1e6:.2f}M"
    return f"{currency_symbol}{value:,.2f}"


def _format_comprehensive_stock_reply(
    yahoo_result: Dict[str, Any],
    returns_result: Optional[Dict[str, Any]],
    *,
    is_advice_query: bool = False,
) -> str:
    """Build factual stock research reply — never buy/sell advice."""
    if yahoo_result.get("status") != "success":
        return yahoo_result.get("message", "Could not fetch stock data.")

    data = yahoo_result.get("data", {})
    symbol = data.get("symbol", "")
    name = data.get("name", symbol)
    currency = data.get("currency", "USD")
    csym = data.get("currency_symbol") or {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}.get(
        currency, _currency_symbol_for_ticker(symbol)
    )
    price = data.get("price", 0)

    lines = [f"📊 {name} ({symbol})", ""]
    lines.append(f"Current price: {csym}{price:,.2f}")

    cap = data.get("market_cap")
    if cap:
        lines.append(f"Market cap: {_format_market_cap(cap, csym)}")

    sector = data.get("sector")
    industry = data.get("industry")
    if sector or industry:
        lines.append(f"Sector: {sector or '—'} | Industry: {industry or '—'}")

    exchange = data.get("exchange")
    if exchange:
        lines.append(f"Exchange: {exchange}")

    listing = data.get("listing_date")
    if listing:
        lines.append(f"Listing / IPO date: {listing}")

    week52 = data.get("fifty_two_week_range") or {}
    if week52.get("low") and week52.get("high"):
        lines.append(f"52-week range: {csym}{week52['low']:,.2f} – {csym}{week52['high']:,.2f}")

    pe = data.get("pe_ratio")
    if pe:
        lines.append(f"P/E ratio: {pe:.2f}")

    div = data.get("dividend_yield")
    if div:
        lines.append(f"Dividend yield: {div:.2f}%")

    returns = data.get("returns") or {}
    ret_parts = []
    for label, key in [
        ("1D", "1d"), ("1W", "1w"), ("1M", "1m"), ("3M", "3m"),
        ("6M", "6m"), ("1Y", "1y"), ("3Y", "3y"),
    ]:
        val = returns.get(key)
        if val is not None:
            ret_parts.append(f"{label}: {val:+.2f}%")
    if ret_parts:
        lines.append("")
        lines.append("Recent returns: " + " | ".join(ret_parts))

    if returns_result and returns_result.get("status") == "success":
        lines.append("")
        lines.append("📈 Historical growth (since listing / max available)")
        if returns_result.get("listing_date"):
            lines.append(f"Data from: {returns_result['listing_date']}")
        lines.append(
            f"Period: {returns_result.get('start_date', '')} → {returns_result.get('end_date', '')}"
        )
        lines.append(
            f"Start → End: {csym}{returns_result.get('start_price', 0):,.2f} → "
            f"{csym}{returns_result.get('end_price', 0):,.2f}"
        )
        total = returns_result.get("total_return_percent")
        if total is not None:
            lines.append(f"Total return since start: {total:+.2f}%")

        yearly = returns_result.get("yearly_returns") or []
        if yearly:
            lines.append("")
            lines.append("Year-by-year returns:")
            for yr in yearly[-8:]:
                y = yr.get("year", "")
                r = yr.get("return_percent", 0)
                lines.append(f"  • {y}: {r:+.2f}%")

    lines.append("")
    if is_advice_query:
        lines.append(
            "⚠️ I cannot provide buy/sell recommendations or say whether it is a good time to invest. "
            "The figures above are factual market data from Yahoo Finance. "
            "Please do your own research and consult a licensed financial advisor before investing."
        )
    else:
        lines.append(
            "ℹ️ Data sourced from Yahoo Finance. This is informational only — not investment advice."
        )

    return "\n".join(lines)


async def _fetch_stock_research(
    symbol_query: str,
    message: str,
    session_id: Optional[str],
    api_key: Optional[str],
) -> Dict[str, Any]:
    """Fetch yahoo_finance + get_stock_returns for any Yahoo-listed symbol."""
    yahoo_result, returns_result = await asyncio.gather(
        _execute_tool(
            "yahoo_finance", {"symbol": symbol_query}, session_id, api_key, user_message=message
        ),
        _execute_tool(
            "get_stock_returns",
            {"symbol": symbol_query, "from_listing": True},
            session_id,
            api_key,
            user_message=message,
        ),
    )
    return {"yahoo": yahoo_result, "returns": returns_result}


async def _maybe_handle_stock_query(
    message: str, session_id: Optional[str], api_key: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive stock research for any Yahoo-listed symbol."""
    if not _is_stock_query(message):
        return None

    symbol_query = _extract_symbol_from_message(message)
    if not symbol_query:
        return None

    from stock_analyzer import resolve_ticker

    symbol_query = resolve_ticker(symbol_query)

    is_advice = _is_investment_advice_query(message)
    research = await _fetch_stock_research(symbol_query, message, session_id, api_key)
    yahoo_result = research["yahoo"]
    returns_result = research["returns"]

    if yahoo_result.get("status") != "success":
        return {
            "reply": yahoo_result.get(
                "message", f"Could not find data for '{symbol_query}' on Yahoo Finance."
            ),
            "tool_results": [
                {"tool": "yahoo_finance", "result": yahoo_result},
                {"tool": "get_stock_returns", "result": returns_result},
            ],
        }

    reply = _format_comprehensive_stock_reply(
        yahoo_result, returns_result, is_advice_query=is_advice
    )
    return {
        "reply": reply,
        "tool_results": [
            {"tool": "yahoo_finance", "result": yahoo_result},
            {"tool": "get_stock_returns", "result": returns_result},
        ],
    }


async def _supplement_stock_tool_results(
    message: str,
    tool_results: List[Dict[str, Any]],
    session_id: Optional[str],
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    """Ensure stock queries always include historical returns, not just live price."""
    if not _is_stock_query(message):
        return tool_results

    has_yahoo = any(
        r.get("tool") == "yahoo_finance" and r.get("result", {}).get("status") == "success"
        for r in tool_results
    )
    has_returns = any(
        r.get("tool") == "get_stock_returns" and r.get("result", {}).get("status") == "success"
        for r in tool_results
    )
    if not has_yahoo or has_returns:
        return tool_results

    symbol = None
    for r in tool_results:
        if r.get("tool") == "yahoo_finance":
            data = r.get("result", {}).get("data", {})
            symbol = data.get("symbol")
            break
    if not symbol:
        symbol = _extract_symbol_from_message(message)
    if not symbol:
        return tool_results

    returns_result = await _execute_tool(
        "get_stock_returns",
        {"symbol": symbol, "from_listing": True},
        session_id,
        api_key,
        user_message=message,
    )
    return tool_results + [{"tool": "get_stock_returns", "result": returns_result}]


def _extract_price_range(message: str) -> Optional[tuple[float, float]]:
    """Extract price range from user message (e.g., '100 to 500', 'between 50-200')."""
    import re
    message_lower = message.lower()
    
    # Patterns: "100 to 500", "100-500", "between 100 and 500", "100 rupee to 500 rupee"
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*(?:rupee|rs|₹|rupees)?',
        r'between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*(?:rupee|rs|₹|rupees)?',
        r'range\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            try:
                min_price = float(match.group(1))
                max_price = float(match.group(2))
                if min_price > max_price:
                    min_price, max_price = max_price, min_price
                return (min_price, max_price)
            except ValueError:
                continue
    
    return None


async def chat_with_agent(
    message: str,
    *,
    session_id: Optional[str],
    api_key: Optional[str],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not message:
        raise ValueError("Message is required.")

    expense_builtin = await _maybe_handle_expense_addition(message, session_id, api_key)
    if expense_builtin is not None:
        return expense_builtin

    builtin = await _maybe_handle_builtin_query(message, session_id, api_key)
    if builtin is not None:
        return builtin

    technical_analysis = await _maybe_handle_technical_stock_analysis(message, session_id, api_key)
    if technical_analysis is not None:
        return technical_analysis

    stock_analysis = await _maybe_handle_stock_query(message, session_id, api_key)
    if stock_analysis is not None:
        return stock_analysis

    gmail_result = await _maybe_handle_gmail_query(message, session_id, api_key)
    if gmail_result is not None:
        return gmail_result

    # Extract price range if mentioned
    price_range = _extract_price_range(message)
    
    # Enhance system prompt if price range is detected
    enhanced_prompt = _build_system_prompt()
    if price_range:
        min_price, max_price = price_range
        enhanced_prompt += f"\n\nIMPORTANT: User has requested stocks in price range {min_price}-{max_price}. You MUST only recommend stocks whose current price falls within this range. If a stock's price is outside this range, DO NOT recommend it. Instead, explain that the stock doesn't match the requested price range."

    convo: List[Dict[str, Any]] = [{"role": "system", "content": enhanced_prompt}]
    if history:
        convo.extend(history)

    convo.append({"role": "user", "content": message})

    initial = await _call_groq(convo, use_tools=True)
    first_choice = initial.choices[0].message

    assistant_payload: Dict[str, Any] = {
        "role": first_choice.role or "assistant",
        "content": first_choice.content or "",
    }

    tool_calls = getattr(first_choice, "tool_calls", None)
    if tool_calls:
        # Add assistant message with tool calls
        assistant_payload["tool_calls"] = []
        for tool_call in tool_calls:
            assistant_payload["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        convo.append(assistant_payload)
        
        tool_results = []
        tool_messages = []
        max_iterations = 3  # Reduced to prevent excessive loops
        iteration = 0

        # Process tool calls in rounds
        while tool_calls and iteration < max_iterations:
            iteration += 1
            
            # Execute all tool calls in this round IN PARALLEL for better performance
            async def execute_single_tool(tool_call):
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                
                exec_result = await _execute_tool(fn_name, args, session_id, api_key, user_message=message)
                return {
                    "tool_call": tool_call,
                    "fn_name": fn_name,
                    "result": exec_result
                }
            
            # Execute all tools in parallel
            parallel_results = await asyncio.gather(*[execute_single_tool(tc) for tc in tool_calls])
            
            # Process results and build tool messages
            for result in parallel_results:
                fn_name = result["fn_name"]
                exec_result = result["result"]
                tool_call = result["tool_call"]
                result_text = json.dumps(exec_result)
                tool_results.append({"tool": fn_name, "result": exec_result})
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": result_text,
                    }
                )
            
            # Add tool results to conversation
            convo.extend(tool_messages)
            tool_messages = []  # Reset for next round
            
            # Get LLM's response to tool results
            follow_up = await _call_groq(convo, use_tools=True)
            follow_up_message = follow_up.choices[0].message
            tool_calls = getattr(follow_up_message, "tool_calls", None)
            
            if tool_calls:
                # LLM wants to make more tool calls - add assistant message
                assistant_payload = {
                    "role": follow_up_message.role or "assistant",
                    "content": follow_up_message.content or "",
                    "tool_calls": []
                }
                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                convo.append(assistant_payload)
            else:
                # No more tool calls - get final reply
                llm_reply = follow_up_message.content or "Done."
                break
        
        # If we still don't have a reply, get one without tools
        if not llm_reply or llm_reply == "Done.":
            final_call = await _call_groq(convo, use_tools=False)
            llm_reply = final_call.choices[0].message.content or "Task completed."

        tool_results = await _supplement_stock_tool_results(
            message, tool_results, session_id, api_key
        )

        formatted_reply = _format_user_friendly_reply(
            llm_reply, tool_results, original_message=message
        )

        # Stock / investment queries: always use factual tool data, never LLM buy/sell opinions
        if _is_stock_query(message) and formatted_reply:
            reply = formatted_reply
        elif formatted_reply and formatted_reply != llm_reply and len(formatted_reply) > 20:
            reply = formatted_reply
        else:
            cleaned_llm = _clean_json_from_reply(llm_reply)
            if (
                cleaned_llm
                and len(cleaned_llm) > 10
                and not _is_investment_advice_query(message)
                and not any(x in cleaned_llm.lower() for x in ["status", "error", "success", "{", "}"])
            ):
                reply = cleaned_llm
            else:
                reply = formatted_reply if formatted_reply else "Task completed successfully."

        return {"reply": reply, "tool_results": tool_results}

    # No tool call, just reply naturally
    reply = assistant_payload["content"] or "Let me know how I can assist."
    # Clean any accidental JSON from natural responses
    reply = _clean_json_from_reply(reply)
    return {"reply": reply, "tool_results": []}


async def _maybe_handle_expense_addition(
    message: str, session_id: Optional[str], api_key: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Handle simple expense logging directly — avoids LLM date hallucination."""
    text = (message or "").strip()
    if not text:
        return None

    lower = text.lower()
    query_markers = [
        "show", "list", "tell me", "what are", "how much", "how many", "summary",
        "where did i", "where i spent", "where did my", "what did i spend",
        "what have i spent", "my expenses", "my spending", "money spent",
        "money did i", "total spend", "expense track", "breakdown",
        "delete", "remove", "check budget", "week summary", "month summary",
        "stock", "share", "price", "return", "investment", "budget status",
    ]
    if any(marker in lower for marker in query_markers):
        return None

    add_markers = ["spent", "paid", "bought", "purchase", "purchased", "add "]
    if not any(marker in lower for marker in add_markers):
        return None

    amount = _extract_amount(text)
    if amount is None:
        return None

    parsed = _extract_expense_details(text, amount)
    if not parsed:
        return None

    result = await _execute_tool("add_expense", parsed, session_id, api_key, user_message=message)
    if result.get("status") != "success":
        return None

    formatted = _format_user_friendly_reply(
        "",
        [{"tool": "add_expense", "result": result}],
        original_message=message,
    )
    return {"reply": formatted, "tool_results": [{"tool": "add_expense", "result": result}]}


def _is_gmail_fetch_query(message: str) -> bool:
    text = (message or "").lower()
    if not any(k in text for k in ("gmail", "email", "inbox", "e-mail", "mail")):
        return False
    return any(
        k in text
        for k in (
            "fetch", "sync", "import", "scan", "read", "pull", "connect",
            "track", "get", "load", "check", "expense", "bill", "receipt",
        )
    )


async def _maybe_handle_gmail_query(
    message: str, session_id: Optional[str], api_key: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fetch expenses from Gmail then show expense summary."""
    if not _is_gmail_fetch_query(message):
        return None

    sync_result = await _execute_tool(
        "sync_gmail_bills",
        {"max_messages": 50},
        session_id,
        api_key,
    )
    tool_results = [{"tool": "sync_gmail_bills", "result": sync_result}]

    if isinstance(sync_result, dict) and sync_result.get("status") == "error":
        err = sync_result.get("message", "Gmail sync failed")
        hint = ""
        if "gmail" in err.lower() or "connect" in err.lower() or "permission" in err.lower():
            hint = (
                "\n\nTo fix: click **Sync with Google** in the top navigation, sign in again, "
                "and allow Gmail access. Then retry or use **Expenses → Import from Gmail**."
            )
        return {"reply": f"❌ {err}{hint}", "tool_results": tool_results}

    today = _now_local()
    start_date = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    expenses = await _execute_tool(
        "list_expenses",
        {"start_date": start_date, "end_date": end_date},
        session_id,
        api_key,
    )
    tool_results.append({"tool": "list_expenses", "result": expenses})

    imported = sync_result.get("imported", 0) if isinstance(sync_result, dict) else 0
    skipped = sync_result.get("skipped", 0) if isinstance(sync_result, dict) else 0
    purged = sync_result.get("purged", 0) if isinstance(sync_result, dict) else 0
    lines = [
        f"📧 Gmail scan complete — imported **{imported}** new bill(s) ({skipped} skipped).",
    ]
    if purged:
        lines.append(f"Removed **{purged}** invalid promo/unconfirmed Gmail import(s).")
    lines.append("")

    if isinstance(sync_result, dict) and sync_result.get("items"):
        lines.append("Newly imported:")
        for item in sync_result["items"][:8]:
            lines.append(
                f"  • {item.get('date', '')}: ₹{item.get('amount', 0):,.2f} "
                f"— {item.get('merchant', 'Unknown')} ({item.get('category', 'Other')})"
            )
        lines.append("")

    if isinstance(expenses, list) and expenses:
        lines.append(f"Your expense track ({start_date} to {end_date}):")
        total = 0
        for exp in expenses[:20]:
            date_str = exp.get("date", "")
            amount = float(exp.get("amount", 0) or 0)
            category = exp.get("category", "Other")
            merchant = exp.get("merchant", "")
            src = (exp.get("metadata") or {}).get("source", "")
            src_tag = " [Gmail]" if src == "gmail_sync" else ""
            line = f"  • {date_str}: ₹{amount:,.2f} ({category})"
            if merchant:
                line += f" — {merchant}"
            line += src_tag
            lines.append(line)
            total += amount
        if len(expenses) > 20:
            lines.append(f"  … and {len(expenses) - 20} more")
        lines.append(f"\nTotal tracked: ₹{total:,.2f}")
    elif imported == 0:
        lines.append(
            "No new bill emails found in Gmail (last 6 months). "
            "Try forwarding receipts to your inbox or add expenses manually."
        )
    else:
        lines.append("Expenses imported — open the **Expenses** tab to see the full list.")

    return {"reply": "\n".join(lines), "tool_results": tool_results}


async def _maybe_handle_builtin_query(message: str, session_id: Optional[str], api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Handle spending/expense queries directly for fast, accurate replies."""
    if not _is_spending_query(message):
        return None

    start_date, end_date, period = _resolve_spending_period(message)
    tool_results: List[Dict[str, Any]] = []

    summary = _coerce_tool_payload(
        await _execute_tool(
            "summarize",
            {"start_date": start_date, "end_date": end_date},
            session_id,
            api_key,
        )
    )
    tool_results.append({"tool": "summarize", "result": summary})

    if isinstance(summary, dict) and summary.get("status") == "error":
        return {
            "reply": summary.get("message", "Could not load your spending data."),
            "tool_results": tool_results,
        }

    wants_detail = _wants_expense_detail_list(message)
    expenses = None
    if wants_detail:
        expenses = _coerce_tool_payload(
            await _execute_tool(
                "list_expenses",
                {"start_date": start_date, "end_date": end_date},
                session_id,
                api_key,
            )
        )
        tool_results.append({"tool": "list_expenses", "result": expenses})
        if isinstance(expenses, dict) and expenses.get("status") == "error":
            return {
                "reply": expenses.get("message", "Could not load expense details."),
                "tool_results": tool_results,
            }

    reply = _format_spending_reply(
        summary if isinstance(summary, list) else [],
        expenses if isinstance(expenses, list) else None,
        period,
        start_date,
        end_date,
        include_details=wants_detail,
    )
    return {"reply": reply, "tool_results": tool_results}


SPENDING_QUERY_MARKERS = (
    "how much",
    "how many",
    "total spend",
    "total expense",
    "money spent",
    "money did i spend",
    "did i spend",
    "have i spent",
    "my spending",
    "my expenses",
    "my expense",
    "spending summary",
    "expense summary",
    "week summary",
    "month summary",
    "where did i spend",
    "where i spend",
    "where did my money",
    "what did i spend",
    "what have i spent",
    "show my expense",
    "show expenses",
    "list expense",
    "list my expense",
    "expense track",
    "track of my expense",
    "breakdown",
    "category wise",
    "by category",
    "spent money",
    "spending this",
)


def _is_spending_query(message: str) -> bool:
    import re

    text = (message or "").lower().strip()
    if not text:
        return False

    if re.search(r"\b(add|log|record|enter|save)\b", text) and re.search(
        r"\b(expense|spent|spending|rupees|rs\.?|₹|\$)\b", text
    ):
        if not re.search(r"\b(show|list|how much|summary|total|where)\b", text):
            return False

    if any(marker in text for marker in SPENDING_QUERY_MARKERS):
        return True

    if any(q in text for q in ("show", "list", "tell", "give", "fetch", "get", "check")):
        if any(w in text for w in ("expense", "expenses", "spent", "spending", "spend")):
            return True

    if "how much" in text and any(w in text for w in ("spend", "spent", "expense", "money")):
        return True

    if any(w in text for w in ("spent", "spending", "expenses")) and "?" in text:
        return True

    return False


def _wants_expense_detail_list(message: str) -> bool:
    text = (message or "").lower()
    detail_markers = (
        "where",
        "list",
        "show each",
        "show all",
        "every expense",
        "each expense",
        "merchant",
        "details",
        "breakdown by",
        "itemized",
        "track",
        "dates",
        "when did",
    )
    if any(marker in text for marker in detail_markers):
        return True
    if "show" in text and "summary" not in text and any(
        w in text for w in ("expense", "spent", "spending")
    ):
        return True
    return False


def _resolve_spending_period(message: str) -> tuple[str, str, str]:
    text = (message or "").lower()
    today = _now_local()

    if "today" in text:
        d = today.strftime("%Y-%m-%d")
        return d, d, "today"
    if "yesterday" in text:
        d = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        return d, d, "yesterday"
    if "last week" in text or "past week" in text:
        start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        return start, today.strftime("%Y-%m-%d"), "last 7 days"
    if "this week" in text:
        start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        return start, today.strftime("%Y-%m-%d"), "this week"
    if "last month" in text or "previous month" in text:
        first_this = datetime(today.year, today.month, 1, tzinfo=APP_TZ)
        last_month_end = first_this - timedelta(days=1)
        start = datetime(last_month_end.year, last_month_end.month, 1, tzinfo=APP_TZ)
        return start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d"), "last month"
    if "this month" in text or "current month" in text:
        start, end = _current_month_range()
        return start, end, "this month"
    if "last 30 days" in text or "past 30 days" in text or "30 days" in text:
        start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        return start, today.strftime("%Y-%m-%d"), "last 30 days"

    start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    return start, today.strftime("%Y-%m-%d"), "last 30 days"


def _format_spending_reply(
    summary: List[Dict[str, Any]],
    expenses: Optional[List[Dict[str, Any]]],
    period: str,
    start_date: str,
    end_date: str,
    *,
    include_details: bool,
) -> str:
    currency_symbol = "₹"
    total = 0.0
    count = 0
    for row in summary:
        total += float(row.get("total", 0) or 0)
        count += int(row.get("count", 0) or 0)

    lines = [f"Your spending for {period} ({start_date} to {end_date}):"]
    lines.append(f"Total: {currency_symbol}{total:,.2f} across {count} transaction(s)")
    lines.append("")

    if summary:
        lines.append("By category:")
        for row in summary[:8]:
            cat = row.get("category", "Other")
            cat_total = float(row.get("total", 0) or 0)
            cat_count = int(row.get("count", 0) or 0)
            lines.append(f"  • {cat}: {currency_symbol}{cat_total:,.2f} ({cat_count} txn)")
        if len(summary) > 8:
            lines.append(f"  … and {len(summary) - 8} more categories")
        lines.append("")

    if include_details and expenses:
        if expenses:
            lines.append("Recent transactions:")
            for exp in expenses[:15]:
                date_str = exp.get("date", "")
                amount = float(exp.get("amount", 0) or 0)
                category = exp.get("category", "Other")
                merchant = exp.get("merchant", "")
                try:
                    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d %b")
                except ValueError:
                    formatted_date = date_str or "?"
                line = f"  • {formatted_date}: {currency_symbol}{amount:,.2f} — {category}"
                if merchant:
                    line += f" @ {merchant}"
                lines.append(line)
            if len(expenses) > 15:
                lines.append(f"  … and {len(expenses) - 15} more")
        else:
            lines.append("No individual transactions found for this period.")
    elif total == 0:
        lines.append(
            "No expenses recorded yet. Add one in chat (e.g. 'Add 150 rupees for snacks today') "
            "or use the Expenses tab."
        )

    return "\n".join(lines)


def _current_month_range() -> tuple[str, str]:
    today = _now_local()
    start = datetime(today.year, today.month, 1, tzinfo=APP_TZ)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _has_explicit_date(text: str) -> bool:
    """Return True if the user message contains a specific calendar date."""
    if not text:
        return False
    text_lower = text.lower()
    if any(w in text_lower for w in ("today", "yesterday", "this week", "this month", "last week")):
        return True
    import re
    if re.search(
        r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)",
        text_lower,
    ):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}", text):
        return True
    return bool(search_dates(text, settings={"RELATIVE_BASE": _now_local()}))


def _normalize_expense_tool_args(
    user_message: Optional[str], name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Override LLM-provided dates with dates parsed from the user's message."""
    if name not in ("add_expense", "quick_add_expense"):
        return arguments

    args = dict(arguments or {})
    if name == "quick_add_expense":
        return args

    if user_message:
        resolved = _extract_date(user_message)
        text_lower = user_message.lower()
        relative_terms = ("today", "yesterday", "this week", "this month", "last week")
        if any(term in text_lower for term in relative_terms) or not _has_explicit_date(user_message):
            args["date"] = resolved
    elif not args.get("date"):
        args["date"] = _today_str()
    return args


def _format_user_friendly_reply(reply: str, tool_results: List[Dict[str, Any]], original_message: str = None) -> str:
    """Format the reply to be user-friendly, hiding backend details."""
    if not tool_results:
        return _clean_json_from_reply(reply)
    
    # Extract price range from original message if provided
    price_range = _extract_price_range(original_message) if original_message else None
    
    # Collect all expense additions
    expense_additions = []
    other_tool_results = []
    
    for result in tool_results:
        tool_name = result.get("tool", "")
        tool_result = result.get("result", {})
        
        if tool_name in ["add_expense", "quick_add_expense"]:
            if tool_result.get("status") == "success":
                amount = tool_result.get("amount", 0)
                category = tool_result.get("category", "expense")
                date = tool_result.get("date", "")
                currency = "INR"  # Default
                metadata = tool_result.get("metadata", {})
                if metadata and isinstance(metadata, dict):
                    currency = metadata.get("currency", "INR")
                
                # Validate and format date
                formatted_date = "today"
                if date:
                    try:
                        if len(date) == 10 and date.count('-') == 2:
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                            formatted_date = date_obj.strftime("%d %b %Y")
                        else:
                            formatted_date = "today"
                    except:
                        formatted_date = "today"
                
                currency_symbol = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}.get(currency, currency)
                expense_additions.append({
                    "amount": amount,
                    "category": category,
                    "date": formatted_date,
                    "currency_symbol": currency_symbol
                })
        else:
            other_tool_results.append(result)
    
    # Format response based on number of expenses added
    if expense_additions:
        if len(expense_additions) == 1:
            exp = expense_additions[0]
            return f"✅ Expense added successfully!\n\n{exp['currency_symbol']}{exp['amount']:,.2f} for {exp['category']} on {exp['date']}"
        else:
            # Multiple expenses
            response = f"✅ Added {len(expense_additions)} expenses successfully!\n\n"
            total = 0
            for exp in expense_additions:
                response += f"• {exp['currency_symbol']}{exp['amount']:,.2f} for {exp['category']} on {exp['date']}\n"
                total += exp['amount']
            response += f"\nTotal: {expense_additions[0]['currency_symbol']}{total:,.2f}"
            return response
    
    # Handle delete_expense results
    delete_results = [r for r in other_tool_results if r.get("tool") == "delete_expense"]
    if delete_results:
        for result in delete_results:
            tool_result = result.get("result", {})
            if tool_result.get("status") == "success":
                return "✅ Expense deleted successfully!"
            elif tool_result.get("status") == "error":
                error_msg = tool_result.get("message", "Failed to delete expense")
                return f"❌ {error_msg}"
    
    # Combined stock research (yahoo_finance + get_stock_returns)
    yahoo_hits = [
        r for r in other_tool_results
        if r.get("tool") == "yahoo_finance" and r.get("result", {}).get("status") == "success"
    ]
    if yahoo_hits:
        yahoo_result = yahoo_hits[0]["result"]
        returns_hits = [
            r for r in other_tool_results
            if r.get("tool") == "get_stock_returns"
        ]
        returns_result = returns_hits[0]["result"] if returns_hits else None
        is_advice = _is_investment_advice_query(original_message or "")
        return _format_comprehensive_stock_reply(
            yahoo_result, returns_result, is_advice_query=is_advice
        )

    # Handle other tool results (only if no expenses were added and no deletions)
    for result in other_tool_results:
        tool_name = result.get("tool", "")
        tool_result = result.get("result", {})
        
        if tool_name == "set_budget":
            if tool_result.get("status") == "success":
                category = tool_result.get("category", "")
                amount = tool_result.get("amount", 0)
                return f"✅ Budget set successfully!\n\n{category}: ₹{amount:,.2f} per month"
        
        elif tool_name == "add_income":
            if tool_result.get("status") == "success":
                amount = tool_result.get("amount", 0)
                source = tool_result.get("source", "")
                return f"✅ Income recorded!\n\n₹{amount:,.2f} from {source}"

        elif tool_name == "sync_gmail_bills":
            if tool_result.get("status") == "success":
                imported = tool_result.get("imported", 0)
                msg = tool_result.get("message", f"Imported {imported} bill(s) from Gmail.")
                items = tool_result.get("items") or []
                if items:
                    msg += "\n\nNew imports:\n"
                    for item in items[:5]:
                        msg += f"• {item.get('date')}: ₹{item.get('amount', 0):,.2f} — {item.get('merchant', '')}\n"
                return f"📧 {msg}"
            elif tool_result.get("status") == "error":
                return f"❌ {tool_result.get('message', 'Gmail sync failed')}"
        
        elif tool_name == "yahoo_finance":
            if tool_result.get("status") == "success":
                returns_hits = [
                    r for r in other_tool_results if r.get("tool") == "get_stock_returns"
                ]
                returns_result = returns_hits[0]["result"] if returns_hits else None
                is_advice = _is_investment_advice_query(original_message or "")
                return _format_comprehensive_stock_reply(
                    tool_result, returns_result, is_advice_query=is_advice
                )

        elif tool_name == "smart_stock_analyze":
            if tool_result.get("status") == "success":
                return _format_stock_analysis_reply(tool_result)
            elif tool_result.get("status") == "error":
                return f"Could not analyze stock: {tool_result.get('message', 'Unknown error')}"

        elif tool_name == "summarize":
            payload = _coerce_tool_payload(tool_result)
            if isinstance(payload, list):
                period = "the selected period"
                return _format_spending_reply(payload, None, period, "", "", include_details=False)

        elif tool_name == "list_expenses":
            payload = _coerce_tool_payload(tool_result)
            if isinstance(payload, list):
                summary_rows: List[Dict[str, Any]] = []
                cat_totals: Dict[str, Dict[str, Any]] = {}
                for exp in payload:
                    cat = exp.get("category", "Other")
                    amt = float(exp.get("amount", 0) or 0)
                    if cat not in cat_totals:
                        cat_totals[cat] = {"category": cat, "total": 0.0, "count": 0}
                    cat_totals[cat]["total"] += amt
                    cat_totals[cat]["count"] += 1
                summary_rows = sorted(cat_totals.values(), key=lambda r: r["total"], reverse=True)
                return _format_spending_reply(
                    summary_rows,
                    payload,
                    "your expenses",
                    "",
                    "",
                    include_details=True,
                )
            if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                return f"❌ {tool_result.get('message', 'Could not list expenses')}"
        
        elif tool_name == "get_stock_return_one_year":
            if tool_result.get("status") == "success":
                symbol = tool_result.get("symbol", "")
                start_price = tool_result.get("start_price", 0)
                end_price = tool_result.get("end_price", 0)
                return_percent = tool_result.get("one_year_return_percent", 0)
                
                # Detect currency symbol based on exchange suffix
                currency_symbol = "$"  # Default to USD
                symbol_upper = symbol.upper()
                if ".NS" in symbol_upper or ".BO" in symbol_upper:  # NSE/BSE (India)
                    currency_symbol = "₹"
                elif ".L" in symbol_upper:  # London Stock Exchange
                    currency_symbol = "£"
                elif ".T" in symbol_upper or ".TWO" in symbol_upper:  # Tokyo Stock Exchange
                    currency_symbol = "¥"
                elif ".PA" in symbol_upper or ".DE" in symbol_upper or ".F" in symbol_upper:  # European exchanges
                    currency_symbol = "€"
                elif ".SS" in symbol_upper or ".SZ" in symbol_upper:  # Shanghai/Shenzhen
                    currency_symbol = "¥"
                elif ".HK" in symbol_upper:  # Hong Kong
                    currency_symbol = "HK$"
                
                response = f"📊 1-Year Return for {symbol}\n"
                response += f"Start Price (1 year ago): {currency_symbol}{start_price:,.2f}\n"
                response += f"End Price (current): {currency_symbol}{end_price:,.2f}\n"
                response += f"1-Year Return: {return_percent:+.2f}%"
                
                return response
        
        elif tool_name == "get_stock_returns":
            if tool_result.get("status") == "success":
                symbol = tool_result.get("symbol", "")
                start_date = tool_result.get("start_date", "")
                end_date = tool_result.get("end_date", "")
                start_price = tool_result.get("start_price", 0)
                end_price = tool_result.get("end_price", 0)
                total_return = tool_result.get("total_return_percent", 0)
                yearly_returns = tool_result.get("yearly_returns", [])
                listing_date = tool_result.get("listing_date")
                years_analyzed = tool_result.get("years_analyzed", 0)
                
                # Detect currency symbol based on exchange suffix
                currency_symbol = "$"  # Default to USD
                symbol_upper = symbol.upper()
                if ".NS" in symbol_upper:  # NSE (India)
                    currency_symbol = "₹"
                elif ".BO" in symbol_upper:  # BSE (India)
                    currency_symbol = "₹"
                elif ".L" in symbol_upper:  # London Stock Exchange
                    currency_symbol = "£"
                elif ".T" in symbol_upper or ".TWO" in symbol_upper:  # Tokyo Stock Exchange
                    currency_symbol = "¥"
                elif ".PA" in symbol_upper or ".DE" in symbol_upper or ".F" in symbol_upper:  # European exchanges
                    currency_symbol = "€"
                elif ".SS" in symbol_upper or ".SZ" in symbol_upper:  # Shanghai/Shenzhen
                    currency_symbol = "¥"
                elif ".HK" in symbol_upper:  # Hong Kong
                    currency_symbol = "HK$"
                
                # Build response
                response = f"📊 Stock Returns Analysis for {symbol}\n\n"
                
                if listing_date:
                    response += f"📅 Listed on: {listing_date}\n"
                
                response += f"Period: {start_date} to {end_date}\n"
                response += f"Start Price: {currency_symbol}{start_price:,.2f}\n"
                response += f"End Price: {currency_symbol}{end_price:,.2f}\n"
                response += f"Total Return: {total_return:+.2f}%\n"
                response += f"Years Analyzed: {years_analyzed}\n\n"
                
                if yearly_returns:
                    response += "📈 Year-by-Year Returns:\n"
                    for yr_data in yearly_returns:
                        year = yr_data.get("year", "")
                        yr_return = yr_data.get("return_percent", 0)
                        yr_start = yr_data.get("start_price", 0)
                        yr_end = yr_data.get("end_price", 0)
                        response += f"  {year}: {yr_return:+.2f}% ({currency_symbol}{yr_start:,.2f} → {currency_symbol}{yr_end:,.2f})\n"
                
                return response
    
    # If no specific formatting, return the LLM's reply (cleaned of JSON)
    return _clean_json_from_reply(reply)


def _clean_json_from_reply(reply: str) -> str:
    """Remove any JSON blocks or backend details from the reply."""
    if not reply:
        return reply
    
    import re
    # Remove JSON code blocks
    reply = re.sub(r'```json\s*\{[^}]*\}\s*```', '', reply, flags=re.DOTALL | re.IGNORECASE)
    reply = re.sub(r'```\s*\{[^}]*\}\s*```', '', reply, flags=re.DOTALL)
    
    # Remove standalone JSON objects
    reply = re.sub(r'\{\s*"[^"]*"\s*:\s*"[^"]*"[^}]*\}', '', reply)
    
    # Remove lines with "status", "id", "message" that look like backend responses
    lines = reply.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that look like JSON key-value pairs
        if re.match(r'^\s*"[^"]*"\s*:\s*', line):
            continue
        # Skip lines with common backend fields
        if any(field in line.lower() for field in ['"status"', '"id"', '"message"', '"result"', 'add_expense', 'success', 'error']):
            if '{' in line or '}' in line:
                continue
        cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines).strip()
    
    # If we removed everything, return a simple message
    if not cleaned or len(cleaned) < 10:
        return "Task completed successfully."
    
    return cleaned


def _maybe_upgrade_quick_add(name: str, arguments: Dict[str, Any]):
    if name != "quick_add_expense":
        return name, arguments
    description = arguments.get("description") or ""
    amount = arguments.get("amount")
    parsed = _extract_expense_details(description, amount)
    if not parsed:
        return name, arguments
    return "add_expense", parsed


CATEGORY_KEYWORDS = {
    "food": "Food",
    "snack": "Food",
    "grocer": "Food",
    "restaurant": "Food",
    "coffee": "Food",
    "travel": "Travel",
    "flight": "Travel",
    "hotel": "Travel",
    "cab": "Travel",
    "uber": "Travel",
    "rent": "Rent",
    "bill": "Bills",
    "electric": "Utilities",
    "water": "Utilities",
    "shopping": "Shopping",
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "movie": "Entertainment",
    "gym": "Health",
    "doctor": "Health",
    "medic": "Health",
    "school": "Education",
    "college": "Education",
}


def _extract_expense_details(description: str, amount_hint: Optional[float]) -> Optional[Dict[str, Any]]:
    if not description and not amount_hint:
        return None
    note = description.strip()
    amount = float(amount_hint) if amount_hint is not None else _extract_amount(description)
    if amount is None:
        return None
    date_str = _extract_date(description)
    category = _detect_category(description)
    merchant = _extract_merchant(description)
    currency = _detect_currency(description)
    # Only add metadata if currency is NOT INR (default)
    metadata = {"currency": currency} if currency and currency != "INR" else {}
    return {
        "date": date_str,
        "amount": amount,
        "category": category,
        "merchant": merchant,
        "note": note or f"Expense entry recorded on {date_str}",
        "metadata": metadata if metadata else None,
    }


def _extract_amount(text: str) -> Optional[float]:
    import re

    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_date(text: str) -> str:
    if not text:
        return _today_str()
    
    text_lower = text.lower()
    today = _now_local()
    
    # Handle common phrases
    if "today" in text_lower:
        return today.strftime("%Y-%m-%d")
    elif "yesterday" in text_lower:
        from datetime import timedelta
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif "this month" in text_lower or "current month" in text_lower:
        # Use today's date for "this month"
        return today.strftime("%Y-%m-%d")
    elif "last week" in text_lower:
        from datetime import timedelta
        return (today - timedelta(days=7)).strftime("%Y-%m-%d")
    elif "this week" in text_lower:
        return today.strftime("%Y-%m-%d")
    
    # Try to parse dates like "23 nov", "nov 23", "23 november 2023", etc.
    import re
    date_pattern = r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?'
    match = re.search(date_pattern, text_lower)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)[:3].lower()
        year = int(match.group(3)) if match.group(3) else today.year
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        if month_str in month_map:
            try:
                today_naive = today.replace(tzinfo=None)
                date_obj = datetime(year, month_map[month_str], day)
                days_ahead = (date_obj - today_naive).days
                if days_ahead > 60:
                    date_obj = datetime(today.year, month_map[month_str], day)
                elif date_obj > today_naive and today.month > month_map[month_str]:
                    date_obj = datetime(today.year + 1, month_map[month_str], day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass  # Invalid date, fall through to dateparser
    
    # Try dateparser for other date formats
    # Use current date as relative base and prefer dates from past (but allow current year)
    settings = {
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": today,
        "PREFER_DAY_OF_MONTH": "first"  # Prefer earlier in month if ambiguous
    }
    results = search_dates(text, settings=settings) if text else None
    if results:
        parsed_date = results[-1][1]
        today_naive = today.replace(tzinfo=None)
        if hasattr(parsed_date, "tzinfo") and parsed_date.tzinfo is not None:
            parsed_date = parsed_date.replace(tzinfo=None)
        if (today_naive - parsed_date).days > 1825:
            parsed_date = parsed_date.replace(year=today.year)
            if parsed_date > today_naive:
                parsed_date = parsed_date.replace(year=today.year - 1)
        return parsed_date.strftime("%Y-%m-%d")
    else:
        # Default to today if no date found
        return today.strftime("%Y-%m-%d")


def _detect_category(text: str) -> str:
    lowered = text.lower()
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return category
    return "Other"


def _extract_merchant(text: str) -> str:
    lowered = text.lower()
    tokens = lowered.split()
    if "at" in tokens:
        idx = tokens.index("at")
        if idx + 1 < len(tokens):
            return tokens[idx + 1].strip(",.")
    if "from" in tokens:
        idx = tokens.index("from")
        if idx + 1 < len(tokens):
            return tokens[idx + 1].strip(",.")
    return ""


CURRENCY_KEYWORDS = {
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "pound": "GBP",
    "gbp": "GBP",
    "yen": "JPY",
    "jpy": "JPY",
    "rupee": "INR",
    "inr": "INR",
}

def _detect_currency(text: str) -> str:
    """Detect currency from text. Defaults to INR unless explicitly mentioned."""
    if not text:
        return "INR"  # Default currency
    
    lowered = text.lower()
    
    # Check for currency symbols first (explicit indicators)
    if "$" in text or "dollar" in lowered or "dollars" in lowered:
        return "USD"
    if "€" in text or "euro" in lowered or "euros" in lowered:
        return "EUR"
    if "£" in text or "pound" in lowered or "pounds" in lowered:
        return "GBP"
    if "¥" in text or "yen" in lowered:
        return "JPY"
    if "₹" in text or "rupee" in lowered or "rupees" in lowered or "inr" in lowered:
        return "INR"
    
    # Check for explicit currency mentions (must be near the amount)
    import re
    # Look for patterns like "500 dollar", "100 euros", "50 pounds"
    currency_patterns = [
        (r'\d+[\s,.]*\d*\s*(?:usd|dollar|dollars)', "USD"),
        (r'\d+[\s,.]*\d*\s*(?:eur|euro|euros)', "EUR"),
        (r'\d+[\s,.]*\d*\s*(?:gbp|pound|pounds)', "GBP"),
        (r'\d+[\s,.]*\d*\s*(?:jpy|yen)', "JPY"),
        (r'\d+[\s,.]*\d*\s*(?:inr|rupee|rupees)', "INR"),
    ]
    
    for pattern, currency in currency_patterns:
        if re.search(pattern, lowered, re.IGNORECASE):
            return currency
    
    # Default to INR if no currency explicitly mentioned
    return "INR"

