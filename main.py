"""
MCP Server - TrackExpensio Financial Tools

This file contains the MCP (Model Context Protocol) server implementation.
All financial tools (expenses, budgets, investments, etc.) are defined here
and registered with the FastMCP server.

The MCP server can be used standalone or accessed via the FastAPI client in api.py
"""

import asyncio
import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd
import yfinance as yf
from fastmcp import FastMCP
from fastmcp.tools.tool import FunctionTool

from auth import UserAuth
from database import Database
from ai_helpers import generate_insights, suggest_category
from services.processing import (
    decode_base64_file,
    run_receipt_rag_pipeline,
)
from services.storage import MongoManager
from audit_logger import audit_logger, AuditAction, AuditSeverity, AuditCategory, DETAILED_AUDIT_TOOLS, parse_audit_dates
from tax_rules import tax_engine, TaxJurisdiction
from fx_timing import fx_manager
from security import security_manager

# Initialize MCP server
mcp = FastMCP("TrackExpensio")

# Initialize components
db = Database()
auth = UserAuth()
mongo_manager = MongoManager()


async def _log_audit(
    user_id: Optional[str],
    action: AuditAction,
    *,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    source: str = "mcp",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
    severity: Optional[AuditSeverity] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """Best-effort audit logging for MCP tools."""
    try:
        await audit_logger.log(
            user_id=user_id,
            action=action,
            details=details,
            success=success,
            error_message=error_message,
            source=source,
            resource_type=resource_type,
            resource_id=resource_id,
            session_id=session_id,
            tool_name=tool_name,
            changes=changes,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            duration_ms=duration_ms,
        )
    except Exception:
        pass


# FastMCP's FunctionTool doesn't expose __call__ by default; patch it so we can use the
# decorated functions like regular coroutines throughout the codebase (e.g., quick_add_expense()).
if not hasattr(FunctionTool, "__call__") or not callable(FunctionTool.__dict__.get("__call__", None)):
    def _functiontool_call(self, *args, **kwargs):  # type: ignore
        return self.fn(*args, **kwargs)
    FunctionTool.__call__ = _functiontool_call  # type: ignore[attr-defined]

# Helper function to get user_id from api_key or session_id
async def _get_user_id(api_key: str = None, session_id: str = None) -> tuple[Optional[str], Optional[str]]:
    """Get user_id from api_key or session_id. Returns (user_id, error_message)"""
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        if session_id and not api_key:
            return None, "Google login required. Please sync your session before continuing."
        return None, "Authentication required. Please provide api_key or linked session."
    return user_id, None


async def _init_mongo_indexes():
    await mongo_manager.ensure_indexes()


def _prime_mongo():
    """Initialize MongoDB indexes, gracefully handle connection errors"""
    try:
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the task
            loop.create_task(_init_mongo_indexes())
        except RuntimeError:
            # No running loop, create a new one
            asyncio.run(_init_mongo_indexes())
    except Exception as e:
        # MongoDB not available - log but don't crash
        import sys
        print(f"⚠️  Warning: MongoDB initialization skipped: {str(e)}", file=sys.stderr)
        print("   MongoDB features will be unavailable until connection is established.", file=sys.stderr)


_prime_mongo()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calc_return(series, periods_from_end):
    try:
        if series is None or series.empty or len(series) <= periods_from_end:
            return None
        latest = series.iloc[-1]
        past = series.iloc[-periods_from_end - 1]
        if past in (None, 0):
            return None
        return round(((latest - past) / past) * 100, 2)
    except Exception:
        return None


_YAHOO_SYMBOL_ALIASES = {
    "NETFLIX": "NFLX",
    "APPLE": "AAPL",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "META": "META",
    "FACEBOOK": "META",
    "NVIDIA": "NVDA",
    "INFOSYS": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "WIPRO": "WIPRO.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
}


def _ticker_has_market_data(symbol: str) -> bool:
    """Return True if Yahoo returns price metadata for this symbol."""
    try:
        info = yf.Ticker(symbol).info or {}
        return bool(
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
            or info.get("shortName")
        )
    except Exception:
        return False


def _search_yahoo_symbol(query: str) -> str:
    """Resolve any company name or ticker to a Yahoo Finance symbol (worldwide)."""
    import re

    raw = (query or "").strip()
    if not raw:
        return ""

    from stock_analyzer import COMPANY_TICKER_MAP

    query_clean = raw.lower()
    if query_clean in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[query_clean]
    for name, ticker in COMPANY_TICKER_MAP.items():
        if name in query_clean or query_clean in name:
            return ticker

    upper = raw.upper()
    if upper in _YAHOO_SYMBOL_ALIASES:
        return _YAHOO_SYMBOL_ALIASES[upper]

    # Exchange-qualified tickers: INFY.NS, VOD.L, 7203.T
    if "." in upper and re.match(r"^[A-Z0-9]{1,12}\.[A-Z]{1,4}$", upper):
        return upper

    # Plain tickers: AAPL, TSLA, PLTR
    if re.match(r"^[A-Z]{1,6}(-[A-Z])?$", upper):
        if _ticker_has_market_data(upper):
            return upper

    # Yahoo Finance search — works for any listed company name
    try:
        from yfinance import Search

        quotes = Search(raw, max_results=10).quotes or []
        for item in quotes:
            if not item.get("isYahooFinance"):
                continue
            if item.get("quoteType") not in ("EQUITY", "ETF", "MUTUALFUND"):
                continue
            sym = (item.get("symbol") or "").strip().upper()
            if sym:
                return sym
    except Exception:
        pass

    return upper


def _resolve_yahoo_symbol(symbol: str) -> str:
    """Backward-compatible alias for symbol resolution."""
    return _search_yahoo_symbol(symbol)


def _build_yahoo_snapshot(symbol: str):
    """Build comprehensive stock snapshot with improved error handling and data fetching."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get info with timeout handling
        try:
            info = ticker.info or {}
        except Exception:
            info = {}
        
        # Try fast_info, fallback gracefully
        try:
            fast = getattr(ticker, "fast_info", None)
            if fast is None:
                fast = {}
        except Exception:
            fast = {}
        
        # Fetch historical data with better error handling
        history_3y = None
        history_1y = None
        try:
            history_3y = ticker.history(period="3y", interval="1d", auto_adjust=True, timeout=10)
        except Exception:
            try:
                history_3y = ticker.history(period="2y", interval="1d", auto_adjust=True, timeout=10)
            except Exception:
                pass
        
        try:
            history_1y = ticker.history(period="1y", interval="1d", auto_adjust=True, timeout=10)
        except Exception:
            try:
                history_1y = ticker.history(period="6mo", interval="1d", auto_adjust=True, timeout=10)
            except Exception:
                pass
        
        # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
        if history_3y is not None and not history_3y.empty:
            if isinstance(history_3y.columns, pd.MultiIndex):
                close_col = ('Close', symbol) if ('Close', symbol) in history_3y.columns else history_3y.columns[0]
                close_series_3y = history_3y[close_col] if isinstance(close_col, tuple) else history_3y['Close']
            else:
                close_series_3y = history_3y["Close"]
        else:
            close_series_3y = None
        
        if history_1y is not None and not history_1y.empty:
            if isinstance(history_1y.columns, pd.MultiIndex):
                close_col = ('Close', symbol) if ('Close', symbol) in history_1y.columns else history_1y.columns[0]
                close_series_1y = history_1y[close_col] if isinstance(close_col, tuple) else history_1y['Close']
            else:
                close_series_1y = history_1y["Close"]
        else:
            close_series_1y = None

        # Calculate 3-year return with better error handling
        three_year_return = None
        if close_series_3y is not None and len(close_series_3y) > 1:
            try:
                latest_price = float(close_series_3y.iloc[-1])
                price_3y_ago = float(close_series_3y.iloc[0])
                if price_3y_ago and price_3y_ago > 0:
                    three_year_return = round(((latest_price - price_3y_ago) / price_3y_ago) * 100, 2)
            except (ValueError, IndexError, TypeError):
                pass

        # Calculate returns with better error handling
        returns = {
            "1d": _calc_return(close_series_1y, 1),
            "1w": _calc_return(close_series_1y, 5),
            "1m": _calc_return(close_series_1y, 21),
            "3m": _calc_return(close_series_1y, 63),
            "6m": _calc_return(close_series_1y, 126),
            "1y": _calc_return(close_series_1y, len(close_series_1y) - 1)
            if close_series_1y is not None and hasattr(close_series_1y, "__len__") and len(close_series_1y) > 1
            else None,
            "3y": three_year_return,
        }

        # Get price with multiple fallbacks
        price = None
        price_sources = [
            fast.get("last_price"),
            info.get("regularMarketPrice"),
            info.get("currentPrice"),
            info.get("previousClose"),
            info.get("regularMarketPreviousClose"),
        ]
        for source in price_sources:
            price = _safe_float(source)
            if price and price > 0:
                break
        
        # If still no price, try to get from history
        if not price and close_series_1y is not None and len(close_series_1y) > 0:
            try:
                price = float(close_series_1y.iloc[-1])
            except (ValueError, IndexError, TypeError):
                pass

        # Get currency with fallbacks
        currency = (
            fast.get("currency") or 
            info.get("currency") or 
            info.get("financialCurrency") or
            ("INR" if ".NS" in symbol.upper() or ".BO" in symbol.upper() else "USD")
        )
        currency_symbol = (
            "₹" if currency == "INR" else
            "£" if currency == "GBP" else
            "€" if currency == "EUR" else
            "¥" if currency == "JPY" else "$"
        )

        # Build payload with all data
        payload = {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName") or info.get("symbol") or symbol.upper(),
            "currency": currency,
            "currency_symbol": currency_symbol,
            "price": price,
            "previous_close": _safe_float(
                fast.get("previous_close") or 
                info.get("previousClose") or 
                info.get("regularMarketPreviousClose")
            ),
            "open": _safe_float(
                fast.get("open") or 
                info.get("open") or 
                info.get("regularMarketOpen")
            ),
            "day_range": {
                "low": _safe_float(
                    fast.get("day_low") or 
                    info.get("dayLow") or 
                    info.get("regularMarketDayLow")
                ),
                "high": _safe_float(
                    fast.get("day_high") or 
                    info.get("dayHigh") or 
                    info.get("regularMarketDayHigh")
                ),
            },
            "fifty_two_week_range": {
                "low": _safe_float(
                    fast.get("year_low") or 
                    info.get("fiftyTwoWeekLow") or 
                    info.get("52WeekLow")
                ),
                "high": _safe_float(
                    fast.get("year_high") or 
                    info.get("fiftyTwoWeekHigh") or 
                    info.get("52WeekHigh")
                ),
            },
            "market_cap": _safe_float(
                fast.get("market_cap") or 
                info.get("marketCap") or 
                info.get("enterpriseValue")
            ),
            "volume": _safe_float(
                fast.get("last_volume") or 
                info.get("volume") or 
                info.get("regularMarketVolume")
            ),
            "avg_volume": _safe_float(
                fast.get("three_month_avg_volume") or 
                info.get("averageVolume") or 
                info.get("averageVolume10days")
            ),
            "pe_ratio": _safe_float(
                info.get("trailingPE") or 
                info.get("forwardPE") or 
                info.get("priceToBook")
            ),
            "dividend_yield": _safe_float(
                (info.get("dividendYield") * 100) if info.get("dividendYield") else 
                info.get("yield") or 
                None
            ),
            "recommendation": (
                info.get("recommendationKey") or 
                info.get("recommendationMean") or 
                None
            ),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "exchange": info.get("exchange") or info.get("fullExchangeName"),
            "returns": returns,
        }

        listing_epoch = info.get("firstTradeDateEpochUtc") or info.get("ipoExpectedDate")
        if listing_epoch and isinstance(listing_epoch, (int, float)):
            try:
                payload["listing_date"] = datetime.fromtimestamp(
                    listing_epoch, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass
        if not payload.get("listing_date"):
            try:
                hist = ticker.history(period="max", interval="1mo")
                if hist is not None and not hist.empty:
                    payload["listing_date"] = hist.index[0].strftime("%Y-%m-%d")
            except Exception:
                pass

        # Validate that we have at least a price
        if not payload.get("price"):
            raise ValueError(f"No price data available for symbol '{symbol}'. The symbol may be invalid or delisted.")

        return payload
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error fetching data for '{symbol}': {str(e)}")


async def _fetch_yahoo_finance_snapshot(symbol: str):
    """Fetch Yahoo Finance data with improved error handling and symbol validation."""
    original = (symbol or "").strip()
    symbol = _search_yahoo_symbol(original)
    if not symbol:
        raise ValueError("Symbol is required.")

    # Validate symbol format
    if len(symbol) < 1 or len(symbol) > 20:
        raise ValueError(f"Invalid symbol format: '{symbol}'. Symbol must be 1-20 characters.")

    try:
        # Try fetching with timeout
        payload = await asyncio.wait_for(
            asyncio.to_thread(_build_yahoo_snapshot, symbol),
            timeout=30.0  # 30 second timeout
        )
        
        if not payload:
            raise ValueError(f"No data returned for symbol '{symbol}'.")
        
        if not payload.get("price") or payload.get("price") <= 0:
            raise ValueError(
                f"Invalid price data for '{symbol}'. Please verify the ticker symbol is correct. "
                f"For Indian stocks, use .NS suffix (e.g., INFY.NS, RELIANCE.NS). "
                f"For US stocks, use plain symbol (e.g., AAPL, TSLA)."
            )
        
        payload["as_of"] = datetime.now(timezone.utc).isoformat() + "Z"
        return payload
        
    except asyncio.TimeoutError:
        raise ValueError(
            f"Request timeout for symbol '{symbol}'. The Yahoo Finance service may be slow. "
            f"Please try again in a moment."
        )
    except ValueError as ve:
        # Re-raise ValueError as-is (already formatted)
        raise ve
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["no data", "symbol", "invalid", "not found", "delisted"]):
            raise ValueError(
                f"Invalid or unavailable ticker symbol '{symbol}'. Please check the symbol and try again.\n"
                f"Examples:\n"
                f"  - US stocks: AAPL, GOOGL, MSFT, TSLA\n"
                f"  - Indian stocks: INFY.NS, RELIANCE.NS, TCS.NS, ITC.NS\n"
                f"  - UK stocks: VOD.L, BP.L\n"
                f"  - ETFs: SPY, QQQ"
            )
        raise ValueError(f"Yahoo Finance error for '{symbol}': {str(e)}")

# User Authentication Tools
@mcp.tool()
async def register_user(username, password):
    """Register a new user account"""
    result = await auth.create_user(username, password)
    if result.get("status") == "success":
        await _log_audit(
            result.get("user_id"),
            AuditAction.API_KEY_GENERATED,
            details={"username": username},
            source="mcp",
        )
    return result

@mcp.tool()
async def login_user(username, password):
    """Login user and get API key"""
    try:
        result = await auth.login_user(username, password)
        if result.get("status") == "success":
            await _log_audit(
                result.get("user_id"),
                AuditAction.LOGIN,
                details={"username": username},
                source="mcp",
            )
        else:
            await _log_audit(
                None,
                AuditAction.FAILED_LOGIN,
                details={"username": username},
                success=False,
                error_message=result.get("message"),
                source="mcp",
            )
        return result
    except Exception as e:
        await _log_audit(
            None,
            AuditAction.FAILED_LOGIN,
            details={"username": username},
            success=False,
            error_message=str(e),
            source="mcp",
        )
        return {"status": "error", "message": f"Login error: {str(e)}"}

@mcp.tool()
async def get_user_info(api_key: str = None, session_id: str = None):
    """Get user information and account details.
    
    Returns user ID and basic account information for the authenticated user.
    """
    user_id, error = await _get_user_id(api_key, session_id)
    if error:
        return {"status": "error", "message": error}
    return {"status": "success", "user_id": user_id}

# Expense Management Tools
@mcp.tool()
async def add_expense(api_key: str = None, session_id: str = None, date: str = None, amount: float = None, category: str = None, note: str = "", merchant: str = "", metadata: Optional[Dict[str, Any]] = None):
    """Add a new expense transaction with automatic AI category suggestion. 
    
    Records an expense with date, amount, category, merchant, and optional note.
    The AI will suggest the most appropriate category based on merchant and note.
    Supports multi-currency expenses via metadata.currency field.
    Default currency is INR unless specified in metadata.currency.
    
    Args:
        date: Expense date in YYYY-MM-DD format (defaults to today if not provided)
        amount: Expense amount (required, must be positive number)
        category: Expense category (Food, Travel, Bills, Shopping, Entertainment, Utilities, Rent, Health, Education, Other)
        merchant: Store or merchant name (optional)
        note: Additional notes about the expense (optional)
        metadata: Optional dict with 'currency' key for non-default currency expenses (default is INR)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Validate and parse date
        if not date or date.strip() == "":
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            # Validate date format - if it's not YYYY-MM-DD, try to parse it using dateparser
            try:
                # Check if it's already in YYYY-MM-DD format
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                # Try to parse using dateparser
                try:
                    from dateparser import parse as dateparse
                    parsed_date = dateparse(date, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": datetime.now(timezone.utc)})
                    if parsed_date:
                        date = parsed_date.strftime("%Y-%m-%d")
                    else:
                        # If parsing fails, default to today
                        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    # If all parsing fails, default to today
                    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # AI category suggestion
        suggested_category, confidence = await suggest_category(merchant, note, float(amount))
        
        # Use suggested category if category not provided
        if not category:
            category = suggested_category
        
        # Ensure amount is valid
        if amount is None or amount <= 0:
            return {"status": "error", "message": "Invalid amount. Amount must be a positive number."}
        
        # Ensure date is valid YYYY-MM-DD format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"status": "error", "message": f"Invalid date format: {date}. Expected YYYY-MM-DD."}
        
        # Add expense to database
        expense_id = await db.add_expense(user_id, date, float(amount), category, note, merchant, metadata=metadata)
        
        if not expense_id:
            await _log_audit(
                user_id,
                AuditAction.EXPENSE_ADDED,
                details={"amount": float(amount), "category": category},
                success=False,
                error_message="Failed to save expense to database",
                resource_type="expense",
            )
            return {"status": "error", "message": "Failed to save expense to database."}
        
        await _log_audit(
            user_id,
            AuditAction.EXPENSE_ADDED,
            details={
                "amount": float(amount),
                "category": category or suggested_category,
                "date": date,
                "merchant": merchant,
                "currency": metadata.get("currency", "INR") if metadata else "INR",
            },
            resource_type="expense",
            resource_id=expense_id,
        )
        
        return {
            "status": "success",
            "id": expense_id,
            "amount": float(amount),
            "category": category or suggested_category,
            "date": date,
            "merchant": merchant,
            "metadata": metadata or {},
            "message": "Expense added successfully",
            "ai_suggestion": {
                "category": suggested_category,
                "confidence": confidence
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding expense: {str(e)}"}

@mcp.tool()
async def list_expenses(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """List all expenses within a specified date range with comprehensive details.
    
    Returns a chronological list (newest first) of expenses with complete information including
    date, amount, category, merchant, notes, and currency.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (required). If not provided, defaults to 6 months ago.
        end_date: End date in YYYY-MM-DD format (required). If not provided, defaults to today.
    
    Returns:
        List of expense dictionaries, each containing:
        - id: Expense ID
        - date: Expense date (YYYY-MM-DD)
        - amount: Expense amount (float)
        - category: Expense category
        - merchant: Merchant/store name
        - note: Additional notes
        - metadata: Additional metadata including currency
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Set default date range if not provided (last 30 days)
        if not start_date and not end_date:
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            start_date = thirty_days_ago.strftime("%Y-%m-%d")
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif not end_date:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif not start_date:
            start_date = "1970-01-01"
        
        # Validate date formats
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return {"status": "error", "message": f"Invalid date format. Please use YYYY-MM-DD format. Received: start_date={start_date}, end_date={end_date}"}
        
        # Validate date range
        if start_date > end_date:
            return {"status": "error", "message": f"Invalid date range: start_date ({start_date}) must be before or equal to end_date ({end_date})"}
        
        expenses = await db.get_expenses(user_id, start_date, end_date)
        
        # Return expenses as a list (not wrapped in status)
        if isinstance(expenses, list):
            return expenses
        return expenses if expenses else []
    except Exception as e:
        return {"status": "error", "message": f"Error listing expenses: {str(e)}"}

@mcp.tool()
async def get_expenses_overview(
    api_key: str = None,
    session_id: str = None,
    period: str = "30d",
    start_date: str = None,
    end_date: str = None,
):
    """Get expenses with total, count, and a breakdown of where the user spent (by category and merchant).

    period: 30d (default), 6m, or all. Custom range via start_date/end_date (YYYY-MM-DD).
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}

        today = datetime.now(timezone.utc)
        period_key = (period or "30d").lower().strip()
        range_start = start_date
        range_end = end_date or today.strftime("%Y-%m-%d")

        if not range_start:
            if period_key in ("all", "all_time", "lifetime"):
                range_start = None
                range_end = None
            elif period_key in ("6m", "6mo", "6months", "180d"):
                range_start = (today - timedelta(days=180)).strftime("%Y-%m-%d")
            else:
                range_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")

        if range_start and range_end:
            try:
                datetime.strptime(range_start, "%Y-%m-%d")
                datetime.strptime(range_end, "%Y-%m-%d")
            except ValueError:
                return {"status": "error", "message": "Dates must be YYYY-MM-DD"}
            if range_start > range_end:
                return {"status": "error", "message": "start_date must be on or before end_date"}

        overview = await db.get_expense_overview(user_id, range_start, range_end)
        overview["status"] = "success"
        overview["period"] = period_key if not start_date else "custom"
        return overview
    except Exception as e:
        return {"status": "error", "message": f"Error loading expense overview: {str(e)}"}


@mcp.tool()
async def summarize(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """Get a summary of expenses grouped by category for a date range.
    
    Provides total spending per category, overall total, and count of transactions.
    Useful for understanding spending patterns and identifying top expense categories.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (required)
        end_date: End date in YYYY-MM-DD format (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        summary = await db.get_expense_summary(user_id, start_date, end_date)
        return summary
    except Exception as e:
        return {"status": "error", "message": f"Error summarizing expenses: {str(e)}"}

# AI-Powered Tools
@mcp.tool()
async def ai_insights(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """Get AI-powered spending insights"""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        expenses = await db.get_expenses(user_id, start_date, end_date)
        expense_data = expenses
        
        insights = await generate_insights(expense_data)
        return {"insights": insights, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"Error generating insights: {str(e)}"}

@mcp.tool()
async def quick_add_expense(api_key: str = None, session_id: str = None, amount: float = None, description: str = None, merchant: str = ""):
    """Quickly add an expense with minimal input using AI-powered category detection.
    
    This is a simplified expense addition tool that requires only amount and description.
    The system automatically:
    - Uses today's date
    - Suggests the best category using AI
    - Extracts merchant name from description if not provided
    
    Args:
        amount: Expense amount (required, must be positive number)
        description: Brief description of the expense (required, e.g., "Lunch at restaurant", "Uber ride")
        merchant: Store or merchant name (optional, will be extracted from description if not provided)
    
    Returns:
        Dictionary with status, expense_id, category suggestion, and success message.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Validate amount
        if amount is None:
            return {"status": "error", "message": "Amount is required and must be a positive number."}
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return {"status": "error", "message": "Amount must be a positive number."}
        except (ValueError, TypeError):
            return {"status": "error", "message": f"Invalid amount: '{amount}'. Amount must be a valid number."}
        
        # Validate description
        if not description or not description.strip():
            return {"status": "error", "message": "Description is required. Please provide a brief description of the expense."}
        
        # Use today's date
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Extract merchant from description if not provided
        if not merchant or not merchant.strip():
            # Try to extract merchant from description (e.g., "Lunch at Restaurant Name" -> "Restaurant Name")
            description_lower = description.lower()
            if "at " in description_lower:
                parts = description.split(" at ", 1)
                if len(parts) > 1:
                    merchant = parts[1].strip()
            elif "from " in description_lower:
                parts = description.split(" from ", 1)
                if len(parts) > 1:
                    merchant = parts[1].strip()
            else:
                merchant = ""
        
        # AI category suggestion
        try:
            suggested_category, confidence = await suggest_category(merchant, description, amount_float)
        except Exception:
            suggested_category = "Other"
            confidence = 0.5
        
        # Add expense
        expense_id = await db.add_expense(
            user_id, 
            today, 
            amount_float, 
            suggested_category, 
            description.strip(), 
            merchant.strip() if merchant else ""
        )
        
        if not expense_id:
            return {"status": "error", "message": "Failed to save expense to database."}
        
        return {
            "status": "success",
            "id": expense_id,
            "message": f"✅ Expense added: ₹{amount_float:,.2f} for {description}",
            "amount": amount_float,
            "category": suggested_category,
            "date": today,
            "merchant": merchant.strip() if merchant else "",
            "ai_suggestion": {
                "category": suggested_category,
                "confidence": round(confidence, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding expense: {str(e)}"}

@mcp.tool()
async def document_expense_from_rag(api_key: str = None, session_id: str = None, document_base64: str = None, filename: str = "receipt.pdf"):
    """Parse bills, receipts, or invoices (PDF/image) using advanced RAG pipeline and automatically extract expense details.
    
    Uses LangGraph + FAISS + Groq LLM to intelligently extract structured information from financial documents.
    Automatically categorizes expenses, detects currency, extracts dates, and identifies merchants.
    
    Supported formats:
    - PDF documents (receipts, invoices, bills)
    - Image files (PNG, JPG, JPEG) containing receipts
    
    Extracted information:
    - Merchant/store name
    - Total amount
    - Currency (auto-detected, defaults to INR)
    - Expense category (Food, Travel, Bills, Shopping, etc.)
    - Transaction date
    - Additional notes/items
    
    Args:
        document_base64: Base64-encoded document content (required)
        filename: Original filename (default: "receipt.pdf")
    
    Returns:
        Dictionary with status, expense_id, and extracted information.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}

        # Validate document_base64
        if not document_base64 or not document_base64.strip():
            return {"status": "error", "message": "Document content is required. Please provide a valid base64-encoded document."}

        # Decode and validate file
        try:
            file_bytes = decode_base64_file(document_base64)
            if not file_bytes or len(file_bytes) == 0:
                return {"status": "error", "message": "Invalid document: file is empty."}
            if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
                return {"status": "error", "message": "Document too large. Maximum size is 10MB."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to decode document: {str(e)}"}

        # Create temporary file
        try:
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name
        except Exception as e:
            return {"status": "error", "message": f"Failed to create temporary file: {str(e)}"}

        # Process document with RAG pipeline
        try:
            extracted = await asyncio.wait_for(
                run_receipt_rag_pipeline(temp_path),
                timeout=60.0  # 60 second timeout for RAG processing
            )
        except asyncio.TimeoutError:
            try:
                os.unlink(temp_path)
            except:
                pass
            return {"status": "error", "message": "Document processing timed out. The document may be too complex or the service is slow. Please try again."}
        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
            return {"status": "error", "message": f"Failed to process document: {str(e)}. Please ensure the document is a valid PDF or image file."}
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        # Validate extracted data
        if not extracted or not isinstance(extracted, dict):
            return {"status": "error", "message": "Failed to extract information from document. The document may be unreadable or corrupted."}

        amount = extracted.get("amount")
        if not amount or amount == 0:
            return {
                "status": "error", 
                "message": "Could not extract amount from document. Please ensure the document contains a clear total amount.",
                "extracted": extracted  # Return partial extraction for debugging
            }

        # Process and validate date
        date_str = extracted.get("date")
        if date_str:
            try:
                # Try to parse and validate date
                from dateparser import parse as dateparse
                parsed_date = dateparse(date_str, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": datetime.now(timezone.utc)})
                if parsed_date:
                    date = parsed_date.strftime("%Y-%m-%d")
                else:
                    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                # If date parsing fails, try direct format check
                try:
                    datetime.strptime(date_str, "%Y-%m-%d")
                    date = date_str
                except ValueError:
                    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Get category with AI suggestion if not provided or if category seems wrong
        category = extracted.get("category") or "Bills"
        merchant = extracted.get("merchant") or "Unknown"
        note = extracted.get("notes") or extracted.get("note") or "Parsed via RAG"
        
        # Use AI to suggest better category if needed
        if category == "Bills" or category == "Other":
            try:
                suggested_category, confidence = await suggest_category(merchant, note, float(amount))
                if confidence > 0.6:  # Use AI suggestion if confident
                    category = suggested_category
            except Exception:
                pass  # Continue with extracted category if AI suggestion fails

        # Build metadata
        metadata = {"source": "rag_upload", "filename": filename}
        currency = extracted.get("currency") or "INR"
        if currency and currency != "INR":
            metadata["currency"] = currency

        # Add expense to database
        try:
            expense_id = await db.add_expense(
                user_id,
                date,
                float(amount),
                category,
                note,
                merchant,
                metadata=metadata,
            )
            
            if not expense_id:
                return {"status": "error", "message": "Failed to save expense to database."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to save expense: {str(e)}"}

        # Save document metadata
        try:
            await mongo_manager.save_document_expense(
                user_id,
                {
                    "filename": filename,
                    "extracted": extracted,
                    "expense_id": expense_id,
                },
            )
        except Exception:
            pass  # Non-critical, continue even if document metadata save fails

        return {
            "status": "success",
            "message": "Document parsed and expense created successfully",
            "expense_id": expense_id,
            "extracted": {
                "merchant": merchant,
                "amount": float(amount),
                "currency": currency,
                "category": category,
                "date": date,
                "notes": note,
            },
        }
    except Exception as exc:
        return {"status": "error", "message": f"Document parsing failed: {str(exc)}"}


@mcp.tool()
async def delete_expense(api_key: str = None, session_id: str = None, expense_id: str = None):
    """Delete an expense by its ID.
    
    Removes an expense transaction from the database permanently. Use this when the user
    wants to remove or delete an expense. The expense_id must be obtained from list_expenses
    first if the user doesn't provide it directly.
    
    Args:
        expense_id: The unique ID of the expense to delete (required)
    
    Returns:
        Dictionary with status and message indicating success or failure.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not expense_id:
            return {"status": "error", "message": "expense_id is required. Use list_expenses to find the expense ID first."}
        if not hasattr(db, "delete_expense"):
            return {"status": "error", "message": "delete_expense not supported by database layer"}
        deleted = await db.delete_expense(user_id, expense_id)
        if not deleted:
            await _log_audit(
                user_id,
                AuditAction.EXPENSE_DELETED,
                details={"expense_id": expense_id},
                success=False,
                error_message="Expense not found or already deleted",
                resource_type="expense",
                resource_id=expense_id,
            )
            return {"status": "error", "message": "Expense not found or already deleted"}
        await _log_audit(
            user_id,
            AuditAction.EXPENSE_DELETED,
            details={"expense_id": expense_id},
            resource_type="expense",
            resource_id=expense_id,
        )
        return {"status": "success", "message": "Expense deleted successfully", "expense_id": expense_id}
    except Exception as exc:
        return {"status": "error", "message": f"Error deleting expense: {str(exc)}"}


@mcp.tool()
async def update_expense(
    api_key: str = None,
    session_id: str = None,
    expense_id: str = None,
    date: str = None,
    amount: float = None,
    category: str = None,
    note: str = None,
    merchant: str = None,
):
    """Update mutable fields of an expense."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not hasattr(db, "update_expense"):
            return {"status": "error", "message": "update_expense not supported by database layer"}
        updated = await db.update_expense(
            user_id,
            expense_id,
            {
                "date": date,
                "amount": float(amount) if amount is not None else None,
                "category": category,
                "note": note,
                "merchant": merchant,
            },
        )
        if not updated:
            await _log_audit(
                user_id,
                AuditAction.EXPENSE_UPDATED,
                details={"expense_id": expense_id, "changes": {"date": date, "amount": amount, "category": category}},
                success=False,
                error_message="Nothing was updated",
                resource_type="expense",
                resource_id=expense_id,
            )
            return {"status": "error", "message": "Nothing was updated"}
        change_set = {k: v for k, v in {"date": date, "amount": amount, "category": category, "note": note, "merchant": merchant}.items() if v is not None}
        await _log_audit(
            user_id,
            AuditAction.EXPENSE_UPDATED,
            details={"expense_id": expense_id},
            changes={"after": change_set},
            resource_type="expense",
            resource_id=expense_id,
        )
        return {"status": "success", "message": "Expense updated"}
    except Exception as exc:
        return {"status": "error", "message": f"Error updating expense: {str(exc)}"}

# Budget Management Tools
@mcp.tool()
async def set_budget(api_key: str = None, session_id: str = None, category: str = None, amount: float = None):
    """Set or update a monthly budget for a specific expense category.
    
    Creates a budget that will be tracked monthly. The system will alert you
    when you approach or exceed the budget limit.
    
    Args:
        category: Expense category name (required). Valid categories: Food, Travel, Bills, Shopping, 
                  Entertainment, Utilities, Rent, Health, Education, Other
        amount: Monthly budget amount (required, must be positive number)
    
    Returns:
        Dictionary with status, budget_id, category, amount, and success message.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Validate category
        valid_categories = ["Food", "Travel", "Bills", "Shopping", "Entertainment", "Utilities", "Rent", "Health", "Education", "Other"]
        if not category:
            return {"status": "error", "message": "Category is required. Valid categories: " + ", ".join(valid_categories)}
        if category not in valid_categories:
            return {"status": "error", "message": f"Invalid category '{category}'. Valid categories: {', '.join(valid_categories)}"}
        
        # Validate amount
        if amount is None:
            return {"status": "error", "message": "Amount is required and must be a positive number."}
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return {"status": "error", "message": "Amount must be a positive number."}
        except (ValueError, TypeError):
            return {"status": "error", "message": f"Invalid amount: '{amount}'. Amount must be a valid number."}
        
        budget_id = await db.set_budget(user_id, category, amount_float)
        
        if not budget_id:
            await _log_audit(
                user_id,
                AuditAction.BUDGET_SET,
                details={"category": category, "amount": amount_float},
                success=False,
                error_message="Failed to save budget to database",
                resource_type="budget",
            )
            return {"status": "error", "message": "Failed to save budget to database."}
        
        await _log_audit(
            user_id,
            AuditAction.BUDGET_SET,
            details={"category": category, "amount": amount_float},
            resource_type="budget",
            resource_id=str(budget_id),
        )
        return {
            "status": "success",
            "id": budget_id,
            "message": f"Budget set successfully: {category} - ₹{amount_float:,.2f} per month",
            "category": category,
            "amount": amount_float
        }
    except Exception as e:
        return {"status": "error", "message": f"Error setting budget: {str(e)}"}

@mcp.tool()
async def check_budget_status(api_key: str = None, session_id: str = None):
    """Check current month's budget status for all categories.
    
    Returns remaining budget, spending percentage, alerts for overspending,
    and AI-powered tips for better financial management.
    
    Shows which categories are within budget, approaching limit, or exceeded.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Get current month date range
        today = datetime.now()
        start_of_month = today.replace(day=1).strftime('%Y-%m-%d')
        end_of_month = today.strftime('%Y-%m-%d')
        
        # Get all budgets for user
        budgets = await db.get_budgets(user_id)
        
        if not budgets:
            return {
                "status": "success",
                "message": "No budgets set. Use set_budget() to create one.",
                "budgets": []
            }
        
        # Get spending by category for current month
        spending = await db.get_spending_by_category(user_id, start_of_month, end_of_month)
        spending_dict = {row['category']: row['total'] for row in spending} if spending else {}
        
        budget_status = []
        alerts = []
        total_remaining = 0
        total_budget = 0
        total_spent = 0
        
        for budget in budgets:
            category = budget.get('category', '')
            budget_amount = float(budget.get('amount', 0))
            spent = spending_dict.get(category, 0)
            remaining = budget_amount - spent
            percentage_used = (spent / budget_amount * 100) if budget_amount > 0 else 0
            
            total_budget += budget_amount
            total_spent += spent
            total_remaining += remaining
            
            status_item = {
                "category": category,
                "budget": budget_amount,
                "spent": spent,
                "remaining": remaining,
                "percentage_used": round(percentage_used, 2)
            }
            
            # Generate alerts
            if remaining < 0:
                alerts.append({
                    "type": "overspending",
                    "category": category,
                    "message": f"⚠️ OVERSPENDING: {category} exceeded by ${abs(remaining):.2f}"
                })
            elif percentage_used >= 90:
                alerts.append({
                    "type": "warning",
                    "category": category,
                    "message": f"⚠️ WARNING: {category} is {percentage_used:.1f}% used (${remaining:.2f} remaining)"
                })
            elif percentage_used >= 75:
                alerts.append({
                    "type": "caution",
                    "category": category,
                    "message": f"💡 CAUTION: {category} is {percentage_used:.1f}% used (${remaining:.2f} remaining)"
                })
            
            budget_status.append(status_item)
        
        # Generate AI tips
        ai_tips = []
        if total_spent > total_budget:
            ai_tips.append("🚨 You're overspending this month! Consider reviewing non-essential expenses.")
        elif total_spent / total_budget > 0.8 if total_budget > 0 else False:
            ai_tips.append("💡 You've used over 80% of your budget. Try to limit discretionary spending.")
        
        # Category-specific tips
        for status in budget_status:
            if status['remaining'] < 0:
                ai_tips.append(f"💡 For {status['category']}: Consider finding cheaper alternatives or reducing frequency.")
            elif status['percentage_used'] > 90:
                ai_tips.append(f"💡 For {status['category']}: You're close to your limit. Track every expense carefully.")
        
        if not ai_tips:
            ai_tips.append("✅ Great job! You're managing your budget well this month.")
        
        return {
            "status": "success",
            "month": today.strftime('%B %Y'),
            "budget_status": budget_status,
            "summary": {
                "total_budget": total_budget,
                "total_spent": total_spent,
                "total_remaining": total_remaining,
                "overall_percentage_used": round((total_spent / total_budget * 100) if total_budget > 0 else 0, 2)
            },
            "alerts": alerts,
            "ai_tips": ai_tips
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking budget status: {str(e)}"}


@mcp.tool()
async def yahoo_finance(symbol: str, api_key: str = None, session_id: str = None):
    """Get comprehensive real-time market data for stocks, ETFs, or mutual funds using Yahoo Finance.
    
    Fetches detailed market information including:
    - Current price, previous close, open price
    - Day range and 52-week range
    - Market capitalization
    - Trading volume and average volume
    - P/E ratio and dividend yield
    - Returns: 1D, 1W, 1M, 3M, 6M, 1Y, 3Y
    - Sector, industry, and analyst recommendations
    
    Supports multiple stock exchanges worldwide:
    - US stocks: Plain symbol (e.g., "AAPL", "GOOGL", "MSFT", "TSLA")
    - Indian stocks: Add .NS suffix for NSE (e.g., "INFY.NS", "RELIANCE.NS", "TCS.NS", "ITC.NS")
    - Indian stocks: Add .BO suffix for BSE (e.g., "RELIANCE.BO")
    - UK stocks: Add .L suffix (e.g., "VOD.L", "BP.L")
    - European stocks: Add country suffix (e.g., ".PA" for Paris, ".DE" for Germany)
    - ETFs: Plain symbol (e.g., "SPY", "QQQ", "VTI")
    
    Args:
        symbol: Stock ticker symbol (required). Must include exchange suffix for non-US stocks.
    
    Returns:
        Dictionary with status, data (containing all market information), or error message.
    
    Examples:
        - US: "AAPL", "GOOGL", "MSFT", "TSLA"
        - India: "INFY.NS", "RELIANCE.NS", "TCS.NS", "ITC.NS"
        - UK: "VOD.L", "BP.L"
        - ETFs: "SPY", "QQQ"
    """
    try:
        data = await _fetch_yahoo_finance_snapshot(symbol)
        return {"status": "success", "data": data}
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Yahoo Finance lookup failed: {str(exc)}"}


@mcp.tool()
async def smart_stock_analyze(
    ticker: str,
    api_key: str = None,
    session_id: str = None,
    include_chart: bool = False,
):
    """Run technical stock analysis: RSI, MACD, Bollinger Bands, buy/sell/hold signal, support/resistance, 30-day projection, and news sentiment.

    Use for technical analysis requests. For simple price/returns queries, use yahoo_finance instead.
    Indian NSE tickers use .NS suffix (e.g. RELIANCE.NS, INFY.NS).
    """
    user_id, err = await _get_user_id(api_key, session_id)
    if not user_id:
        return {"status": "error", "message": err or "Authentication required"}

    from stock_analyzer import analyze_stock

    try:
        result = await asyncio.to_thread(analyze_stock, ticker.strip())
    except Exception as exc:
        return {"status": "error", "message": f"Stock analysis failed: {exc}"}

    if result.get("error"):
        return {"status": "error", "message": result["error"], "ticker": result.get("ticker")}

    if not include_chart:
        result = dict(result)
        result.pop("chart_json", None)

    return {"status": "success", **result}


@mcp.tool()
async def get_stock_return_one_year(symbol: str, api_key: str = None, session_id: str = None):
    """Calculate the 1-year return percentage for a given stock symbol.
    
    Fetches historical price data for the past year and calculates the percentage return
    from the start price to the current price.
    
    Args:
        symbol: Stock ticker symbol (required, e.g., "AAPL", "TSLA", "INFY.NS" for Indian stocks)
    
    Returns:
        Dictionary with symbol, start_price, end_price, and one_year_return_percent
    """
    try:
        original = (symbol or "").strip()
        if not original:
            return {"status": "error", "message": "Symbol is required"}
        symbol = _search_yahoo_symbol(original)

        result = await asyncio.to_thread(_get_one_year_return, symbol)
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        return {
            "status": "success",
            "symbol": result["symbol"],
            "start_price": result["start_price"],
            "end_price": result["end_price"],
            "one_year_return_percent": result["one_year_return_percent"]
        }
    except Exception as exc:
        return {"status": "error", "message": f"Failed to calculate 1-year return: {str(exc)}"}


@mcp.tool()
async def get_stock_returns(symbol: str, years: int = None, from_listing: bool = False, api_key: str = None, session_id: str = None):
    """Calculate stock returns for flexible time periods with year-by-year breakdown.
    
    This tool works for ANY stock symbol from ANY exchange worldwide. It can calculate returns for:
    - Specific number of years (e.g., last 7 years)
    - From listing date to today (if from_listing=True)
    - All available historical data (if years=None and from_listing=False)
    
    Returns overall return and year-by-year breakdown showing returns for each year.
    
    Args:
        symbol: Stock ticker symbol (required). Works for any stock worldwide:
            - US stocks: "AAPL", "TSLA", "GOOGL", "MSFT"
            - Indian stocks: "RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS"
            - UK stocks: "VOD.L", "BP.L"
            - Japanese stocks: "7203.T", "6758.T"
            - European stocks: "SAP.DE", "ASML.AS"
            - And any other stock available on Yahoo Finance
        years: Number of years to look back (optional, e.g., 7 for last 7 years)
        from_listing: If True, calculate returns from the stock's listing date to today (optional)
    
    Examples:
        - "Get returns for AAPL for last 7 years" -> symbol="AAPL", years=7
        - "Get returns from when TSLA was listed" -> symbol="TSLA", from_listing=True
        - "Get all returns for RELIANCE.NS" -> symbol="RELIANCE.NS", years=None, from_listing=False
        - "Show me every year returns for ITC.NS from listing" -> symbol="ITC.NS", from_listing=True
    
    Returns:
        Dictionary with symbol, dates, prices, total return, and yearly returns breakdown
    """
    try:
        original = (symbol or "").strip()
        if not original:
            return {"status": "error", "message": "Symbol is required"}
        symbol = _search_yahoo_symbol(original)

        # Calculate flexible returns using async thread
        result = await asyncio.to_thread(_get_stock_returns_flexible, symbol, years, from_listing)
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        return {
            "status": "success",
            "symbol": result["symbol"],
            "start_date": result["start_date"],
            "end_date": result["end_date"],
            "start_price": result["start_price"],
            "end_price": result["end_price"],
            "total_return_percent": result["total_return_percent"],
            "years_analyzed": result["years_analyzed"],
            "yearly_returns": result["yearly_returns"],
            "listing_date": result.get("listing_date")
        }
    except Exception as exc:
        return {"status": "error", "message": f"Failed to calculate stock returns: {str(exc)}"}


def _get_one_year_return(symbol: str):
    """Synchronous helper to calculate 1-year return."""
    today = datetime.now(timezone.utc)
    one_year_ago = today - timedelta(days=365)
    
    try:
        data = yf.download(symbol, start=one_year_ago, end=today, progress=False)
        
        if data.empty:
            return {"error": f"No data found for symbol '{symbol}'. Please verify the ticker symbol is correct."}
        
        start_price = float(data["Close"].iloc[0])
        end_price = float(data["Close"].iloc[-1])
        
        percent_return = ((end_price - start_price) / start_price) * 100
        
        return {
            "symbol": symbol,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "one_year_return_percent": round(percent_return, 2)
        }
    except Exception as e:
        return {"error": f"Error fetching data for {symbol}: {str(e)}"}


def _get_stock_returns_flexible(symbol: str, years: int = None, from_listing: bool = False):
    """Calculate stock returns for flexible time periods.
    
    Args:
        symbol: Stock ticker symbol
        years: Number of years to look back (if None and from_listing=False, uses max available)
        from_listing: If True, calculate from listing date to today
    """
    try:
        symbol = _search_yahoo_symbol(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Get listing date from info
        listing_date = None
        if from_listing:
            # Try to get listing date from various fields
            listing_date = info.get("firstTradeDateEpochUtc")
            if listing_date:
                listing_date = datetime.fromtimestamp(listing_date)
            else:
                # Fallback: try to get earliest available data
                hist = ticker.history(period="max", interval="1d")
                if not hist.empty:
                    listing_date = hist.index[0].to_pydatetime()
        
        # Determine date range
        end_date = datetime.now(timezone.utc)
        if from_listing and listing_date:
            start_date = listing_date
        elif years:
            start_date = end_date - timedelta(days=years * 365)
        else:
            # Get max available data
            start_date = None
        
        # Download historical data
        if start_date:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        else:
            data = ticker.history(period="max", interval="1d", auto_adjust=True)
        
        if data.empty:
            return {"error": f"No data found for symbol '{symbol}'. Please verify the ticker symbol is correct."}
        
        # Calculate overall return
        start_price = float(data["Close"].iloc[0])
        end_price = float(data["Close"].iloc[-1])
        total_return = ((end_price - start_price) / start_price) * 100
        
        # Calculate year-by-year returns
        yearly_returns = []
        
        # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            close_col = ('Close', symbol) if ('Close', symbol) in data.columns else data.columns[0]
            data_close = data[close_col] if isinstance(close_col, tuple) else data['Close']
        else:
            data_close = data['Close']
        
        # Convert to DataFrame with Date index
        data_df = pd.DataFrame({
            'Date': data.index,
            'Close': data_close.values
        })
        data_df['Year'] = pd.to_datetime(data_df['Date']).dt.year
        
        # Group by year and calculate returns
        years_data = {}
        for idx, row in data_df.iterrows():
            year = row['Year']
            if year not in years_data:
                years_data[year] = []
            years_data[year].append(row['Close'])
        
        # Calculate year-over-year returns
        sorted_years = sorted(years_data.keys())
        for i, year in enumerate(sorted_years):
            year_prices = years_data[year]
            if len(year_prices) > 1:
                year_start = year_prices[0]
                year_end = year_prices[-1]
                year_return = ((year_end - year_start) / year_start) * 100
                yearly_returns.append({
                    "year": year,
                    "start_price": round(year_start, 2),
                    "end_price": round(year_end, 2),
                    "return_percent": round(year_return, 2)
                })
        
        # Format dates
        start_date_str = data_df['Date'].iloc[0]
        end_date_str = data_df['Date'].iloc[-1]
        if hasattr(start_date_str, 'strftime'):
            start_date_str = start_date_str.strftime("%Y-%m-%d")
        else:
            start_date_str = str(start_date_str)[:10]
        if hasattr(end_date_str, 'strftime'):
            end_date_str = end_date_str.strftime("%Y-%m-%d")
        else:
            end_date_str = str(end_date_str)[:10]
        
        result = {
            "symbol": symbol,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "total_return_percent": round(total_return, 2),
            "years_analyzed": len(sorted_years),
            "yearly_returns": yearly_returns
        }
        
        if from_listing and listing_date:
            result["listing_date"] = listing_date.strftime("%Y-%m-%d")
        
        return result
    except Exception as e:
        return {"error": f"Error fetching data for {symbol}: {str(e)}"}

# Recurring Expenses Tools
@mcp.tool()
async def add_recurring_expense(api_key: str = None, session_id: str = None, amount: float = None, category: str = None, frequency: str = None, merchant: str = ""):
    """Add a recurring expense (subscriptions, rent, EMI, OTT apps, etc.)
    
    Frequency options: 'daily', 'weekly', 'monthly', 'yearly'
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        valid_frequencies = ['daily', 'weekly', 'monthly', 'yearly']
        if frequency.lower() not in valid_frequencies:
            return {
                "status": "error",
                "message": f"Invalid frequency. Must be one of: {', '.join(valid_frequencies)}"
            }
        
        recurring_id = await db.add_recurring_expense(
            user_id, 
            float(amount), 
            category, 
            frequency.lower(), 
            merchant
        )
        
        return {
            "status": "success",
            "id": recurring_id,
            "message": f"Recurring expense added: ${amount} for {category} ({frequency})",
            "amount": float(amount),
            "category": category,
            "frequency": frequency.lower(),
            "merchant": merchant
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding recurring expense: {str(e)}"}

@mcp.tool()
async def generate_monthly_recurring(api_key: str = None, session_id: str = None):
    """Generate expenses for the current month based on recurring expense templates"""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Get all recurring expenses
        recurring_expenses = await db.get_recurring_expenses(user_id)
        
        if not recurring_expenses:
            return {
                "status": "success",
                "message": "No recurring expenses found. Use add_recurring_expense() to create one.",
                "generated": []
            }
        
        today = datetime.now()
        start_of_month = today.replace(day=1)
        current_date = today
        
        generated_expenses = []
        
        for recurring in recurring_expenses:
            amount = float(recurring.get('amount', 0))
            category = recurring.get('category', '')
            frequency = recurring.get('frequency', 'monthly').lower()
            merchant = recurring.get('merchant', '')
            recurring_id = recurring.get('id')
            
            # Calculate how many times this expense should occur in current month
            occurrences = 0
            expense_date = None
            
            if frequency == 'daily':
                # Add daily expenses from start of month to today
                occurrences = (current_date - start_of_month).days + 1
                expense_date = start_of_month.strftime('%Y-%m-%d')
            elif frequency == 'weekly':
                # Add weekly expenses (approximately 4-5 per month)
                occurrences = 1  # One per month for weekly
                expense_date = start_of_month.strftime('%Y-%m-%d')
            elif frequency == 'monthly':
                # One per month
                occurrences = 1
                expense_date = start_of_month.strftime('%Y-%m-%d')
            elif frequency == 'yearly':
                # Check if this year's expense should be added
                if start_of_month.month == 1:  # Only in January
                    occurrences = 1
                    expense_date = start_of_month.strftime('%Y-%m-%d')
            
            if occurrences > 0 and expense_date:
                # Check if expense already exists for this month
                existing = await db.check_recurring_expense_generated(user_id, recurring_id, start_of_month.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                
                if not existing:
                    # Generate the expense(s)
                    if frequency == 'daily':
                        # Generate multiple daily expenses
                        for day_offset in range(occurrences):
                            expense_date = (start_of_month + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                            note = f"Recurring {frequency} expense"
                            expense_id = await db.add_expense(user_id, expense_date, amount, category, note, merchant)
                            generated_expenses.append({
                                "id": expense_id,
                                "date": expense_date,
                                "amount": amount,
                                "category": category,
                                "merchant": merchant
                            })
                    else:
                        # Single expense for weekly/monthly/yearly
                        note = f"Recurring {frequency} expense"
                        expense_id = await db.add_expense(user_id, expense_date, amount, category, note, merchant)
                        generated_expenses.append({
                            "id": expense_id,
                            "date": expense_date,
                            "amount": amount,
                            "category": category,
                            "merchant": merchant
                        })
                        
                        # Mark as generated
                        await db.mark_recurring_generated(recurring_id, expense_date)
        
        total_amount = sum(exp['amount'] for exp in generated_expenses)
        
        return {
            "status": "success",
            "message": f"Generated {len(generated_expenses)} recurring expense(s) for {today.strftime('%B %Y')}",
            "month": today.strftime('%B %Y'),
            "generated": generated_expenses,
            "total_amount": total_amount,
            "count": len(generated_expenses)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating recurring expenses: {str(e)}"}

# Income Tracking Tools
@mcp.tool()
async def add_income(api_key: str = None, session_id: str = None, date: str = None, amount: float = None, source: str = None, category: str = "Salary", note: str = ""):
    """Record an income transaction such as salary, freelance work, dividends, or other earnings.
    
    Tracks all sources of income to help calculate net worth and financial health.
    Supports multiple income categories for comprehensive financial tracking.
    
    Args:
        date: Income date in YYYY-MM-DD format. If not provided, defaults to today.
        amount: Income amount (required, must be positive number)
        source: Source of income (required, e.g., "Company Name", "Freelance Client", "Dividends", "Rental Income")
        category: Income category (default: "Salary"). Valid options: Salary, Freelance, Investment, Business, Rental, Other
        note: Additional notes (optional, e.g., "Monthly salary", "Project payment")
    
    Returns:
        Dictionary with status, income_id, amount, source, category, and success message.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Validate amount
        if amount is None:
            return {"status": "error", "message": "Amount is required and must be a positive number."}
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return {"status": "error", "message": "Amount must be a positive number."}
        except (ValueError, TypeError):
            return {"status": "error", "message": f"Invalid amount: '{amount}'. Amount must be a valid number."}
        
        # Validate source
        if not source or not source.strip():
            return {"status": "error", "message": "Source is required. Please provide the source of income (e.g., company name, client name)."}
        
        # Validate and parse date
        if not date or date.strip() == "":
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                try:
                    from dateparser import parse as dateparse
                    parsed_date = dateparse(date, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": datetime.now(timezone.utc)})
                    if parsed_date:
                        date = parsed_date.strftime("%Y-%m-%d")
                    else:
                        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Validate category
        valid_categories = ["Salary", "Freelance", "Investment", "Business", "Rental", "Other"]
        if category not in valid_categories:
            category = "Salary"  # Default to Salary if invalid
        
        income_id = await db.add_income(user_id, date, amount_float, source.strip(), category, note)
        
        if not income_id:
            await _log_audit(
                user_id,
                AuditAction.INCOME_ADDED,
                details={"amount": amount_float, "source": source.strip()},
                success=False,
                error_message="Failed to save income to database",
                resource_type="income",
            )
            return {"status": "error", "message": "Failed to save income to database."}
        
        await _log_audit(
            user_id,
            AuditAction.INCOME_ADDED,
            details={"amount": amount_float, "source": source.strip(), "category": category, "date": date},
            resource_type="income",
            resource_id=str(income_id),
        )
        return {
            "status": "success",
            "id": income_id,
            "message": f"Income recorded: ₹{amount_float:,.2f} from {source}",
            "amount": amount_float,
            "source": source.strip(),
            "category": category,
            "date": date
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding income: {str(e)}"}

@mcp.tool()
async def list_income(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """List income transactions in date range"""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        income_list = await db.get_income(user_id, start_date, end_date)
        total_income = sum(float(item.get("amount", 0)) for item in income_list)
        return {
            "status": "success",
            "income": income_list,
            "total": total_income
        }
    except Exception as e:
        return {"status": "error", "message": f"Error listing income: {str(e)}"}

@mcp.tool()
async def get_income_summary(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """Get income summary by source and category"""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        summary = await db.get_income_summary(user_id, start_date, end_date)
        return {
            "status": "success",
            "summary": summary,
            "period": f"{start_date} to {end_date}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting income summary: {str(e)}"}

# Savings Goals Tools
@mcp.tool()
async def add_savings_goal(
    api_key: str = None,
    session_id: str = None,
    goal_name: str = None,
    target_amount: float = None,
    target_date: str = None,
    current_amount: float = 0,
    goal_number: str = "",
):
    """Create a new savings goal with a custom name and optional goal number."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not goal_name or not str(goal_name).strip():
            return {"status": "error", "message": "Goal name is required"}
        if target_amount is None or float(target_amount) <= 0:
            return {"status": "error", "message": "Target amount must be a positive number"}
        if not target_date:
            target_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%d")
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return {"status": "error", "message": "target_date must be YYYY-MM-DD"}

        goal_id = await db.save_savings_goal(
            user_id,
            goal_name.strip(),
            float(target_amount),
            target_date,
            float(current_amount or 0),
            goal_number=str(goal_number or ""),
        )
        await _log_audit(
            user_id,
            AuditAction.SETTINGS_CHANGED,
            details={"action": "savings_goal_added", "goal_name": goal_name, "goal_number": goal_number},
            resource_type="savings_goal",
            resource_id=goal_id,
        )
        label = f"#{goal_number} " if goal_number else ""
        return {
            "status": "success",
            "id": goal_id,
            "message": f"Savings goal created: {label}{goal_name}",
            "goal_name": goal_name,
            "goal_number": goal_number,
            "target_amount": float(target_amount),
            "current_amount": float(current_amount or 0),
            "target_date": target_date,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding savings goal: {str(e)}"}


@mcp.tool()
async def set_savings_goal(api_key: str = None, session_id: str = None, goal_name: str = None, target_amount: float = None, target_date: str = None, current_amount: float = 0, goal_id: str = None, goal_number: str = ""):
    """Create or update a savings goal with target amount and deadline.
    
    Use goal_id to update an existing goal. Without goal_id, creates a new goal.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not goal_name or not str(goal_name).strip():
            return {"status": "error", "message": "Goal name is required"}
        if target_amount is None or float(target_amount) <= 0:
            return {"status": "error", "message": "Target amount must be positive"}
        if not target_date:
            return {"status": "error", "message": "target_date is required (YYYY-MM-DD)"}

        if goal_id:
            gid = await db.save_savings_goal(
                user_id, goal_name.strip(), float(target_amount), target_date,
                float(current_amount), goal_number=str(goal_number or ""), goal_id=goal_id,
            )
        else:
            gid = await db.save_savings_goal(
                user_id, goal_name.strip(), float(target_amount), target_date,
                float(current_amount), goal_number=str(goal_number or ""),
            )

        return {
            "status": "success",
            "id": gid,
            "message": f"Savings goal saved: {goal_name} — ₹{float(target_amount):,.2f} by {target_date}",
            "goal_name": goal_name,
            "goal_number": goal_number,
            "target_amount": float(target_amount),
            "current_amount": float(current_amount),
            "target_date": target_date,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error setting savings goal: {str(e)}"}


@mcp.tool()
async def delete_savings_goal(api_key: str = None, session_id: str = None, goal_id: str = None):
    """Remove a savings goal by its ID."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not goal_id:
            return {"status": "error", "message": "goal_id is required"}
        goal = await db.get_savings_goal(user_id, goal_id)
        if not goal:
            return {"status": "error", "message": "Savings goal not found"}
        deleted = await db.delete_savings_goal(user_id, goal_id)
        if not deleted:
            return {"status": "error", "message": "Could not delete savings goal"}
        await _log_audit(
            user_id,
            AuditAction.SETTINGS_CHANGED,
            details={"action": "savings_goal_deleted", "goal_name": goal.get("goal_name")},
            resource_type="savings_goal",
            resource_id=goal_id,
        )
        return {"status": "success", "message": f"Removed goal: {goal.get('goal_name', 'Goal')}", "goal_id": goal_id}
    except Exception as e:
        return {"status": "error", "message": f"Error deleting savings goal: {str(e)}"}


@mcp.tool()
async def list_savings_goals(api_key: str = None, session_id: str = None):
    """List all savings goals for the user."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        goals = await db.get_savings_goals(user_id)
        return {"status": "success", "goals": goals, "count": len(goals)}
    except Exception as e:
        return {"status": "error", "message": f"Error listing savings goals: {str(e)}"}


@mcp.tool()
async def track_savings_progress(api_key: str = None, session_id: str = None):
    """Track progress on all savings goals with AI recommendations"""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        goals = await db.get_savings_goals(user_id)
        
        if not goals:
            return {
                "status": "success",
                "message": "No savings goals set. Use set_savings_goal() to create one.",
                "goals": []
            }
        
        progress_list = []
        for goal in goals:
            goal_id = goal.get('id')
            goal_name = goal.get('goal_name', '')
            target_amount = float(goal.get('target_amount', 0))
            current_amount = float(goal.get('current_amount', 0))
            target_date = goal.get('target_date', '')
            
            progress_percentage = (current_amount / target_amount * 100) if target_amount > 0 else 0
            remaining = target_amount - current_amount
            
            # Calculate days until target
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            days_remaining = (target_dt - datetime.now()).days
            daily_savings_needed = remaining / days_remaining if days_remaining > 0 else 0
            
            progress_list.append({
                "goal_id": goal_id,
                "goal_name": goal_name,
                "goal_number": goal.get("goal_number", ""),
                "target_amount": target_amount,
                "current_amount": current_amount,
                "remaining": remaining,
                "progress_percentage": round(progress_percentage, 2),
                "target_date": target_date,
                "days_remaining": days_remaining,
                "daily_savings_needed": round(daily_savings_needed, 2)
            })
        
        # Generate AI recommendations
        ai_recommendations = []
        for progress in progress_list:
            if progress['progress_percentage'] < 50:
                ai_recommendations.append(f"💡 {progress['goal_name']}: You're at {progress['progress_percentage']:.1f}%. Consider increasing monthly savings.")
            elif progress['daily_savings_needed'] > 0:
                ai_recommendations.append(f"💡 {progress['goal_name']}: Save ${progress['daily_savings_needed']:.2f} daily to reach your goal on time.")
        
        return {
            "status": "success",
            "goals": progress_list,
            "ai_recommendations": ai_recommendations
        }
    except Exception as e:
        return {"status": "error", "message": f"Error tracking savings progress: {str(e)}"}

# Debt Management Tools
@mcp.tool()
async def add_debt(api_key: str = None, session_id: str = None, creditor_name: str = None, total_amount: float = None, interest_rate: float = None, minimum_payment: float = None, due_date: str = None, debt_type: str = "Credit Card"):
    """Add a debt obligation such as credit card, personal loan, mortgage, or student loan.
    
    Tracks debt details to help manage payments and calculate payoff strategies.
    
    Args:
        creditor_name: Name of creditor/lender (required)
        total_amount: Total debt amount (required, must be positive)
        interest_rate: Annual interest rate as percentage (required, e.g., 18.5 for 18.5%)
        minimum_payment: Minimum monthly payment (required)
        due_date: Next payment due date in YYYY-MM-DD format (required)
        debt_type: Type of debt (default: "Credit Card", options: Credit Card, Personal Loan, Mortgage, Student Loan, Auto Loan, Other)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        debt_id = await db.add_debt(
            user_id, 
            creditor_name, 
            float(total_amount), 
            float(interest_rate), 
            float(minimum_payment), 
            due_date, 
            debt_type
        )
        
        return {
            "status": "success",
            "id": debt_id,
            "message": f"Debt added: ${total_amount} to {creditor_name}",
            "creditor": creditor_name,
            "total_amount": float(total_amount),
            "interest_rate": float(interest_rate),
            "debt_type": debt_type
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding debt: {str(e)}"}

@mcp.tool()
async def record_debt_payment(api_key: str = None, session_id: str = None, debt_id: str = None, amount: float = None, payment_date: str = None):
    """Record a payment made toward a debt.
    
    Tracks debt payments to monitor progress in paying off debts.
    
    Args:
        debt_id: ID of the debt (required)
        amount: Payment amount (required, must be positive)
        payment_date: Payment date in YYYY-MM-DD format (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        payment_id = await db.record_debt_payment(user_id, debt_id, float(amount), payment_date)
        
        return {
            "status": "success",
            "id": payment_id,
            "message": f"Payment recorded: ${amount}",
            "amount": float(amount),
            "payment_date": payment_date
        }
    except Exception as e:
        return {"status": "error", "message": f"Error recording payment: {str(e)}"}

@mcp.tool()
async def get_debt_summary(api_key: str = None, session_id: str = None):
    """Get comprehensive summary of all debts with AI-powered payoff strategies.
    
    Returns total debt, remaining balances, and recommends avalanche or snowball
    payoff methods based on your debt profile.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        debts = await db.get_debts(user_id)
        
        if not debts:
            return {
                "status": "success",
                "message": "No debts recorded.",
                "debts": []
            }
        
        total_debt = 0
        debt_list = []
        
        for debt in debts:
            debt_id = debt.get('id')
            creditor = debt.get('creditor_name', '')
            total_amount = float(debt.get('total_amount', 0))
            interest_rate = float(debt.get('interest_rate', 0))
            minimum_payment = float(debt.get('minimum_payment', 0))
            debt_type = debt.get('debt_type', '')
            
            # Get payments made
            payments = await db.get_debt_payments(user_id, debt_id)
            paid_amount = sum(float(p[2]) for p in payments) if payments else 0
            remaining = total_amount - paid_amount
            
            total_debt += remaining
            
            debt_list.append({
                "debt_id": debt_id,
                "creditor": creditor,
                "total_amount": total_amount,
                "paid_amount": paid_amount,
                "remaining": remaining,
                "interest_rate": interest_rate,
                "minimum_payment": minimum_payment,
                "debt_type": debt_type
            })
        
        # Generate AI payoff strategies
        strategies = []
        if debt_list:
            # Sort by interest rate (highest first - avalanche method)
            sorted_debts = sorted(debt_list, key=lambda x: x['interest_rate'], reverse=True)
            strategies.append("💡 AVALANCHE METHOD: Pay off highest interest rate debt first to save on interest.")
            strategies.append(f"💡 Focus on: {sorted_debts[0]['creditor']} ({sorted_debts[0]['interest_rate']}% interest)")
        
        return {
            "status": "success",
            "total_debt": total_debt,
            "debts": debt_list,
            "payoff_strategies": strategies
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting debt summary: {str(e)}"}

# Investment Tracking Tools
@mcp.tool()
async def add_investment(api_key: str = None, session_id: str = None, investment_name: str = None, investment_type: str = None, amount: float = None, purchase_date: str = None, current_value: float = None):
    """Add an investment to your portfolio (stocks, bonds, mutual funds, crypto, etc.).
    
    Tracks investments to monitor portfolio performance and calculate gains/losses.
    
    Args:
        investment_name: Name or ticker of investment (required)
        investment_type: Type of investment (required, e.g., "Stock", "Bond", "Mutual Fund", "Crypto", "ETF")
        amount: Purchase amount (required, must be positive)
        purchase_date: Purchase date in YYYY-MM-DD format (required)
        current_value: Current market value (optional, defaults to purchase amount)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        current_val = float(current_value) if current_value else float(amount)
        investment_id = await db.add_investment(
            user_id, 
            investment_name, 
            investment_type, 
            float(amount), 
            purchase_date, 
            current_val
        )
        
        return {
            "status": "success",
            "id": investment_id,
            "message": f"Investment added: {investment_name}",
            "investment_name": investment_name,
            "investment_type": investment_type,
            "amount": float(amount),
            "current_value": current_val
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding investment: {str(e)}"}

@mcp.tool()
async def update_investment_value(api_key: str = None, session_id: str = None, investment_id: str = None, current_value: float = None, update_date: str = None):
    """Update the current market value of an investment.
    
    Use this to mark-to-market your investments and track performance over time.
    
    Args:
        investment_id: ID of the investment (required)
        current_value: Current market value (required, must be positive)
        update_date: Update date in YYYY-MM-DD format (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        await db.update_investment_value(user_id, investment_id, float(current_value), update_date)
        
        return {
            "status": "success",
            "message": f"Investment value updated to ${current_value}",
            "current_value": float(current_value),
            "update_date": update_date
        }
    except Exception as e:
        return {"status": "error", "message": f"Error updating investment: {str(e)}"}

@mcp.tool()
async def get_investment_portfolio(api_key: str = None, session_id: str = None):
    """Get comprehensive portfolio summary with performance analysis.
    
    Returns total portfolio value, gains/losses, performance by investment type,
    and AI-powered insights for portfolio optimization.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        investments = await db.get_investments(user_id)
        
        if not investments:
            return {
                "status": "success",
                "message": "No investments recorded.",
                "portfolio": []
            }
        
        total_invested = 0
        total_current_value = 0
        portfolio = []
        
        for inv in investments:
            investment_name = inv.get('investment_name', '')
            investment_type = inv.get('investment_type', '')
            amount = float(inv.get('amount', 0))
            current_value = float(inv.get('current_value', amount))
            
            total_invested += amount
            total_current_value += current_value
            
            gain_loss = current_value - amount
            gain_loss_percentage = (gain_loss / amount * 100) if amount > 0 else 0
            
            portfolio.append({
                "investment_name": investment_name,
                "investment_type": investment_type,
                "invested": amount,
                "current_value": current_value,
                "gain_loss": round(gain_loss, 2),
                "gain_loss_percentage": round(gain_loss_percentage, 2)
            })
        
        total_gain_loss = total_current_value - total_invested
        total_gain_loss_percentage = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0
        
        # Generate AI insights
        insights = []
        if total_gain_loss_percentage > 10:
            insights.append("📈 Great performance! Your portfolio is up significantly.")
        elif total_gain_loss_percentage < -10:
            insights.append("📉 Consider reviewing your investment strategy. Portfolio is down significantly.")
        else:
            insights.append("💡 Portfolio performance is stable. Consider diversifying for better returns.")
        
        return {
            "status": "success",
            "portfolio": portfolio,
            "summary": {
                "total_invested": total_invested,
                "total_current_value": total_current_value,
                "total_gain_loss": round(total_gain_loss, 2),
                "total_gain_loss_percentage": round(total_gain_loss_percentage, 2)
            },
            "ai_insights": insights
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting portfolio: {str(e)}"}

# Financial Reports Tools
@mcp.tool()
async def generate_financial_report(api_key: str = None, session_id: str = None, start_date: str = None, end_date: str = None):
    """Generate comprehensive financial report with income, expenses, savings, investments, and net worth.
    
    Creates a complete financial snapshot including spending analysis, savings rate,
    investment performance, debt status, and overall net worth calculation.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (required)
        end_date: End date in YYYY-MM-DD format (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Get expenses
        expenses = await db.get_expenses(user_id, start_date, end_date)
        total_expenses = sum(float(exp.get("amount", 0)) for exp in expenses) if expenses else 0
        expense_summary = await db.get_expense_summary(user_id, start_date, end_date)
        
        # Get income
        income_list = await db.get_income(user_id, start_date, end_date)
        total_income = sum(float(inc.get("amount", 0)) for inc in income_list) if income_list else 0
        
        # Get savings goals progress
        goals = await db.get_savings_goals(user_id)
        total_savings = sum(float(g.get('current_amount', 0)) for g in goals) if goals else 0
        
        # Get debt summary
        debts = await db.get_debts(user_id)
        debt_payments = {}
        total_debt = 0
        for debt in debts:
            debt_id = debt.get('id')
            total_amount = float(debt.get('total_amount', 0))
            payments = await db.get_debt_payments(user_id, debt_id)
            paid = sum(float(p[2]) for p in payments) if payments else 0
            remaining = total_amount - paid
            total_debt += remaining
        
        # Get investments
        investments = await db.get_investments(user_id)
        total_investments = sum(float(inv.get('current_value', 0)) for inv in investments) if investments else 0
        
        # Calculate net worth
        net_worth = total_income - total_expenses + total_savings + total_investments - total_debt
        
        # Generate insights
        savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
        
        return {
            "status": "success",
            "period": f"{start_date} to {end_date}",
            "income": {
                "total": total_income,
                "transactions": len(income_list) if income_list else 0
            },
            "expenses": {
                "total": total_expenses,
                "transactions": len(expenses) if expenses else 0,
                "by_category": expense_summary
            },
            "savings": {
                "total": total_savings,
                "goals_count": len(goals) if goals else 0
            },
            "debt": {
                "total": total_debt,
                "debts_count": len(debts) if debts else 0
            },
            "investments": {
                "total_value": total_investments,
                "investments_count": len(investments) if investments else 0
            },
            "net_worth": net_worth,
            "savings_rate": round(savings_rate, 2),
            "insights": {
                "net_income": total_income - total_expenses,
                "financial_health": "Excellent" if savings_rate > 20 else "Good" if savings_rate > 10 else "Needs Improvement"
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating report: {str(e)}"}

# Tax Calculation Tools with Multi-Jurisdiction Support
@mcp.tool()
async def estimate_taxes(api_key: str = None, session_id: str = None, tax_year: int = None, jurisdiction: str = "IN"):
    """Estimate tax liability based on income and deductible expenses with multi-jurisdiction support.
    
    Calculates estimated taxes using jurisdiction-specific tax rules for:
    - India (IN), USA (US), UK (GB), Canada (CA), Australia (AU), Germany (DE), France (FR), Singapore (SG), UAE (AE), Switzerland (CH)
    
    Args:
        tax_year: Tax year (required, e.g., 2024). Defaults to current year if not provided.
        jurisdiction: Tax jurisdiction code (default: "IN" for India). Options: IN, US, GB, CA, AU, DE, FR, SG, AE, CH
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        if not tax_year:
            tax_year = datetime.now().year
        
        # Get jurisdiction
        try:
            tax_jurisdiction = TaxJurisdiction(jurisdiction.upper())
        except ValueError:
            return {"status": "error", "message": f"Unsupported jurisdiction: {jurisdiction}. Supported: IN, US, GB, CA, AU, DE, FR, SG, AE, CH"}
        
        # Initialize tax engine for jurisdiction
        engine = tax_engine
        engine.jurisdiction = tax_jurisdiction
        
        start_date = f"{tax_year}-01-01"
        end_date = f"{tax_year}-12-31"
        
        # Get income for the year
        income_list = await db.get_income(user_id, start_date, end_date)
        total_income = sum(float(inc.get("amount", 0)) if isinstance(inc, dict) else float(inc[3]) for inc in income_list) if income_list else 0
        
        # Get expenses for deductible calculation
        expenses = await db.get_expenses(user_id, start_date, end_date)
        
        # Convert expenses to list of dicts for tax engine
        expense_dicts = []
        for exp in expenses:
            if isinstance(exp, dict):
                expense_dicts.append(exp)
            else:
                # Legacy format
                expense_dicts.append({
                    "category": exp[4] if len(exp) > 4 else "Other",
                    "amount": float(exp[3]) if len(exp) > 3 else 0,
                })
        
        # Calculate tax using multi-jurisdiction engine
        tax_result = engine.estimate_tax_from_expenses(
            income=total_income,
            expenses=expense_dicts,
            jurisdiction=tax_jurisdiction,
        )
        
        await _log_audit(
            user_id,
            AuditAction.TAX_ESTIMATED,
            details={
                "tax_year": tax_year,
                "jurisdiction": jurisdiction,
                "gross_income": total_income,
                "estimated_tax": tax_result.get("total_tax", 0),
            },
            success=True,
        )
        
        return {
            "status": "success",
            "tax_year": tax_year,
            "jurisdiction": jurisdiction,
            **tax_result,
            "recommendations": [
                "💡 Keep receipts for all deductible expenses",
                "💡 Consider contributing to retirement accounts to reduce taxable income",
                "💡 Consult a tax professional for accurate filing",
                f"💡 Deductible categories for {jurisdiction}: {', '.join(engine.get_deductible_categories()[:5])}",
            ]
        }
    except Exception as e:
        return {"status": "error", "message": f"Error estimating taxes: {str(e)}"}


@mcp.tool()
async def convert_currency_with_timing(
    api_key: str = None,
    session_id: str = None,
    amount: float = None,
    from_currency: str = None,
    to_currency: str = None,
    date: str = None,
):
    """Convert currency amount using historical exchange rates with proper timing.
    
    Uses the exchange rate from the specified date (or transaction date) for accurate
    financial calculations. Caches rates for performance.
    
    Args:
        amount: Amount to convert (required)
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'INR')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'INR')
        date: Date for conversion in YYYY-MM-DD format (uses historical rate if in past, defaults to today)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        if not amount or amount <= 0:
            return {"status": "error", "message": "Amount must be a positive number"}
        
        if not from_currency or not to_currency:
            return {"status": "error", "message": "Both from_currency and to_currency are required"}
        
        # Parse date
        conversion_date = None
        if date:
            try:
                conversion_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {"status": "error", "message": f"Invalid date format: {date}. Expected YYYY-MM-DD"}
        
        # Convert using FX manager
        result = await fx_manager.convert_amount(
            amount=amount,
            from_currency=from_currency,
            to_currency=to_currency,
            date=conversion_date,
        )
        
        return {
            "status": "success",
            **result,
        }
    except Exception as e:
        return {"status": "error", "message": f"Currency conversion failed: {str(e)}"}


@mcp.tool()
async def get_audit_logs(
    api_key: str = None,
    session_id: str = None,
    start_date: str = None,
    end_date: str = None,
    action: str = None,
    success: bool = None,
    source: str = None,
    category: str = None,
    severity: str = None,
    search: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """Search audit logs with filters, pagination, and full event metadata."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        try:
            start_dt, end_dt = parse_audit_dates(start_date, end_date) if (start_date or end_date) else (None, None)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        action_enum = None
        if action:
            try:
                action_enum = AuditAction(action.lower())
            except ValueError:
                pass
        result = await audit_logger.search_logs(
            user_id=user_id,
            start_date=start_dt,
            end_date=end_dt,
            action=action_enum,
            success=success,
            source=source,
            category=category,
            severity=severity,
            search=search,
            limit=min(limit, 500),
            offset=max(0, offset),
        )
        return {"status": "success", **result, "summary_7d": await audit_logger.get_action_counts(user_id, hours=24 * 7)}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving audit logs: {str(e)}"}


@mcp.tool()
async def get_audit_dashboard(
    api_key: str = None,
    session_id: str = None,
    hours: int = 168,
):
    """Audit analytics: success rate, timeline, breakdowns, and security alerts."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        dashboard = await audit_logger.get_dashboard(user_id, hours=min(hours, 24 * 30))
        return {"status": "success", "dashboard": dashboard}
    except Exception as e:
        return {"status": "error", "message": f"Error loading audit dashboard: {str(e)}"}


@mcp.tool()
async def export_audit_logs(
    api_key: str = None,
    session_id: str = None,
    start_date: str = None,
    end_date: str = None,
    action: str = None,
):
    """Export audit trail as CSV for compliance and offline review."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        start_dt, end_dt = (None, None)
        if start_date or end_date:
            start_dt, end_dt = parse_audit_dates(start_date or "2000-01-01", end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        action_enum = AuditAction(action.lower()) if action else None
        csv_data = await audit_logger.export_logs_csv(user_id, start_dt, end_dt, action_enum)
        await _log_audit(user_id, AuditAction.DATA_EXPORTED, details={"rows": csv_data.count("\n") - 1}, source="mcp")
        return {"status": "success", "format": "csv", "data": csv_data, "filename": f"audit_export_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"}
    except Exception as e:
        return {"status": "error", "message": f"Export failed: {str(e)}"}

# Credit Score Monitoring Tools
@mcp.tool()
async def record_credit_score(api_key: str = None, session_id: str = None, score: int = None, date: str = None, credit_bureau: str = "General"):
    """Record a credit score snapshot from a credit bureau.
    
    Tracks credit score over time to monitor credit health and identify trends.
    
    Args:
        score: Credit score (required, typically 300-850 range)
        date: Date in YYYY-MM-DD format (required)
        credit_bureau: Credit bureau name (default: "General", options: Experian, Equifax, TransUnion, General)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        score_id = await db.record_credit_score(user_id, int(score), date, credit_bureau)
        
        return {
            "status": "success",
            "id": score_id,
            "message": f"Credit score recorded: {score}",
            "score": int(score),
            "date": date,
            "credit_bureau": credit_bureau
        }
    except Exception as e:
        return {"status": "error", "message": f"Error recording credit score: {str(e)}"}

@mcp.tool()
async def get_credit_score_trend(api_key: str = None, session_id: str = None):
    """Get credit score trend over time with AI-powered improvement tips.
    
    Analyzes credit score history to identify trends and provides personalized
    recommendations for improving your credit score.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        scores = await db.get_credit_scores(user_id)
        
        if not scores:
            return {
                "status": "success",
                "message": "No credit scores recorded. Use record_credit_score() to add one.",
                "scores": []
            }
        
        # Get latest score
        latest_score = scores[0] if scores else None
        current_score = int(latest_score[1]) if latest_score else 0
        
        # Calculate trend
        if len(scores) > 1:
            previous_score = int(scores[1][1])
            trend = current_score - previous_score
            trend_direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
        else:
            trend = 0
            trend_direction = "no trend data"
        
        # Generate improvement tips
        tips = []
        if current_score < 580:
            tips.append("💡 Your credit score is poor. Focus on paying bills on time and reducing debt.")
        elif current_score < 670:
            tips.append("💡 Your credit score is fair. Keep paying bills on time and maintain low credit utilization.")
        elif current_score < 740:
            tips.append("💡 Your credit score is good. Maintain current habits to reach excellent range.")
        else:
            tips.append("💡 Excellent credit score! Keep up the good work.")
        
        if trend < 0:
            tips.append("⚠️ Your score is declining. Review recent credit activity and payment history.")
        
        return {
            "status": "success",
            "current_score": current_score,
            "trend": trend,
            "trend_direction": trend_direction,
            "score_history": [{"date": s[2], "score": int(s[1]), "bureau": s[3]} for s in scores[:10]],
            "improvement_tips": tips
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting credit score trend: {str(e)}"}

# Bill Reminders Tools
@mcp.tool()
async def add_bill_reminder(api_key: str = None, session_id: str = None, bill_name: str = None, amount: float = None, due_date: str = None, frequency: str = "monthly", category: str = "Bills"):
    """Add a recurring bill reminder for utilities, subscriptions, rent, or other regular payments.
    
    Creates reminders that will help you never miss a payment deadline.
    
    Args:
        bill_name: Name of the bill (required, e.g., "Electricity", "Netflix", "Rent")
        amount: Bill amount (required, must be positive)
        due_date: Next due date in YYYY-MM-DD format (required)
        frequency: Payment frequency (default: "monthly", options: weekly, monthly, quarterly, yearly)
        category: Expense category (default: "Bills")
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        reminder_id = await db.add_bill_reminder(
            user_id, 
            bill_name, 
            float(amount), 
            due_date, 
            frequency, 
            category
        )
        
        return {
            "status": "success",
            "id": reminder_id,
            "message": f"Bill reminder added: {bill_name} - ${amount} due {due_date}",
            "bill_name": bill_name,
            "amount": float(amount),
            "due_date": due_date,
            "frequency": frequency
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding bill reminder: {str(e)}"}

@mcp.tool()
async def get_upcoming_bills(api_key: str = None, session_id: str = None, days_ahead: int = 30):
    """Get list of all bills due within the specified number of days.
    
    Helps you plan ahead and ensure you have funds available for upcoming payments.
    
    Args:
        days_ahead: Number of days to look ahead (default: 30)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        today = datetime.now()
        end_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        bills = await db.get_upcoming_bills(user_id, today.strftime('%Y-%m-%d'), end_date)
        
        if not bills:
            return {
                "status": "success",
                "message": f"No bills due in the next {days_ahead} days.",
                "bills": []
            }
        
        bill_list = []
        reminders = []
        
        for bill in bills:
            bill_name = bill.get('bill_name', '')
            amount = float(bill.get('amount', 0))
            due_date = bill.get('due_date', '')
            
            due_dt = datetime.strptime(due_date, '%Y-%m-%d')
            days_until = (due_dt - today).days
            
            bill_list.append({
                "bill_name": bill_name,
                "amount": amount,
                "due_date": due_date,
                "days_until_due": days_until
            })
            
            if days_until <= 3:
                reminders.append(f"🚨 URGENT: {bill_name} (${amount}) is due in {days_until} days!")
            elif days_until <= 7:
                reminders.append(f"⚠️ REMINDER: {bill_name} (${amount}) is due in {days_until} days.")
        
        return {
            "status": "success",
            "upcoming_bills": bill_list,
            "reminders": reminders,
            "total_amount_due": sum(b['amount'] for b in bill_list)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting upcoming bills: {str(e)}"}

# Multi-Currency Support Tools
@mcp.tool()
async def add_expense_multicurrency(api_key: str = None, session_id: str = None, date: str = None, amount: float = None, currency: str = None, category: str = None, note: str = "", merchant: str = ""):
    """Add an expense in a different currency (automatically converts to base currency).
    
    Records expenses in foreign currencies and converts them to your base currency
    using current exchange rates for accurate financial tracking.
    
    Args:
        date: Expense date in YYYY-MM-DD format (required)
        amount: Expense amount in the specified currency (required)
        currency: Currency code (required, e.g., "USD", "EUR", "GBP", "JPY")
        category: Expense category (required)
        note: Additional notes (optional)
        merchant: Store or merchant name (optional)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Get exchange rate (simplified - in production, use real API)
        base_currency = await db.get_user_base_currency(user_id) or "USD"
        exchange_rate = await db.get_exchange_rate(currency, base_currency)
        
        converted_amount = float(amount) * exchange_rate
        
        expense_id = await db.add_expense(user_id, date, converted_amount, category, note, merchant)
        
        return {
            "status": "success",
            "id": expense_id,
            "message": f"Expense added: {amount} {currency} = {converted_amount:.2f} {base_currency}",
            "original_amount": float(amount),
            "original_currency": currency,
            "converted_amount": round(converted_amount, 2),
            "base_currency": base_currency,
            "exchange_rate": exchange_rate
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding multi-currency expense: {str(e)}"}

@mcp.tool()
async def set_base_currency(api_key: str = None, session_id: str = None, currency: str = None):
    """Set the user's base currency for all financial calculations and conversions.
    
    All amounts will be displayed and calculated in this currency. Multi-currency
    expenses will be converted to this base currency.
    
    Args:
        currency: Currency code (required, e.g., "INR", "USD", "EUR", "GBP")
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        await db.set_base_currency(user_id, currency)
        
        return {
            "status": "success",
            "message": f"Base currency set to {currency}",
            "base_currency": currency
        }
    except Exception as e:
        return {"status": "error", "message": f"Error setting base currency: {str(e)}"}

# Enhanced User Access Control
@mcp.tool()
async def create_user_role(api_key: str = None, session_id: str = None, role_name: str = None, permissions: Dict[str, Any] = None):
    """Create a custom user role with specific permissions (Admin feature).
    
    Allows administrators to define custom roles with specific access permissions
    for account sharing and multi-user scenarios.
    
    Args:
        role_name: Name of the role (required)
        permissions: Dictionary of permissions (required, e.g., {"view": True, "edit": False})
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # Check if user is admin
        is_admin = await auth.is_admin(user_id)
        if not is_admin:
            return {"status": "error", "message": "Admin access required"}
        
        role_id = await db.create_role(role_name, permissions)
        
        return {
            "status": "success",
            "id": role_id,
            "message": f"Role created: {role_name}",
            "role_name": role_name,
            "permissions": permissions
        }
    except Exception as e:
        return {"status": "error", "message": f"Error creating role: {str(e)}"}

@mcp.tool()
async def share_account_access(api_key: str = None, session_id: str = None, target_username: str = None, access_level: str = "view"):
    """Share account access with another user for family or shared account scenarios.
    
    Allows you to grant view or edit access to your financial data to trusted users.
    
    Args:
        target_username: Username of the user to share with (required)
        access_level: Access level (default: "view", options: "view", "edit")
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        access_id = await db.share_account_access(user_id, target_username, access_level)
        
        return {
            "status": "success",
            "id": access_id,
            "message": f"Account access shared with {target_username} ({access_level} level)",
            "target_user": target_username,
            "access_level": access_level
        }
    except Exception as e:
        return {"status": "error", "message": f"Error sharing access: {str(e)}"}

# Financial Institution Integration Tools
@mcp.tool()
async def connect_bank_account(api_key: str = None, session_id: str = None, bank_name: str = None, account_type: str = None, account_number_last4: str = None):
    """Connect a bank account for automatic transaction import.
    
    Links your bank account to enable automatic transaction syncing and categorization.
    
    Args:
        bank_name: Name of the bank (required)
        account_type: Type of account (required, e.g., "Checking", "Savings", "Credit Card")
        account_number_last4: Last 4 digits of account number for identification (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        connection_id = await db.connect_bank_account(
            user_id, 
            bank_name, 
            account_type, 
            account_number_last4
        )
        
        return {
            "status": "success",
            "id": connection_id,
            "message": f"Bank account connected: {bank_name} ({account_type})",
            "bank_name": bank_name,
            "account_type": account_type,
            "note": "Use sync_bank_transactions() to import transactions"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error connecting bank account: {str(e)}"}

@mcp.tool()
async def sync_bank_transactions(api_key: str = None, session_id: str = None, connection_id: str = None, start_date: str = None, end_date: str = None):
    """Sync transactions from a connected bank account.
    
    Imports transactions from your bank and automatically categorizes them using AI.
    
    Args:
        connection_id: ID of the bank connection (required)
        start_date: Start date in YYYY-MM-DD format (required)
        end_date: End date in YYYY-MM-DD format (required)
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        
        # In production, this would call bank API
        # For now, return a framework response
        sync_result = await db.sync_bank_transactions(user_id, connection_id, start_date, end_date)
        synced_count = sync_result.get("synced_count", 0)
        note = sync_result.get("note", "Transactions imported")

        from services.bank_plaid import plaid_service
        if synced_count == 0:
            if not plaid_service.is_configured():
                note = "Configure PLAID_CLIENT_ID and PLAID_SECRET in .env, then connect via Plaid Link"
            elif note == "manual_connection":
                note = "No transactions imported — ensure connection has plaid_access_token from Link flow"
        return {
            "status": "success",
            "message": f"Synced {synced_count} transactions from bank account",
            "connection_id": connection_id,
            "period": f"{start_date} to {end_date}",
            "transactions_synced": synced_count,
            "note": note,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error syncing transactions: {str(e)}"}


@mcp.tool()
async def sync_gmail_bills(api_key: str = None, session_id: str = None, max_messages: int = 40):
    """Scan the user's Gmail inbox for payment receipts, order confirmations, and bank
    transaction alert emails, then import them as expenses.

    Requires Google sign-in with Gmail read permission. Use when the user asks to fetch,
    sync, or import expenses from Gmail/email.
    """
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        from services.gmail_bills import gmail_bill_service
        result = await gmail_bill_service.sync_bills(user_id, max_messages=max_messages)
        return result
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Gmail sync failed: {str(exc)}"}


@mcp.tool()
async def get_plaid_link_token(api_key: str = None, session_id: str = None):
    """Get a Plaid Link token to connect a bank (requires PLAID_* env vars)."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        from services.bank_plaid import plaid_service
        return await plaid_service.create_link_token(user_id)
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def connect_bank_with_plaid(
    api_key: str = None,
    session_id: str = None,
    public_token: str = None,
    bank_name: str = "Linked Bank",
    account_type: str = "Checking",
    account_number_last4: str = "0000",
):
    """Exchange Plaid public_token and save bank connection for sync."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not public_token:
            return {"status": "error", "message": "public_token is required from Plaid Link"}
        from services.bank_plaid import plaid_service
        exchanged = await plaid_service.exchange_public_token(public_token)
        if exchanged.get("status") != "success":
            return exchanged
        connection_id = await db.connect_bank_account(
            user_id,
            bank_name,
            account_type,
            account_number_last4,
            plaid_access_token=exchanged.get("access_token"),
            plaid_item_id=exchanged.get("item_id"),
        )
        return {
            "status": "success",
            "id": connection_id,
            "message": "Bank linked via Plaid. Use sync_bank_transactions to import.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def send_bill_reminder_email(
    api_key: str = None,
    session_id: str = None,
    to_email: str = None,
    bill_name: str = None,
    due_date: str = None,
    amount: float = None,
):
    """Send bill due reminder email (requires SMTP_* in .env)."""
    try:
        user_id, error = await _get_user_id(api_key, session_id)
        if error:
            return {"status": "error", "message": error}
        if not to_email or not bill_name or not due_date or amount is None:
            return {"status": "error", "message": "to_email, bill_name, due_date, and amount are required"}
        from services.notifications import notification_service
        result = await notification_service.send_bill_reminder(
            to_email, bill_name, due_date, float(amount)
        )
        if result.get("status") == "success":
            await _log_audit(user_id, AuditAction.SETTINGS_CHANGED, details={"alert": "bill_email", "bill": bill_name})
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_help():
    """Get help and usage guide"""
    return {
        "status": "success",
        "message": "TrackExpensio - Simple & Fast!",
        "commands": [
            "# Authentication",
            "register_user(username, password) - Create account",
            "login_user(username, password) - Login and get API key",
            "get_user_info(api_key) - Get user information",
            "# Expense Management",
            "add_expense(api_key, date, amount, category, note, merchant) - Add expense",
            "quick_add_expense(api_key, amount, description, merchant) - Quick add",
            "list_expenses(api_key, start_date, end_date) - List expenses",
            "summarize(api_key, start_date, end_date) - Get summary",
            "ai_insights(api_key, start_date, end_date) - Get AI insights",
            "# Budget Management",
            "set_budget(api_key, category, amount) - Set monthly budget",
            "check_budget_status(api_key) - Check budget with alerts",
            "# Recurring Expenses",
            "add_recurring_expense(api_key, amount, category, frequency, merchant) - Add recurring",
            "generate_monthly_recurring(api_key) - Generate monthly recurring",
            "# Income Tracking",
            "add_income(api_key, date, amount, source, category, note) - Add income",
            "list_income(api_key, start_date, end_date) - List income",
            "get_income_summary(api_key, start_date, end_date) - Income summary",
            "# Savings Goals",
            "set_savings_goal(api_key, goal_name, target_amount, target_date) - Set goal",
            "track_savings_progress(api_key) - Track progress",
            "# Debt Management",
            "add_debt(api_key, creditor, amount, interest_rate, min_payment, due_date) - Add debt",
            "record_debt_payment(api_key, debt_id, amount, date) - Record payment",
            "get_debt_summary(api_key) - Debt summary",
            "# Investment Tracking",
            "add_investment(api_key, name, type, amount, purchase_date) - Add investment",
            "update_investment_value(api_key, investment_id, value, date) - Update value",
            "get_investment_portfolio(api_key) - Portfolio summary",
            "# Financial Reports",
            "generate_financial_report(api_key, start_date, end_date) - Full report",
            "# Tax Calculation",
            "estimate_taxes(api_key, tax_year) - Estimate tax liability",
            "# Credit Score",
            "record_credit_score(api_key, score, date, bureau) - Record score",
            "get_credit_score_trend(api_key) - Score trend",
            "# Bill Reminders",
            "add_bill_reminder(api_key, bill_name, amount, due_date, frequency) - Add reminder",
            "get_upcoming_bills(api_key, days_ahead) - Upcoming bills",
            "# Multi-Currency",
            "add_expense_multicurrency(api_key, date, amount, currency, category) - Multi-currency",
            "set_base_currency(api_key, currency) - Set base currency",
            "# Access Control",
            "create_user_role(api_key, role_name, permissions) - Create role",
            "share_account_access(api_key, username, access_level) - Share access",
            "# Bank Integration",
            "connect_bank_account(api_key, bank_name, account_type, last4) - Connect bank",
            "sync_bank_transactions(api_key, connection_id, start_date, end_date) - Sync transactions"
        ],
        "examples": [
            "quick_add_expense(api_key, 25.50, 'Morning coffee', 'Starbucks')",
            "set_budget(api_key, 'Food', 500)",
            "check_budget_status(api_key)",
            "add_income(api_key, '2024-01-15', 5000, 'Salary', 'Salary', 'Monthly salary')",
            "set_savings_goal(api_key, 'Vacation', 5000, '2024-12-31')",
            "add_debt(api_key, 'Credit Card', 2000, 18.5, 50, '2024-02-15')",
            "add_investment(api_key, 'AAPL Stock', 'Stock', 1000, '2024-01-01')",
            "generate_financial_report(api_key, '2024-01-01', '2024-01-31')",
            "estimate_taxes(api_key, 2024)",
            "get_upcoming_bills(api_key, 30)"
        ],
        "features": [
            "✅ Budget System with Alerts",
            "✅ Recurring Expenses Auto-Tracking",
            "✅ AI-Powered Expense Categorization",
            "✅ Income Tracking",
            "✅ Savings Goals",
            "✅ Debt Management",
            "✅ Investment Tracking",
            "✅ Financial Reports",
            "✅ Tax Estimation",
            "✅ Credit Score Monitoring",
            "✅ Bill Reminders",
            "✅ Multi-Currency Support",
            "✅ User Access Control",
            "✅ Bank Integration Framework"
        ]
    }

TOOL_REGISTRY = {
    "register_user": register_user,
    "login_user": login_user,
    "get_user_info": get_user_info,
    "add_expense": add_expense,
    "list_expenses": list_expenses,
    "get_expenses_overview": get_expenses_overview,
    "summarize": summarize,
    "ai_insights": ai_insights,
    "quick_add_expense": quick_add_expense,
    "document_expense_from_rag": document_expense_from_rag,
    "delete_expense": delete_expense,
    "update_expense": update_expense,
    "set_budget": set_budget,
    "check_budget_status": check_budget_status,
    "add_recurring_expense": add_recurring_expense,
    "generate_monthly_recurring": generate_monthly_recurring,
    "add_income": add_income,
    "list_income": list_income,
    "get_income_summary": get_income_summary,
    "set_savings_goal": set_savings_goal,
    "add_savings_goal": add_savings_goal,
    "delete_savings_goal": delete_savings_goal,
    "list_savings_goals": list_savings_goals,
    "track_savings_progress": track_savings_progress,
    "add_debt": add_debt,
    "record_debt_payment": record_debt_payment,
    "get_debt_summary": get_debt_summary,
    "add_investment": add_investment,
    "update_investment_value": update_investment_value,
    "get_investment_portfolio": get_investment_portfolio,
    "generate_financial_report": generate_financial_report,
    "estimate_taxes": estimate_taxes,
    "record_credit_score": record_credit_score,
    "get_credit_score_trend": get_credit_score_trend,
    "add_bill_reminder": add_bill_reminder,
    "get_upcoming_bills": get_upcoming_bills,
    "add_expense_multicurrency": add_expense_multicurrency,
    "set_base_currency": set_base_currency,
    "create_user_role": create_user_role,
    "share_account_access": share_account_access,
    "connect_bank_account": connect_bank_account,
    "connect_bank_with_plaid": connect_bank_with_plaid,
    "get_plaid_link_token": get_plaid_link_token,
    "sync_bank_transactions": sync_bank_transactions,
    "sync_gmail_bills": sync_gmail_bills,
    "send_bill_reminder_email": send_bill_reminder_email,
    "yahoo_finance": yahoo_finance,
    "smart_stock_analyze": smart_stock_analyze,
    "get_stock_return_one_year": get_stock_return_one_year,
    "get_stock_returns": get_stock_returns,
    "convert_currency_with_timing": convert_currency_with_timing,
    "get_audit_logs": get_audit_logs,
    "get_audit_dashboard": get_audit_dashboard,
    "export_audit_logs": export_audit_logs,
    "get_help": get_help,
}

# Start the server
if __name__ == "__main__":
    # For Claude Desktop, use STDIO transport
    # For HTTP server, use: mcp.run(transport="http", host="0.0.0.0", port=8000)
    mcp.run()  # Default is STDIO, which Claude Desktop uses
