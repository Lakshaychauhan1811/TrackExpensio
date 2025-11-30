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
import dateparser
from dateparser.search import search_dates

from groq import Groq

from main import TOOL_REGISTRY

GROQ_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are TrackExpensio, a helpful AI finance assistant. Your role is to help users manage their finances through natural conversation.

CORE PRINCIPLES:
1. Always use tools to perform actions (add expenses, check budgets, fetch stock data, etc.)
2. When user mentions MULTIPLE tasks in one message, you MUST call ALL relevant tools IN PARALLEL (same tool_call batch)
   - Example: "What's ITC share price and I spent 500 in food this week" → call both yahoo_finance AND add_expense together
   - Example: "Add 2000 for travel and set savings goal of 30000" → call add_expense AND set_savings_goal together
3. When user mentions MULTIPLE expenses in one message, you MUST call add_expense SEPARATELY for EACH expense (but all in parallel)
4. Default currency is INR unless user explicitly mentions another (dollar, euro, $, €, etc.)
5. Extract dates from user's message - DEFAULT to CURRENT DATE (2025) unless user specifically mentions a past date
6. Be conversational and helpful - answer questions naturally when appropriate

EXPENSE ADDITION:
- Extract for EACH expense: amount, category, date, merchant (if mentioned), currency
- Categories: Food, Travel, Bills, Shopping, Entertainment, Utilities, Rent, Health, Education, Other
- If user lists multiple expenses, make separate add_expense calls for each (all in parallel)
- Always use proper date format (YYYY-MM-DD) - parse natural dates like "23 nov" or "november 23" correctly
- If no date mentioned, use TODAY's date (current date in 2025)

EXPENSE QUERIES:
- When user asks to see expenses (e.g., "show my expenses", "tell me this month expenses", "where did I spend"), use list_expenses tool
- Always include date range (start_date and end_date) when calling list_expenses
- For "this month" queries, use current month's first day to today
- For "last week" queries, use 7 days ago to today
- When showing expenses, format them clearly with: date, amount, category, merchant (where spent)
- If user asks "where I spent" or "where did I spend", make sure to show the merchant/location for each expense

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

STOCK QUERIES:
- Use yahoo_finance tool for stock prices, market cap, returns
- Use get_stock_returns for historical returns analysis
- Support all stock exchanges (US, India.NS, UK.L, etc.)
- These are QUERY tools, not storage tools
- **CRITICAL: When user specifies a price range (e.g., "stocks between ₹100-500"), you MUST check if the stock price falls within that range BEFORE recommending it**
- If a stock's price is outside the requested range, DO NOT recommend it. Instead, explain that you can only look up specific stock symbols, not search by price range
- If user asks for stock recommendations in a price range, suggest they provide specific stock symbols to check, or explain that you need specific ticker symbols to look up prices
- Always verify the stock price matches user's constraints before presenting it

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
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
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
            "name": "yahoo_finance",
            "description": "Get real-time market data for stocks, ETFs, or mutual funds. Returns price, market cap, volume, P/E ratio, dividend yield, returns (1D/1W/1M/3M/6M/1Y/3Y), sector, industry, and recommendations. Use ticker symbols like AAPL, TSLA, or INFY.NS for Indian stocks.",
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
) -> Dict[str, Any]:
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return {"status": "error", "message": f"Tool {name} not available."}

    name, arguments = _maybe_upgrade_quick_add(name, arguments)
    kwargs = dict(arguments or {})

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
    return result if isinstance(result, dict) else {"status": "success", "data": result}


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

    builtin = await _maybe_handle_builtin_query(message, session_id, api_key)
    if builtin is not None:
        return builtin

    # Extract price range if mentioned
    price_range = _extract_price_range(message)
    
    # Enhance system prompt if price range is detected
    enhanced_prompt = SYSTEM_PROMPT
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
                
                exec_result = await _execute_tool(fn_name, args, session_id, api_key)
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

        # Format user-friendly reply (pass original message for price range checking)
        formatted_reply = _format_user_friendly_reply(llm_reply, tool_results, original_message=message)
        
        # Use formatted reply if it's different, otherwise use cleaned LLM reply
        if formatted_reply and formatted_reply != llm_reply and len(formatted_reply) > 20:
            reply = formatted_reply
        else:
            # Use LLM's natural response if it's good, otherwise use formatted
            cleaned_llm = _clean_json_from_reply(llm_reply)
            if cleaned_llm and len(cleaned_llm) > 10 and not any(x in cleaned_llm.lower() for x in ['status', 'error', 'success', '{', '}']):
                reply = cleaned_llm
            else:
                reply = formatted_reply if formatted_reply else "Task completed successfully."
        
        return {"reply": reply, "tool_results": tool_results}

    # No tool call, just reply naturally
    reply = assistant_payload["content"] or "Let me know how I can assist."
    # Clean any accidental JSON from natural responses
    reply = _clean_json_from_reply(reply)
    return {"reply": reply, "tool_results": []}


async def _maybe_handle_builtin_query(message: str, session_id: Optional[str], api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Handle common expense queries directly for faster responses."""
    text = (message or "").lower()
    
    # Determine date range based on query
    if "this month" in text or "current month" in text:
        start_date, end_date = _current_month_range()
        period = "this month"
    elif "last week" in text:
        today = datetime.now(timezone.utc)
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        period = "last week"
    elif "this week" in text:
        today = datetime.now(timezone.utc)
        start_date = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        period = "this week"
    else:
        # Default to current month
        start_date, end_date = _current_month_range()
        period = "this month"
    
    # If user asks for expenses with dates and merchants
    if any(keyword in text for keyword in ["expense", "spent", "spend", "where"]) and any(keyword in text for keyword in ["date", "when", "where", "show", "list", "tell"]):
        expenses = await _execute_tool(
            "list_expenses",
            {"start_date": start_date, "end_date": end_date},
            session_id,
            api_key,
        )
        
        if isinstance(expenses, dict) and expenses.get("status") == "error":
            return expenses
        
        if isinstance(expenses, list) and expenses:
            reply_lines = [f"Here are your expenses for {period}:"]
            total = 0
            for exp in expenses:
                date_str = exp.get("date", "")
                amount = exp.get("amount", 0)
                category = exp.get("category", "Other")
                merchant = exp.get("merchant", "")
                currency = exp.get("metadata", {}).get("currency", "INR")
                currency_symbol = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}.get(currency, currency)
                
                # Format date
                try:
                    if date_str:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        formatted_date = date_obj.strftime("%d %b %Y")
                    else:
                        formatted_date = "Unknown date"
                except:
                    formatted_date = date_str
                
                # Build expense line
                expense_line = f"• {formatted_date}: {currency_symbol}{amount:,.2f} for {category}"
                if merchant:
                    expense_line += f" at {merchant}"
                reply_lines.append(expense_line)
                total += amount
            
            reply_lines.append(f"\nTotal: {currency_symbol}{total:,.2f}")
            return {"reply": "\n".join(reply_lines), "tool_results": [{"tool": "list_expenses", "result": expenses}]}
        else:
            return {"reply": f"I couldn't find any expenses for {period}.", "tool_results": [{"tool": "list_expenses", "result": expenses}]}
    
    return None


def _current_month_range() -> tuple[str, str]:
    today = datetime.now(timezone.utc)
    start = datetime(today.year, today.month, 1)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


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
        
        elif tool_name == "yahoo_finance":
            if tool_result.get("status") == "success":
                data = tool_result.get("data", {})
                symbol = data.get("symbol", "")
                name = data.get("name", "")
                price = data.get("price", 0)
                currency = data.get("currency", "USD")
                currency_symbol = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}.get(currency, currency)
                
                # If price range exists and stock doesn't match, warn user
                if price_range:
                    min_price, max_price = price_range
                    if price < min_price or price > max_price:
                        response = f"⚠️ {name or symbol} ({symbol}) - Price: {currency_symbol}{price:,.2f}\n\n"
                        response += f"This stock's price ({currency_symbol}{price:,.2f}) is outside your requested range of {currency_symbol}{min_price:,.2f} - {currency_symbol}{max_price:,.2f}.\n\n"
                        response += "I can only look up specific stock symbols. To find stocks in your price range, you can:\n"
                        response += "• Search for stocks on financial websites (e.g., NSE, BSE for Indian stocks)\n"
                        response += "• Use stock screeners to filter by price range\n"
                        response += "• Provide me with specific stock symbols you'd like me to check\n\n"
                        response += "Would you like me to check any specific stock symbols for you?"
                        return response
                
                # Format market cap
                market_cap = data.get("market_cap")
                market_cap_str = ""
                if market_cap:
                    if market_cap >= 1e12:
                        market_cap_str = f"Market Cap: {currency_symbol}{market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"Market Cap: {currency_symbol}{market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"Market Cap: {currency_symbol}{market_cap/1e6:.2f}M"
                    else:
                        market_cap_str = f"Market Cap: {currency_symbol}{market_cap:,.2f}"
                
                # Get returns
                returns = data.get("returns", {})
                returns_parts = []
                if returns:
                    if returns.get("1y") is not None:
                        returns_parts.append(f"1Y: {returns['1y']:+.2f}%")
                    if returns.get("3y") is not None:
                        returns_parts.append(f"3Y: {returns['3y']:+.2f}%")
                
                # Build response
                response = f"📈 {name or symbol} ({symbol})\n"
                response += f"Price: {currency_symbol}{price:,.2f}"
                if market_cap_str:
                    response += f"\n{market_cap_str}"
                if returns_parts:
                    response += f"\nReturns: {' | '.join(returns_parts)}"
                
                return response
        
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
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    text_lower = text.lower()
    today = datetime.now(timezone.utc)
    
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
        # Default to CURRENT YEAR (2025) if year not specified
        year = int(match.group(3)) if match.group(3) else today.year
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        if month_str in month_map:
            try:
                date_obj = datetime(year, month_map[month_str], day)
                # If date is in future (more than 30 days ahead), assume it's from current year but adjust if needed
                # Only adjust if it's clearly in the future (e.g., if today is Nov 2025 and user says "23 dec", use 2025)
                # But if user says "23 jan" and we're in Nov, they likely mean Jan 2026, so keep it
                # For simplicity, if date is more than 2 months in future, use current year
                days_ahead = (date_obj - today).days
                if days_ahead > 60:  # More than 2 months ahead
                    date_obj = datetime(today.year, month_map[month_str], day)
                # If still in future and we're past that month this year, use next year
                elif date_obj > today and today.month > month_map[month_str]:
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
        # pick last found date (most specific)
        parsed_date = results[-1][1]
        # Ensure date is not too far in the past (more than 5 years) - likely user meant current year
        if (today - parsed_date).days > 1825:  # More than 5 years
            # Try to parse again with current year
            parsed_date = parsed_date.replace(year=today.year)
            # If that makes it future, use previous occurrence
            if parsed_date > today:
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

