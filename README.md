# TrackExpensio

> AI-powered personal finance assistant — track expenses, sync banks, and get insights through natural language.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| AI / Agent | Groq (Llama 3.3), FastMCP, function-calling |
| Database | MongoDB (Motor async) |
| Auth | Google OAuth, API keys, guest sessions |
| Integrations | Plaid, Gmail, Yahoo Finance, Whisper |
| Frontend | HTML, CSS, Vanilla JS |

---

## Features

- **AI Chat** — add expenses, set budgets, and query data in plain English
- **Bank Sync** — connect accounts via Plaid; import and deduplicate transactions automatically
- **India Bank Support** — Account Aggregator (Setu / Finvu / OneMoney) + SMS parser for Indian bank alerts
- **Gmail Parsing** — extract real transactions from bank alert emails, filter out promotions
- **Receipt Upload** — RAG pipeline extracts merchant, amount, and date from bill images/PDFs
- **Voice Commands** — speak expenses via Whisper speech-to-text
- **Budgets & Goals** — set category budgets and track savings progress
- **Investments & Debt** — portfolio tracking, debt payoff records
- **Tax Estimation** — multi-jurisdiction engine (IN, US, GB, CA, AU, DE, FR, SG, AE, CH)
- **FX Conversion** — historical exchange rates with MongoDB caching
- **Reports** — monthly financial summaries exported as PDF
- **Security** — bcrypt passwords, rate limiting, IP blocking, audit logs

---

## Quick Start

**Prerequisites:** Python 3.10+, MongoDB

```bash
git clone <repo-url> && cd project
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=ai_expense_tracker
GROQ_API_KEY=your_groq_api_key
GROQ_AGENT_MODEL=llama-3.3-70b-versatile

# Optional
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
PLAID_CLIENT_ID=...
PLAID_SECRET=...
OPENAI_API_KEY=...         # for Whisper voice
SMTP_HOST=smtp.gmail.com   # for bill reminders
```

```bash
python run_app.py
# → http://127.0.0.1:8080
```

---

## Project Structure

```
├── main.py              # 44 MCP tools (expenses, budgets, investments, …)
├── api.py               # FastAPI routes and middleware
├── chat_agent.py        # Groq agent with function-calling
├── database.py          # MongoDB operations
├── auth.py              # API key + session authentication
├── security.py          # Rate limiting, IP blocking, bcrypt
├── tax_rules.py         # Multi-jurisdiction tax engine
├── fx_timing.py         # Historical FX rate management
├── audit_logger.py      # Action audit trail
├── services/
│   ├── bank_plaid.py        # Plaid Link + transaction sync
│   ├── account_aggregator.py # India AA integration
│   ├── gmail_bills.py        # Gmail receipt parsing
│   ├── sms_parser.py         # Indian bank SMS parser
│   ├── transaction_filters.py # Promo / credit filtering
│   ├── notifications.py      # Email bill reminders
│   └── pdf_reports.py        # PDF report generation
└── static/
    ├── plaid_link.js
    ├── india_bank_sync.js
    └── voice_recording.js
```

---

## How MCP Works in TrackExpensio

**MCP (Model Context Protocol)** is the backbone of the AI layer. Instead of hardcoding what the AI can do, every financial action is registered as a typed Python function — a *tool* — that the agent can discover and call at runtime.

### The Pattern

```
User message → Groq agent reads tool schemas → picks the right tool(s) → Python executes → MongoDB writes
```

The AI never writes to the database directly. It proposes a tool call; Python validates and executes it. This keeps the AI layer stateless and the business logic safe and testable.

### How Tools Are Defined (`main.py`)

```python
@mcp.tool()
async def add_expense(
    api_key: str, date: str, amount: float,
    category: str, note: str, merchant: str
) -> dict:
    """Add a new expense entry."""
    user_id = await auth.authenticate_user(api_key=api_key)
    return await db.add_expense(user_id, date, amount, category, note, merchant)
```

Each `@mcp.tool()` decorator registers the function in the `TOOL_REGISTRY` with its name, docstring, and parameter schema — all automatically exposed to the agent.

### How the Agent Uses Tools (`chat_agent.py`)

The Groq agent receives the full tool registry as function schemas. When a user says *"Add ₹500 for lunch at Swiggy"*, the agent:

1. Parses intent from the message
2. Selects `add_expense` from the registry
3. Fills in `amount=500`, `merchant="Swiggy"`, `category="Food"`, `date=today`
4. Returns a structured tool call — Python executes it

For complex messages like *"Add lunch expense and show this month's total"*, the agent issues **parallel tool calls** — `add_expense` and `list_expenses` run simultaneously.

### Three Ways to Call a Tool

| Method | How |
|---|---|
| AI Chat | Agent selects and calls tools automatically |
| REST API | `POST /api/mcp/<tool_name>` with JSON body |
| Claude Desktop | `python mcp_server.py` — runs in STDIO mode |

### Why This Architecture

Adding a new capability means writing one Python function with `@mcp.tool()`. The agent can use it immediately — no prompt changes, no new API routes needed. It makes the system naturally extensible.

---

## MCP Tools (44 total)

Expenses · Budgets · Income · Recurring · Savings Goals · Debt · Investments · Reports · Tax · Credit Score · Bill Reminders · Multi-Currency · FX Conversion · Yahoo Finance · Bank Sync · Audit Logs · User Management

All tools are callable via chat, the REST API (`POST /api/mcp/<tool>`), or Claude Desktop (STDIO mode).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Chat with the AI assistant |
| `POST` | `/api/mcp/{tool_name}` | Call any MCP tool directly |
| `POST` | `/api/session/create` | Create a guest session |
| `GET` | `/api/user/profile` | Get user profile (after Google login) |
| `GET` | `/api/health` | Health check |

---

## Optional Extras

```bash
# Receipt RAG + local Whisper (~3.5 GB)
pip install -r requirements-ml.txt

# Standalone MCP server for Claude Desktop
python mcp_server.py
```

---

## Deployment

Render one-click deploy via `render.yaml`. Set `MONGODB_URI`, `GROQ_API_KEY`, and other secrets in the Render dashboard — never commit them.

---

## License

MIT
