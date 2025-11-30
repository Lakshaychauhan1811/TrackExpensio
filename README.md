# TrackExpensio

A complete financial intelligence assistant powered by FastMCP, LangGraph, and Groq AI. Track expenses, manage budgets, analyze investments, and get AI-powered insights—all through natural language.

## Features

- 💰 **Expense Tracking** - Add expenses via chat or forms with automatic categorization
- 📊 **Budget Management** - Set budgets and get real-time alerts
- 📈 **Yahoo Finance Integration** - Get live stock and mutual fund data
- 🧾 **Document Processing** - Upload receipts/bills, AI extracts expense details automatically
- 💬 **AI Chat Assistant** - Natural language interface for all financial operations
- 🔐 **Google OAuth** - Secure login with data persistence

## Quick Start

### Prerequisites

- Python 3.10+
- MongoDB (local or cloud)
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd project
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=ai_expense_tracker
GROQ_API_KEY=your_groq_api_key_here
GROQ_AGENT_MODEL=llama-3.3-70b-versatile

# Optional: Google OAuth (for login)
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://127.0.0.1:8080/auth/google/callback
GOOGLE_PROJECT_ID=your_project_id
```

### Run the Application

**For Web Application (Recommended):**
```bash
python run_app.py
```

Or with auto-reload for development:
```bash
python run_app.py --reload
```

The server will start at `http://127.0.0.1:8080`

**Important:** For web applications, always use `run_app.py`. The MCP tools are integrated into FastAPI endpoints (`/api/mcp/{tool_name}`), so you don't need a separate MCP server.

**Standalone MCP Server (Optional - for MCP Protocol Clients only):**
```bash
# STDIO mode (for Claude Desktop)
python mcp_server.py

# HTTP/SSE mode (for direct MCP protocol access)
python mcp_server.py --transport http --host 0.0.0.0 --port 8000
python mcp_server.py --transport sse --host 0.0.0.0 --port 8000
```

**Note:** The standalone server uses the MCP protocol, which is different from the FastAPI REST API. For web apps, use `run_app.py`.

## Project Structure

- **`main.py`** - MCP server with all financial tools (expenses, budgets, investments, etc.)
- **`mcp_server.py`** - Standalone MCP server entry point (for MCP protocol clients)
- **`api.py`** - FastAPI web server and client interface
- **`chat_agent.py`** - LangGraph-powered AI chat agent
- **`services/`** - RAG processing and storage utilities
- **`static/`** - Frontend assets (CSS, JavaScript)
- **`templates/`** - HTML templates

## Available Tools

The MCP server provides **40+ financial tools** including:
- Expense Management (add, list, summarize, delete, update)
- Budget Management (set budget, check status)
- Income Tracking (add, list, summarize)
- Savings Goals (set goals, track progress)
- Debt Management (add debt, record payments, get summary)
- Investment Tracking (add investments, update values, portfolio)
- Yahoo Finance Integration (real-time stock data)
- Document Processing (RAG-based receipt parsing)
- Bill Reminders (add reminders, get upcoming bills)
- Financial Reports (comprehensive reports)
- Tax Estimation
- Credit Score Tracking
- Multi-Currency Support
- Bank Integration Framework

## Usage

1. Open `http://127.0.0.1:8080` in your browser
2. A guest session is created automatically
3. Use the AI chat to:
   - "Add 500 rupees for groceries today from Big Bazaar"
   - "Show my expenses this month"
   - "What's the price of AAPL stock?"
   - "Set a budget of 5000 for Food category"
4. Or use the UI sections for manual entry

## API Endpoints

- `POST /api/chat` - Chat with AI assistant
- `POST /api/mcp/{tool_name}` - Call MCP tools directly
- `POST /api/session/create` - Create guest session
- `GET /api/user/profile` - Get user profile (after Google login)

## License

MIT
