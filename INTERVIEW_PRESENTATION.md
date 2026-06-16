# TrackExpensio — Interview Presentation Guide

Use this document to explain and demo your project in technical interviews.

---

## 2-Minute Elevator Script (memorize this)

> Hi, I'd like to show you **TrackExpensio** — an AI-powered personal finance assistant I built.
>
> The problem it solves is simple: people track money in too many places — chat, receipts, bank apps, email. I wanted one system where you can just say *"I spent 500 on lunch today"* or *"show my expenses this month"* and it actually works.
>
> Architecturally, the core idea is **MCP — Model Context Protocol**. All financial actions — add expense, set budget, sync bank, fetch stock prices — are exposed as **tools** in `main.py`. A **Groq-powered AI agent** reads the user's message, picks the right tools, and executes them. **FastAPI** serves the web UI and REST API, and **MongoDB** stores everything per user.
>
> For integrations, I connected **Plaid** for bank transaction sync, **Gmail** for bill parsing, **Yahoo Finance** for live stock data, and a **RAG pipeline** for receipt uploads.
>
> Let me walk you through a quick live demo — chat, dashboard, and bank sync.

**Time:** ~90–120 seconds. Then go straight into the demo.

---

## Slide-by-Slide Outline (8 slides)

### Slide 1 — Title
**TrackExpensio**  
AI-Powered Personal Finance Assistant  
*Your Name · Python · FastAPI · MCP · Groq · MongoDB*

**Say:** "Full-stack agentic finance app with natural language as the main interface."

---

### Slide 2 — Problem
- Manual expense tracking is tedious
- Data scattered: chat, receipts, banks, email
- Users want conversational control, not forms only

**Say:** "People don't want another spreadsheet — they want to talk to their finance app."

---

### Slide 3 — Solution
- Natural language chat for all finance actions
- Web dashboard for manual entry and charts
- Bank sync, receipt AI, bill reminders, investments

**Say:** "One platform: chat + UI + integrations."

---

### Slide 4 — Architecture

```
Browser → FastAPI (api.py) → Chat Agent (Groq) → MCP Tools (main.py) → MongoDB
                                    ↓
                              Plaid / Gmail / Yahoo Finance
```

**Say:** "Clean separation: HTTP layer, AI orchestration, business logic as tools, data in MongoDB."

---

### Slide 5 — Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, Uvicorn |
| AI | Groq (Llama 3.3), function-calling agent |
| Protocol | MCP (FastMCP) — 40+ financial tools |
| Database | MongoDB (Motor async) |
| Auth | API keys, sessions, Google OAuth |
| Integrations | Plaid, Gmail, Yahoo Finance |
| AI extras | RAG (receipts), Whisper (voice), FAISS |
| Frontend | HTML, CSS, JavaScript |

---

### Slide 6 — Key Features (demo these)
1. **AI Chat** — add/list expenses, budgets, stocks in natural language
2. **Bank Sync** — Plaid Link → import transactions → deduplicate
3. **Receipt Upload** — RAG extracts amount, merchant, date
4. **Budgets & Savings** — set limits, track goals
5. **Reports & Insights** — monthly overview, AI insights

---

### Slide 7 — Technical Highlights
- MCP tools are real Python functions with auth + validation
- Agent can call **multiple tools in parallel** (e.g. expense + stock price)
- Plaid: link token → exchange → paginated sync, skip pending/credits
- Per-user data isolation, rate limiting, audit logs
- Extensible: new tool in registry → agent can use it without rewriting chat

---

### Slide 8 — Challenges & Next Steps

**Challenges solved:**
- AI tool selection → system prompt + schemas + date context
- Duplicate bank txns → dedup by Plaid `transaction_id`
- Multi-intent messages → parallel tool calls

**Future:**
- Plaid webhooks for real-time updates
- Encrypt access tokens at rest
- Scheduled Gmail bill sync
- Mobile app

**Closing:** "Not just CRUD — an agentic finance platform where NL drives real actions through MCP."

---

## Live Demo Script (5–7 minutes)

### Before the interview
```bash
# Terminal 1 — ensure MongoDB is running
# Terminal 2
cd project
python run_app.py
```
Open: **http://127.0.0.1:8080**

Check `.env`: `GROQ_API_KEY`, `MONGODB_URI`, `PLAID_CLIENT_ID`, `PLAID_SECRET`

---

### Demo Part 1 — AI Chat (2 min) ⭐ Most important

| Step | You type / say | What to explain |
|------|----------------|-----------------|
| 1 | "Add 500 rupees for lunch at Swiggy today" | Agent calls `add_expense` tool |
| 2 | "Show my expenses this month" | Agent calls `list_expenses` with date range |
| 3 | "What's the price of AAPL stock?" | Agent calls `yahoo_finance` |
| 4 | "Set a budget of 5000 for Food" | Agent calls `set_budget` |

**Say while demoing:**
> "The model doesn't write to the database directly. It proposes a tool call; Python executes it with validation. That's the MCP pattern."

---

### Demo Part 2 — Dashboard (1 min)

- Open Expenses / Dashboard section
- Show the lunch expense appeared
- Point at categories, totals, charts

**Say:**
> "Chat and UI share the same backend tools and MongoDB collections."

---

### Demo Part 3 — Bank Sync / Plaid (2 min)

1. Go to **Bank Sync (Plaid)**
2. Status should show "Plaid ready (sandbox)"
3. Click **Connect Bank**
4. Sandbox login: `user_good` / `pass_good` (First Platypus Bank)
5. Select connection, date range, click **Sync**

**Say:**
> "Standard Plaid Link flow: link token from our API, user consents in Plaid UI, we exchange public token for access token, store it in MongoDB, then sync transactions. We paginate results, skip pending transactions, only import debits as expenses, and deduplicate by transaction ID."

---

### Demo Part 4 — Receipt (optional, 1 min)

- Upload a bill/receipt image or PDF
- Show extracted fields

**Say:**
> "RAG pipeline: embed document, extract structured expense fields, save via the same expense model."

---

## Code Walkthrough (if they ask "show me the code")

Walk in this order — 2–3 minutes:

| File | What to show | One line |
|------|--------------|----------|
| `main.py` | `@mcp.tool()` on `add_expense`, `TOOL_REGISTRY` at bottom | "All finance logic lives as MCP tools" |
| `chat_agent.py` | `TOOL_REGISTRY` import, Groq function calling | "Agent picks tools from registry" |
| `api.py` | `/api/chat`, `/api/bank/plaid/link-token` | "FastAPI bridges UI to tools" |
| `services/bank_plaid.py` | `create_link_token`, `fetch_transactions` | "Plaid client with pagination and error handling" |
| `database.py` | `add_expense`, `sync_bank_transactions`, `expense_exists_by_plaid_id` | "MongoDB ops + dedup" |

---

## Architecture Diagram (for whiteboard)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Browser   │────▶│   FastAPI    │────▶│  Chat Agent     │
│  (HTML/JS)  │     │   api.py     │     │  Groq + tools   │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                       │
                           │              ┌────────▼────────┐
                           │              │  MCP Tools      │
                           └─────────────▶│  main.py (40+)  │
                                          └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    ▼                              ▼                              ▼
              ┌──────────┐                  ┌──────────┐                  ┌──────────┐
              │ MongoDB  │                  │  Plaid   │                  │  Gmail   │
              └──────────┘                  └──────────┘                  └──────────┘
```

---

## Interview Q&A — Short Answers

**Why MCP?**  
Standard way to expose backend capabilities to AI agents. Tools are discoverable and callable — same pattern as Claude Desktop / Cursor.

**Why Groq?**  
Fast inference and reliable function-calling for responsive chat.

**Why MongoDB?**  
Flexible schema for expenses, metadata (Plaid IDs, Gmail IDs), user settings, audit logs.

**How do you stop the AI from hallucinating expenses?**  
Model only proposes tool calls. Writes go through `add_expense()` in Python with typed params and auth.

**How does auth work?**  
Every tool accepts `api_key` or `session_id`. `auth.authenticate_user()` resolves `user_id`. Data queries always filter by `user_id`.

**How does Plaid work in your app?**  
1. `GET /api/bank/plaid/link-token` → Plaid Link opens in browser  
2. User links bank → `public_token` sent to `POST /api/bank/plaid/exchange`  
3. Access token stored in `bank_connections` collection  
4. Sync calls `/transactions/get` with pagination, maps to expenses  

**What happens on re-sync?**  
`expense_exists_by_plaid_id()` skips already-imported transactions.

**Biggest challenge?**  
Getting the agent to call the right tools for multi-intent messages — solved with system prompt, parallel tool calls, and explicit date context.

**What would you do for production?**  
Plaid webhooks, encrypt tokens at rest, `/transactions/sync` with cursors, HTTPS, secrets manager, monitoring.

**How is this different from a ChatGPT wrapper?**  
Persistent user data, 40+ domain-specific tools, bank/Gmail/receipt integrations, audit logs, and a full web UI — not just a chat box.

---

## One-Liners to Drop Naturally

1. "I separated HTTP, AI orchestration, and business logic into distinct layers."
2. "The AI proposes actions; Python executes them — that's intentional guardrails."
3. "MCP makes the system extensible — add a tool, the agent can use it."
4. "Every MongoDB document is scoped by `user_id` for isolation."
5. "Plaid follows the standard Link → exchange → sync flow."

---

## Pre-Interview Checklist

- [ ] MongoDB running
- [ ] `python run_app.py` works at http://127.0.0.1:8080
- [ ] `GROQ_API_KEY` set in `.env`
- [ ] Plaid sandbox keys set (for bank demo)
- [ ] Practiced 4 chat commands once
- [ ] Practiced Plaid connect + sync once
- [ ] Read the 2-minute script out loud twice
- [ ] Know where `main.py`, `chat_agent.py`, `api.py`, `bank_plaid.py` are

---

## 30-Second Version (if time is very short)

> TrackExpensio is an AI finance assistant. Users chat in natural language; a Groq agent calls MCP tools to add expenses, sync banks via Plaid, and fetch stock data. FastAPI + MongoDB backend, full web UI. Let me show the chat and bank sync.

Then demo: one chat command + one Plaid sync.
