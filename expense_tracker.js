// TrackExpensio Web Controller
let sessionId = localStorage.getItem('sessionId') || '';
let apiKey = localStorage.getItem('apiKey') || ''; // For registered users
let currentSection = 'dashboard';
let isGoogleLinked = localStorage.getItem('googleLinked') === 'true';
let chatHistory = [];
let sessionLinked = false;
const BASE_CURRENCY_KEY = 'baseCurrency';
const FEATURE_PODS = [
    { icon: 'fas fa-piggy-bank', title: 'Budget Radar', description: 'Monthly/category budgets with live AI alerts & nudges.' },
    { icon: 'fas fa-sync-alt', title: 'Recurring Hub', description: 'Subscriptions, rent, EMI auto-posted with cron-style MCP agents.' },
    { icon: 'fas fa-brain', title: 'AI Categorizer', description: 'Merchant + note embeddings feed a zero-shot classifier.' },
    { icon: 'fas fa-dollar-sign', title: 'Income Streams', description: 'Salary, freelancing, dividends tracked separately.' },
    { icon: 'fas fa-bullseye', title: 'Savings Goals', description: 'Daily savings recommendations plus streak tracking.' },
    { icon: 'fas fa-credit-card', title: 'Debt Coach', description: 'Avalanche/snowball payoff plans with autopay reminders.' },
    { icon: 'fas fa-chart-bar', title: 'Investment Lens', description: 'Mark-to-market view with gain/loss sentiment.' },
    { icon: 'fas fa-file-alt', title: 'Reports Studio', description: 'FastAPI PDF generator + sharable dashboards.' },
    { icon: 'fas fa-calculator', title: 'Tax Copilot', description: 'Deduction tagging & FY wise liability estimator.' },
    { icon: 'fas fa-star', title: 'Credit Monitor', description: 'Score snapshots, factors & personalized fixes.' },
    { icon: 'fas fa-bell', title: 'Bill Radar', description: 'Upcoming dues timeline with push + email alerts.' },
    { icon: 'fas fa-globe', title: 'FX Wallet', description: 'Multi-currency logging with live FX conversion.' },
    { icon: 'fas fa-users', title: 'Access Control', description: 'Role-based sharing (view/edit) for family & CA.' },
    { icon: 'fas fa-university', title: 'Bank Sync', description: 'Plaid-compatible fetcher pipes data into MCP tools.' },
];

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await initializeApp();
    setupEventListeners();
    initFeatureMarquee();
    showSection('dashboard');
    
    // Check for Google login success
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('google_login') === 'success') {
        const sessionParam = urlParams.get('session_id');
        if (sessionParam) {
            sessionId = sessionParam;
            localStorage.setItem('sessionId', sessionId);
        }
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
        // Refresh session and show success
        await refreshSessionStatus(true);
        await loadUserProfile();
        showMessage('✅ Successfully logged in with Google!', 'success');
    } else {
        await refreshSessionStatus();
        await loadUserProfile();
    }
    
    if (ensureAuthenticated(false)) {
        loadDashboardData();
    }
});

async function initializeApp() {
    // Auto-create guest session if not exists
    if (!sessionId && !apiKey) {
        try {
            const response = await fetch('/api/session/create', { method: 'POST' });
            const data = await response.json();
            if (data.session_id) {
                sessionId = data.session_id;
                localStorage.setItem('sessionId', sessionId);
                console.log('✅ Guest session created');
            }
        } catch (error) {
            console.error('Failed to create session:', error);
            showToast('Failed to initialize. Please refresh the page.', 'error');
        }
    }
    if (!localStorage.getItem(BASE_CURRENCY_KEY)) {
        localStorage.setItem(BASE_CURRENCY_KEY, 'INR');
    }
    
    // Dashboard section toggled after session status resolves
}

function setupEventListeners() {
    // Sidebar navigation
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.dataset.section;
            showSection(section);
        });
    });
    
    // Gmail Connect button
    document.getElementById('gmailConnectBtn')?.addEventListener('click', initiateGmailConnect);
    
    // Chat input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
    
    // Close modals on outside click
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            closeModal(e.target.id);
        }
    });
    
    // Document upload (dashboard)
    document.getElementById('docUpload')?.addEventListener('change', (event) => {
        const file = event.target.files?.[0];
        if (file) {
            uploadDocument(file);
        }
    });
    
    document.getElementById('chatDocButton')?.addEventListener('click', () => {
        document.getElementById('chatDocInput')?.click();
    });
    document.getElementById('chatDocInput')?.addEventListener('change', handleChatDocUpload);
    document.getElementById('marketForm')?.addEventListener('submit', lookupTicker);
}

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    const section = document.getElementById(sectionName);
    if (section) {
        section.classList.add('active');
        currentSection = sectionName;
    }
    
    // Update sidebar active state
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.section === sectionName) {
            btn.classList.add('active');
        }
    });
    
    // Load section data
    loadSectionData(sectionName);
}

function initFeatureMarquee() {
    const grid = document.getElementById('featureGrid');
    if (!grid) return;
    grid.innerHTML = FEATURE_PODS.map(pod => `
        <article class="feature-card">
            <header><i class="${pod.icon}"></i>${pod.title}</header>
            <p>${pod.description}</p>
        </article>
    `).join('');
}

async function initiateGmailConnect() {
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const param = apiKey ? `api_key=${encodeURIComponent(apiKey)}` : `session_id=${encodeURIComponent(sessionId)}`;
    const popup = window.open(`/auth/google/login?${param}`, 'googleLogin', 'width=600,height=700');
    
    // Listen for login success message from popup
    window.addEventListener('message', async function(event) {
        if (event.data && event.data.type === 'google_login_success') {
            const newSessionId = event.data.session_id;
            if (newSessionId) {
                sessionId = newSessionId;
                localStorage.setItem('sessionId', sessionId);
            }
            await refreshSessionStatus(true);
            await loadUserProfile();
            showMessage('✅ Successfully logged in with Google!', 'success');
        }
    });
    
    showMessage('Opening Gmail login...', 'info');
}

async function uploadDocument(file) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const form = new FormData();
    form.append('file', file);
    if (apiKey) form.append('api_key', apiKey);
    if (sessionId) form.append('session_id', sessionId);
    try {
        const res = await fetch('/api/doc-expense', { method: 'POST', body: form });
        if (!res.ok) throw new Error('Failed to parse document');
        const data = await res.json();
        showMessage(data.message || 'Bill parsed successfully', 'success');
        loadExpenses();
        loadDashboardData();
    } catch (error) {
        console.error(error);
        showMessage('Document extraction failed', 'error');
    }
}

async function handleChatDocUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Ask user for prompt/instruction if they want to add one
    const userPrompt = window.prompt('Add an instruction or prompt for this document (optional).\n\nFor example: "Add this as a business expense" or "Categorize this as Travel"\n\nLeave empty to auto-process:', '');
    
    addChatMessage(`📎 Uploaded ${file.name}${userPrompt ? `\n💬 Instruction: ${userPrompt}` : ''}`, 'user');
    chatHistory.push({ role: 'user', content: `Uploaded document: ${file.name}${userPrompt ? ` with instruction: ${userPrompt}` : ''}` });
    await uploadDocumentFromChat(file, userPrompt);
    event.target.value = '';
}

async function uploadDocumentFromChat(file, userPrompt = null) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const placeholder = addChatMessage('Processing your document...', 'bot');
    const form = new FormData();
    form.append('file', file);
    if (apiKey) form.append('api_key', apiKey);
    if (sessionId) form.append('session_id', sessionId);
    if (userPrompt && userPrompt.trim()) {
        form.append('prompt', userPrompt.trim());
    }
    try {
        const res = await fetch('/api/chat/upload-doc', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to parse document');
        const summaryText = data.message || formatDocSummary(data.summary);
        updateChatMessage(placeholder, summaryText);
        chatHistory.push({ role: 'assistant', content: summaryText });
        if (data.summary?.raw_text) {
            chatHistory.push({
                role: 'system',
                content: `Document transcript:\n${data.summary.raw_text}`,
            });
        }
        loadExpenses();
        loadDashboardData();
    } catch (error) {
        console.error(error);
        updateChatMessage(placeholder, 'Document processing failed. Please try again.');
        showMessage('Document extraction failed', 'error');
    }
}

function formatDocSummary(summary = {}) {
    const merchant = summary.merchant || 'Unknown merchant';
    const rawAmount = parseFloat(summary.amount || 0);
    const amount = formatCurrency(Number.isNaN(rawAmount) ? 0 : rawAmount, summary.currency);
    const category = summary.category || 'Other';
    const date = formatDisplayDate(summary.date);
    return `🧾 Processed ${merchant} receipt\nAmount: ${amount}\nCategory: ${category}\nDate: ${date}\nExpense has been saved automatically.`;
}

function loadSectionData(section) {
    if (!ensureAuthenticated(false)) {
        return;
    }
    const auth = getAuthPayload();
    if (!auth) {
        return;
    }
    
    switch(section) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'expenses':
            loadExpenses();
            break;
        case 'income':
            loadIncome();
            break;
        case 'budget':
            loadBudgetStatus();
            break;
        case 'bills':
            loadUpcomingBills();
            break;
        case 'savings':
            loadSavingsGoals();
            break;
        case 'investments':
            loadInvestments();
            break;
        case 'debt':
            loadDebts();
            break;
        case 'chat':
            // Chat is ready
            break;
        default:
            showMessage(`Loading ${section}...`, 'info');
    }
}

// Helper to get auth payload (session_id or api_key)
function getAuthPayload() {
    if (apiKey) return { api_key: apiKey };
    if (sessionId) return { session_id: sessionId };
    return null;
}

async function loadDashboardData() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        // Get current month dates
        const today = new Date();
        const startDate = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
        const endDate = today.toISOString().split('T')[0];
        
        // Load expenses summary
        const expenseSummary = await callMCPTool('summarize', {
            ...auth,
            start_date: startDate,
            end_date: endDate
        });
        
        // Load income summary
        const incomeSummary = await callMCPTool('get_income_summary', {
            ...auth,
            start_date: startDate,
            end_date: endDate
        });
        
        // Calculate totals
        const totalExpense = expenseSummary.reduce((sum, item) => sum + parseFloat(item.total || 0), 0);
        const totalIncome = incomeSummary.total || 0;
        
        // Update UI
        document.getElementById('totalExpense').textContent = formatCurrency(totalExpense);
        document.getElementById('totalIncome').textContent = formatCurrency(totalIncome);
        
        // Load savings goals
        const savingsGoals = await callMCPTool('track_savings_progress', auth);
        
        const totalSavings = savingsGoals.goals?.reduce((sum, goal) => sum + parseFloat(goal.current_amount || 0), 0) || 0;
        document.getElementById('totalSavings').textContent = formatCurrency(totalSavings);
        
        // Calculate net worth (simplified)
        const netWorth = totalIncome - totalExpense + totalSavings;
        document.getElementById('netWorth').textContent = formatCurrency(netWorth);
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showMessage('Error loading dashboard data', 'error');
    }
}

async function loadExpenses() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        // Load expenses from last 6 months to show recent expenses
        const today = new Date();
        const sixMonthsAgo = new Date(today.getFullYear(), today.getMonth() - 6, 1);
        const startDate = sixMonthsAgo.toISOString().split('T')[0];
        const endDate = today.toISOString().split('T')[0];
        
        const result = await callMCPTool('list_expenses', {
            ...auth,
            start_date: startDate,
            end_date: endDate
        });
        
        // Handle both direct array and wrapped response
        const expenses = Array.isArray(result) ? result : (result?.data || result?.expenses || []);
        
        displayExpenses(expenses);
    } catch (error) {
        console.error('Error loading expenses:', error);
        showMessage('Error loading expenses', 'error');
    }
}

function displayExpenses(expenses) {
    const container = document.getElementById('expenseList');
    if (!container) return;
    
    // Ensure expenses is an array
    if (!expenses) {
        expenses = [];
    } else if (!Array.isArray(expenses)) {
        // If it's wrapped in a response object, extract the array
        expenses = expenses.data || expenses.expenses || [];
    }
    
    if (expenses.length === 0) {
        container.innerHTML = '<p class="empty-state">No expenses found</p>';
        return;
    }
    
    // Sort by date (newest first)
    expenses.sort((a, b) => {
        const dateA = new Date(a.date || 0);
        const dateB = new Date(b.date || 0);
        return dateB - dateA;
    });
    
    container.innerHTML = expenses.map(expense => {
        const amount = expense.amount || 0;
        const category = expense.category || 'Other';
        const date = expense.date || '';
        const merchant = expense.merchant || '—';
        const note = expense.note || '';
        const currency = expense.metadata?.currency;
        
        return `
        <div class="expense-item">
            <div class="expense-body">
                <div class="expense-header">
                    <span class="expense-category">${category}</span>
                    <span class="expense-amount">${formatCurrency(amount, currency)}</span>
                </div>
                <div class="expense-merchant">${merchant}</div>
                <div class="expense-note">${note}</div>
            </div>
            <div class="expense-meta">
                <span class="expense-date">${formatDisplayDate(date)}</span>
                ${expense.metadata?.source ? `<span class="expense-tag">${expense.metadata.source}</span>` : ''}
            </div>
        </div>
        `;
    }).join('');
}

async function loadIncome() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const today = new Date();
        const startDate = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
        const endDate = today.toISOString().split('T')[0];
        
        const income = await callMCPTool('list_income', {
            ...auth,
            start_date: startDate,
            end_date: endDate
        });
        
        displayIncome(income.income || []);
    } catch (error) {
        console.error('Error loading income:', error);
        showMessage('Error loading income', 'error');
    }
}

function displayIncome(incomeList) {
    const container = document.getElementById('incomeList');
    if (!container) return;
    
    if (!incomeList || incomeList.length === 0) {
        container.innerHTML = '<p class="empty-state">No income recorded</p>';
        return;
    }
    
    container.innerHTML = incomeList.map(income => `
        <div class="income-item">
            <div class="income-icon">
                <i class="fas fa-dollar-sign"></i>
            </div>
            <div class="income-details">
                <h3>${income.source || 'Income'}</h3>
                <p>${income.category || ''}</p>
                <span class="income-date">${income.date}</span>
            </div>
            <div class="income-amount">
                <span>${formatCurrency(income.amount)}</span>
            </div>
        </div>
    `).join('');
}

async function loadBudgetStatus() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const status = await callMCPTool('check_budget_status', auth);
        
        displayBudgetStatus(status);
    } catch (error) {
        console.error('Error loading budget status:', error);
        showMessage('Error loading budget status', 'error');
    }
}

function displayBudgetStatus(status) {
    const container = document.getElementById('budgetStatus');
    if (!container) return;
    
    if (!status.budget_status || status.budget_status.length === 0) {
        container.innerHTML = '<p class="empty-state">No budgets set. Click "Set Budget" to create one.</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="budget-summary">
            <h3>Budget Summary for ${status.month}</h3>
            <div class="budget-stats">
                <div class="budget-stat">
                    <span class="label">Total Budget:</span>
                    <span class="value">${formatCurrency(status.summary.total_budget)}</span>
                </div>
                <div class="budget-stat">
                    <span class="label">Total Spent:</span>
                    <span class="value">${formatCurrency(status.summary.total_spent)}</span>
                </div>
                <div class="budget-stat">
                    <span class="label">Remaining:</span>
                    <span class="value">${formatCurrency(status.summary.total_remaining)}</span>
                </div>
            </div>
        </div>
        <div class="budget-list">
            ${status.budget_status.map(budget => `
                <div class="budget-item">
                    <div class="budget-category">${budget.category}</div>
                    <div class="budget-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${Math.min(budget.percentage_used, 100)}%"></div>
                        </div>
                        <div class="budget-amounts">
                            <span>${formatCurrency(budget.spent)} / ${formatCurrency(budget.budget)}</span>
                            <span class="remaining">${formatCurrency(budget.remaining)} left</span>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        ${status.alerts && status.alerts.length > 0 ? `
            <div class="budget-alerts">
                <h4>Alerts</h4>
                ${status.alerts.map(alert => `
                    <div class="alert ${alert.type}">${alert.message}</div>
                `).join('')}
            </div>
        ` : ''}
        ${status.ai_tips && status.ai_tips.length > 0 ? `
            <div class="budget-tips">
                <h4>AI Tips</h4>
                ${status.ai_tips.map(tip => `
                    <div class="tip">${tip}</div>
                `).join('')}
            </div>
        ` : ''}
    `;
}

// Chat Functions
function sendChatMessage() {
    if (!ensureAuthenticated()) return;
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Initializing session...', 'info');
        setTimeout(() => sendChatMessage(), 1000);
        return;
    }
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    
    // Process with AI
    processChatMessage(message);
}

function sendExample(text) {
    document.getElementById('chatInput').value = text;
    sendChatMessage();
}

function addChatMessage(text, type) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${type === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <p>${text}</p>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageDiv;
}

async function processChatMessage(message) {
    const placeholder = addChatMessage('Processing...', 'bot');
    const auth = getAuthPayload();
    chatHistory.push({ role: 'user', content: message });
    try {
        const payload = {
            message,
            session_id: sessionId || auth?.session_id || null,
            api_key: apiKey || auth?.api_key || null,
            history: chatHistory
        };
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || 'Chat failed');
        }
        chatHistory.push({ role: 'assistant', content: data.reply });
        updateChatMessage(placeholder, data.reply || 'Done.');
        renderToolResults(placeholder, data.tool_results);
        refreshDataForTools(data.tool_results);
    } catch (error) {
        console.error('Error processing chat:', error);
        updateChatMessage(placeholder, error.message || 'Sorry, something went wrong.');
    }
}

function renderToolResults(messageElement, toolResults = []) {
    // Don't show backend JSON to users - the formatted reply is already shown
    // This function is kept for potential future use but doesn't display anything
    return;
}

function refreshDataForTools(toolResults = []) {
    if (!Array.isArray(toolResults) || toolResults.length === 0) return;
    const names = toolResults.map(r => r.tool);
    
    // Expenses
    if (names.some(t => ['add_expense', 'quick_add_expense', 'delete_expense', 'update_expense', 'document_expense_from_rag'].includes(t))) {
        loadExpenses();
        loadDashboardData();
    }
    
    // Budget
    if (names.some(t => ['set_budget', 'check_budget_status'].includes(t))) {
        loadBudgetStatus();
        loadDashboardData();
    }
    
    // Income
    if (names.some(t => ['add_income', 'list_income', 'update_income', 'delete_income'].includes(t))) {
        loadIncome();
        loadDashboardData();
    }
    
    // Bill Reminders
    if (names.some(t => ['add_bill_reminder', 'get_upcoming_bills', 'update_bill_reminder', 'delete_bill_reminder'].includes(t)) && typeof loadUpcomingBills === 'function') {
        loadUpcomingBills();
    }
    
    // Savings Goals
    if (names.some(t => ['set_savings_goal', 'track_savings_progress', 'update_savings_goal', 'delete_savings_goal'].includes(t))) {
        loadSavingsGoals();
        loadDashboardData();
    }
    
    // Investments
    if (names.some(t => ['add_investment', 'get_investment_portfolio', 'update_investment_value', 'delete_investment'].includes(t))) {
        loadInvestments();
        loadDashboardData();
    }
    
    // Debts
    if (names.some(t => ['add_debt', 'get_debt_summary', 'record_debt_payment', 'delete_debt'].includes(t))) {
        loadDebts();
        loadDashboardData();
    }
    
    // Stock queries (yahoo_finance, get_stock_returns, etc.) - no UI refresh needed, just display in chat
    
    // Currency
    if (names.includes('set_base_currency')) {
        const result = toolResults.find(r => r.tool === 'set_base_currency');
        const currency = result?.result?.base_currency;
        if (currency) {
            applyBaseCurrency(currency);
        }
    }
}

function updateChatMessage(messageElement, text) {
    if (!messageElement) return;
    const contentDiv = messageElement.querySelector('.message-content');
    if (contentDiv) {
        // Clear all existing content (including any tool results JSON)
        contentDiv.innerHTML = '';
        // Add the new text, preserving line breaks
        const paragraph = document.createElement('p');
        paragraph.innerHTML = text.replace(/\n/g, '<br>');
        contentDiv.appendChild(paragraph);
    }
}

// MCP Tool Caller (simplified - in production, this would call your MCP server)
async function callMCPTool(toolName, params) {
    // This is a placeholder - in production, you'd make an HTTP request to your MCP server
    // or use WebSocket/SSE for real-time communication
    
    try {
        const response = await fetch(`/api/mcp/${toolName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`Error calling MCP tool ${toolName}:`, error);
        throw error;
    }
}

// Utility Functions
function formatCurrency(amount, currencyOverride) {
    const currency = (currencyOverride || getBaseCurrency()).toUpperCase();
    const locale = currency === 'INR' ? 'en-IN' : 'en-US';
    try {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency
        }).format(amount || 0);
    } catch {
        return `${currency} ${(amount || 0).toFixed(2)}`;
    }
}

function getBaseCurrency() {
    return (localStorage.getItem(BASE_CURRENCY_KEY) || 'INR').toUpperCase();
}

function applyBaseCurrency(newCurrency) {
    if (!newCurrency) return;
    localStorage.setItem(BASE_CURRENCY_KEY, newCurrency.toUpperCase());
    showMessage(`Base currency set to ${newCurrency.toUpperCase()}`, 'success');
    loadDashboardData();
    loadExpenses();
}

async function refreshSessionStatus(showToast = false) {
    if (apiKey) {
        sessionLinked = true;
        toggleAuthOverlay(false);
        return true;
    }
    if (!sessionId) return false;
    const wasLinked = sessionLinked;
    try {
        const res = await fetch(`/api/session/status?session_id=${encodeURIComponent(sessionId)}`);
        if (!res.ok) throw new Error('Status check failed');
        const data = await res.json();
        sessionLinked = data.is_linked;
        toggleAuthOverlay(!sessionLinked);
        localStorage.setItem('googleLinked', sessionLinked ? 'true' : 'false');
        if (sessionLinked && !wasLinked) {
            if (showToast) showMessage('Google account linked! You can start using TrackExpensio.', 'success');
            await loadUserProfile();
            loadDashboardData();
            loadExpenses();
            loadIncome();
            loadBudgetStatus();
        } else if (sessionLinked) {
            await loadUserProfile();
        }
    } catch (error) {
        console.warn('Session status check failed', error);
        sessionLinked = false;
        toggleAuthOverlay(true);
        if (showToast) showMessage('Please sync with Google to continue.', 'warning');
    }
    return sessionLinked;
}

let userProfile = { name: null, email: null, picture: null };

async function loadUserProfile() {
    if (!sessionId && !apiKey) return;
    try {
        const params = new URLSearchParams();
        if (sessionId) params.append('session_id', sessionId);
        if (apiKey) params.append('api_key', apiKey);
        const res = await fetch(`/api/user/profile?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to load profile');
        userProfile = await res.json();
        updateUserDisplay();
    } catch (error) {
        console.warn('Failed to load user profile', error);
    }
}

function updateUserDisplay() {
    const gmailBtn = document.getElementById('gmailConnectBtn');
    const userProfileContainer = document.getElementById('userProfileContainer');
    if (!gmailBtn || !userProfileContainer) return;
    
    // Check if user is logged in (either via sessionLinked or has profile)
    const isLoggedIn = sessionLinked || (userProfile && userProfile.email);
    
    if (isLoggedIn && userProfile && userProfile.email) {
        // Hide sync button, show user profile
        gmailBtn.style.display = 'none';
        userProfileContainer.style.display = 'block';
        
        // Update profile display
        const name = userProfile.name || userProfile.email.split('@')[0];
        const email = userProfile.email;
        const picture = userProfile.picture;
        
        // Update avatar
        const avatars = document.querySelectorAll('#userAvatar, #dropdownAvatar');
        const avatarFallbacks = document.querySelectorAll('.user-avatar-fallback, .dropdown-avatar-fallback');
        
        if (picture) {
            avatars.forEach(avatar => {
                avatar.src = picture;
                avatar.style.display = 'block';
            });
            avatarFallbacks.forEach(fallback => {
                fallback.style.display = 'none';
            });
        } else {
            avatars.forEach(avatar => {
                avatar.style.display = 'none';
            });
            avatarFallbacks.forEach(fallback => {
                fallback.style.display = 'flex';
            });
        }
        
        // Update name and email
        document.getElementById('userName').textContent = name;
        document.getElementById('dropdownUserName').textContent = name;
        document.getElementById('dropdownUserEmail').textContent = email;
        
        // Setup dropdown toggle
        setupUserProfileDropdown();
    } else {
        // Show sync button, hide user profile
        gmailBtn.style.display = 'flex';
        userProfileContainer.style.display = 'none';
        
        // Show connect button
        gmailBtn.innerHTML = `
            <i class="fas fa-cloud"></i> 
            <span class="nav-btn-text">Sync with Google</span>
        `;
        gmailBtn.title = 'Save your data permanently with Google';
        gmailBtn.classList.remove('logged-in');
        gmailBtn.style.pointerEvents = 'auto';
        gmailBtn.style.opacity = '1';
    }
}

function setupUserProfileDropdown() {
    const container = document.getElementById('userProfileContainer');
    const trigger = document.getElementById('userProfileTrigger');
    const dropdown = document.getElementById('userProfileDropdown');
    const signOutBtn = document.getElementById('signOutBtn');
    
    if (!container || !trigger || !dropdown) return;
    
    // Toggle dropdown
    trigger.onclick = (e) => {
        e.stopPropagation();
        container.classList.toggle('active');
    };
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) {
            container.classList.remove('active');
        }
    });
    
    // Sign out functionality
    if (signOutBtn) {
        signOutBtn.onclick = async (e) => {
            e.stopPropagation();
            await signOut();
        };
    }
}

async function signOut() {
    try {
        // Clear local storage
        localStorage.removeItem('sessionId');
        localStorage.removeItem('apiKey');
        sessionId = null;
        apiKey = null;
        userProfile = { name: null, email: null, picture: null };
        sessionLinked = false;
        
        // Update UI
        updateUserDisplay();
        
        // Reload page to reset state
        window.location.href = '/';
    } catch (error) {
        console.error('Sign out error:', error);
        showMessage('Error signing out. Please refresh the page.', 'error');
    }
}

function toggleAuthOverlay(show) {
    const overlay = document.getElementById('authOverlay');
    if (!overlay) return;
    if (show && !apiKey) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

function ensureAuthenticated(showToast = true) {
    if (apiKey || sessionLinked) {
        toggleAuthOverlay(false);
        return true;
    }
    if (showToast) {
        showMessage('Please sync with Google before using TrackExpensio.', 'warning');
    }
    toggleAuthOverlay(true);
    return false;
}

function formatDisplayDate(dateStr) {
    if (!dateStr) return '';
    const dateObj = new Date(dateStr);
    if (Number.isNaN(dateObj.getTime())) {
        return dateStr;
    }
    return dateObj.toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric'
    });
}

async function lookupTicker(event) {
    event?.preventDefault();
    const symbolInput = document.getElementById('marketSymbol');
    const symbol = symbolInput?.value.trim().toUpperCase();
    if (!symbol) {
        showMessage('Enter a ticker symbol', 'warning');
        return;
    }
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const container = document.getElementById('marketResult');
    if (container) container.textContent = 'Fetching market data...';
    try {
        const response = await callMCPTool('yahoo_finance', { ...auth, symbol });
        if (!response) {
            throw new Error('No response from server');
        }
        if (response.status === 'error') {
            if (container) container.innerHTML = `<div class="error-message">${response.message || 'Unable to fetch ticker data'}</div>`;
            showMessage(response.message || 'Failed to fetch market data', 'error');
            return;
        }
        const snapshot = response.data || response;
        if (!snapshot || !snapshot.price) {
            if (container) container.innerHTML = `<div class="error-message">No data available for ${symbol}. Please check the ticker symbol.</div>`;
            showMessage(`No data found for ${symbol}`, 'error');
            return;
        }
        if (container) {
            container.innerHTML = renderMarketResult(snapshot);
        }
        showMessage(`✅ Fetched data for ${symbol}`, 'success');
    } catch (error) {
        console.error('Yahoo Finance error:', error);
        const errorMsg = error.message || 'Unable to fetch ticker. Please try again.';
        if (container) container.innerHTML = `<div class="error-message">${errorMsg}</div>`;
        showMessage(errorMsg, 'error');
    }
}

function renderMarketResult(snapshot = {}) {
    if (!snapshot.price) {
        return '<div class="error-message">No data returned for that symbol.</div>';
    }
    const range = snapshot.fifty_two_week_range || {};
    const dayRange = snapshot.day_range || {};
    const returns = snapshot.returns || {};
    const formatNumber = (num) => {
        if (num === null || num === undefined) return '-';
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toLocaleString('en-IN', { maximumFractionDigits: 2 });
    };
    const formatReturn = (ret) => {
        if (ret === null || ret === undefined) return '-';
        const sign = ret >= 0 ? '+' : '';
        const color = ret >= 0 ? '#10b981' : '#ef4444';
        return `<span style="color: ${color}">${sign}${ret.toFixed(2)}%</span>`;
    };
    return `
        <div class="market-result-card">
            <div class="market-header">
                <h3>${snapshot.name || snapshot.symbol || 'N/A'}</h3>
                <div class="market-ticker">${snapshot.symbol || ''} · ${snapshot.currency || 'USD'}</div>
            </div>
            <div class="market-price">
                <div class="price-main">${formatNumber(snapshot.price)}</div>
                ${snapshot.previous_close ? `<div class="price-change">Prev Close: ${formatNumber(snapshot.previous_close)}</div>` : ''}
            </div>
            <div class="market-details">
                <div class="detail-row">
                    <span>Day Range:</span>
                    <span>${formatNumber(dayRange.low)} – ${formatNumber(dayRange.high)}</span>
                </div>
                <div class="detail-row">
                    <span>52W Range:</span>
                    <span>${formatNumber(range.low)} – ${formatNumber(range.high)}</span>
                </div>
                ${snapshot.market_cap ? `<div class="detail-row"><span>Market Cap:</span><span>${formatNumber(snapshot.market_cap)}</span></div>` : ''}
                ${snapshot.volume ? `<div class="detail-row"><span>Volume:</span><span>${formatNumber(snapshot.volume)}</span></div>` : ''}
                ${snapshot.pe_ratio ? `<div class="detail-row"><span>P/E Ratio:</span><span>${formatNumber(snapshot.pe_ratio)}</span></div>` : ''}
                ${snapshot.dividend_yield ? `<div class="detail-row"><span>Dividend Yield:</span><span>${snapshot.dividend_yield.toFixed(2)}%</span></div>` : ''}
            </div>
            ${Object.keys(returns).length > 0 ? `
            <div class="market-returns">
                <h4>Returns</h4>
                <div class="returns-grid">
                    ${returns['1d'] !== null && returns['1d'] !== undefined ? `<div><span>1D:</span> ${formatReturn(returns['1d'])}</div>` : ''}
                    ${returns['1w'] !== null && returns['1w'] !== undefined ? `<div><span>1W:</span> ${formatReturn(returns['1w'])}</div>` : ''}
                    ${returns['1m'] !== null && returns['1m'] !== undefined ? `<div><span>1M:</span> ${formatReturn(returns['1m'])}</div>` : ''}
                    ${returns['3m'] !== null && returns['3m'] !== undefined ? `<div><span>3M:</span> ${formatReturn(returns['3m'])}</div>` : ''}
                    ${returns['6m'] !== null && returns['6m'] !== undefined ? `<div><span>6M:</span> ${formatReturn(returns['6m'])}</div>` : ''}
                    ${returns['1y'] !== null && returns['1y'] !== undefined ? `<div><span>1Y:</span> ${formatReturn(returns['1y'])}</div>` : ''}
                </div>
            </div>
            ` : ''}
            ${snapshot.sector ? `<div class="market-meta"><span>Sector:</span> ${snapshot.sector}</div>` : ''}
            ${snapshot.industry ? `<div class="market-meta"><span>Industry:</span> ${snapshot.industry}</div>` : ''}
            ${snapshot.recommendation ? `<div class="market-meta"><span>Recommendation:</span> ${snapshot.recommendation}</div>` : ''}
        </div>
    `;
}

async function loadUpcomingBills() {
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const data = await callMCPTool('get_upcoming_bills', auth);
        renderBillList(data);
    } catch (error) {
        console.error('Error loading bills:', error);
        showMessage('Error loading bill reminders', 'error');
    }
}

function renderBillList(data = {}) {
    const container = document.getElementById('billList');
    if (!container) return;
    const bills = data.upcoming_bills || data.bills || [];
    if (!bills.length) {
        container.innerHTML = '<p class="empty-state">No bills due soon. Ask the assistant to set one!</p>';
        return;
    }
    container.innerHTML = bills.map(bill => `
        <div class="bill-item">
            <div class="bill-info">
                <h3>${bill.bill_name || 'Bill'}</h3>
                <div class="bill-meta">Due in ${bill.days_until_due ?? '?'} day(s) · ${formatDisplayDate(bill.due_date)}</div>
            </div>
            <div class="bill-amount">
                ${formatCurrency(bill.amount)}
                <span>${bill.due_date}</span>
            </div>
        </div>
    `).join('');
}

// Load Savings Goals
async function loadSavingsGoals() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const data = await callMCPTool('track_savings_progress', auth);
        renderSavingsGoals(data);
    } catch (error) {
        console.error('Error loading savings goals:', error);
        // Don't show error message if function doesn't exist yet
    }
}

function renderSavingsGoals(data = {}) {
    const container = document.querySelector('#savings .savings-list, #savings .content-section');
    if (!container) return;
    
    const goals = data.goals || data.savings_goals || [];
    if (!goals.length) {
        const existingEmpty = container.querySelector('.empty-state');
        if (!existingEmpty) {
            container.innerHTML = '<p class="empty-state">No savings goals yet. Ask the assistant to set one!</p>';
        }
        return;
    }
    
    // Create list if it doesn't exist
    let list = container.querySelector('.savings-list');
    if (!list) {
        list = document.createElement('div');
        list.className = 'savings-list';
        container.innerHTML = '';
        container.appendChild(list);
    }
    
    list.innerHTML = goals.map(goal => {
        const target = goal.target_amount || 0;
        const current = goal.current_amount || 0;
        const progress = target > 0 ? (current / target) * 100 : 0;
        const remaining = Math.max(0, target - current);
        
        return `
        <div class="savings-goal-item">
            <div class="goal-header">
                <h3>${goal.goal_name || 'Savings Goal'}</h3>
                <span class="goal-amount">${formatCurrency(target)}</span>
            </div>
            <div class="goal-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.min(100, progress)}%"></div>
                </div>
                <div class="progress-text">
                    ${formatCurrency(current)} / ${formatCurrency(target)} (${progress.toFixed(1)}%)
                </div>
            </div>
            <div class="goal-meta">
                Remaining: ${formatCurrency(remaining)} · Target: ${formatDisplayDate(goal.target_date || '')}
            </div>
        </div>
        `;
    }).join('');
}

// Load Investments
async function loadInvestments() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const data = await callMCPTool('get_investment_portfolio', auth);
        renderInvestments(data);
    } catch (error) {
        console.error('Error loading investments:', error);
    }
}

function renderInvestments(data = {}) {
    const container = document.querySelector('#investments .investments-list, #investments .content-section');
    if (!container) return;
    
    // get_investment_portfolio returns portfolio data with investments array
    const investments = data.portfolio?.investments || data.investments || [];
    if (!investments.length) {
        const existingEmpty = container.querySelector('.empty-state');
        if (!existingEmpty) {
            container.innerHTML = '<p class="empty-state">No investments recorded yet. Ask the assistant to add one!</p>';
        }
        return;
    }
    
    let list = container.querySelector('.investments-list');
    if (!list) {
        list = document.createElement('div');
        list.className = 'investments-list';
        container.innerHTML = '';
        container.appendChild(list);
    }
    
    list.innerHTML = investments.map(inv => `
        <div class="investment-item">
            <div class="investment-header">
                <h3>${inv.investment_name || inv.name || 'Investment'}</h3>
                <span class="investment-value">${formatCurrency(inv.current_value || inv.amount || 0)}</span>
            </div>
            <div class="investment-meta">
                Type: ${inv.investment_type || 'Stock'} · 
                Purchase Amount: ${formatCurrency(inv.amount || 0)} · 
                Purchase Date: ${formatDisplayDate(inv.purchase_date || '')}
            </div>
        </div>
    `).join('');
}

// Load Debts
async function loadDebts() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const data = await callMCPTool('get_debt_summary', auth);
        renderDebts(data);
    } catch (error) {
        console.error('Error loading debts:', error);
    }
}

function renderDebts(data = {}) {
    const container = document.querySelector('#debt .debts-list, #debt .content-section');
    if (!container) return;
    
    // get_debt_summary returns debts array
    const debts = data.debts || [];
    if (!debts.length) {
        const existingEmpty = container.querySelector('.empty-state');
        if (!existingEmpty) {
            container.innerHTML = '<p class="empty-state">No debts recorded yet.</p>';
        }
        return;
    }
    
    let list = container.querySelector('.debts-list');
    if (!list) {
        list = document.createElement('div');
        list.className = 'debts-list';
        container.innerHTML = '';
        container.appendChild(list);
    }
    
    list.innerHTML = debts.map(debt => {
        const total = debt.total_amount || 0;
        const paid = debt.paid_amount || 0;
        const remaining = debt.remaining || (total - paid);
        const progress = total > 0 ? (paid / total) * 100 : 0;
        
        return `
        <div class="debt-item">
            <div class="debt-header">
                <h3>${debt.creditor || debt.creditor_name || 'Debt'}</h3>
                <span class="debt-amount">${formatCurrency(remaining)} remaining</span>
            </div>
            <div class="debt-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.min(100, progress)}%"></div>
                </div>
                <div class="progress-text">
                    ${formatCurrency(paid)} / ${formatCurrency(total)} paid (${progress.toFixed(1)}%)
                </div>
            </div>
            <div class="debt-meta">
                Interest Rate: ${debt.interest_rate || 0}% · Type: ${debt.debt_type || 'Credit Card'}
            </div>
        </div>
        `;
    }).join('');
}

function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// API key functions removed - using guest sessions now

function showAddExpenseModal() {
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Initializing...', 'info');
        return;
    }
    showModal('addExpenseModal');
}

function showAddIncomeModal() {
    // Similar to showAddExpenseModal
    showMessage('Add Income modal - Coming soon', 'info');
}

function showSetBudgetModal() {
    // Similar to showAddExpenseModal
    showMessage('Set Budget modal - Coming soon', 'info');
}

async function addExpense(event) {
    event.preventDefault();
    
    if (!ensureAuthenticated()) {
        return;
    }
    
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required', 'warning');
        return;
    }
    
    const formData = {
        ...auth,
        date: document.getElementById('expenseDate').value,
        amount: document.getElementById('expenseAmount').value,
        category: document.getElementById('expenseCategory').value,
        merchant: document.getElementById('expenseMerchant').value,
        note: document.getElementById('expenseNote').value
    };
    
    try {
        const result = await callMCPTool('add_expense', formData);
        showMessage('Expense added successfully!', 'success');
        closeModal('addExpenseModal');
        document.getElementById('addExpenseForm').reset();
        loadExpenses();
        loadDashboardData();
    } catch (error) {
        showMessage('Error adding expense', 'error');
    }
}

function showMessage(message, type = 'info') {
    // Create a toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

