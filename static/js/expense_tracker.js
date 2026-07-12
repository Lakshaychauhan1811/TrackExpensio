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
            showMessage('Failed to initialize. Please refresh the page.', 'error');
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
    document.querySelectorAll('.market-form').forEach(form => {
        form.addEventListener('submit', lookupTicker);
    });
    setupQuickActions();
    setupGmailMessageListener();
    setupDocumentDropzone();
    
    // Voice recording
    const micButton = document.getElementById('chatMicButton');
    if (micButton) {
        micButton.addEventListener('click', toggleVoiceRecording);
    }

    document.getElementById('helpBtn')?.addEventListener('click', showHelpModal);
}

function openQuickAction(action) {
    if (!ensureAuthenticated()) return;
    switch (action) {
        case 'expense':
            showAddExpenseModal();
            break;
        case 'income':
            showAddIncomeModal();
            break;
        case 'budget':
            showSetBudgetModal();
            break;
        case 'chat':
            showSection('chat');
            document.getElementById('chatInput')?.focus();
            break;
        default:
            break;
    }
}

function setupQuickActions() {
    document.querySelectorAll('[data-quick-action]').forEach(btn => {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            openQuickAction(btn.dataset.quickAction);
        });
    });
    document.getElementById('dashboardGmailBtn')?.addEventListener('click', (event) => {
        event.preventDefault();
        initiateGmailConnect();
    });
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

let gmailMessageListenerReady = false;

function setupGmailMessageListener() {
    if (gmailMessageListenerReady) return;
    gmailMessageListenerReady = true;
    window.addEventListener('message', async (event) => {
        if (!event.data || event.data.type !== 'google_login_success') return;
        const newSessionId = event.data.session_id;
        if (newSessionId) {
            sessionId = newSessionId;
            localStorage.setItem('sessionId', sessionId);
        }
        await refreshSessionStatus(true);
        await loadUserProfile();
        loadGmailSyncStatus();
        loadDashboardData();
        showMessage('Gmail connected — you can import bills from the Expenses tab.', 'success');
    });
}

async function initiateGmailConnect() {
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const param = apiKey ? `api_key=${encodeURIComponent(apiKey)}` : `session_id=${encodeURIComponent(sessionId)}`;
    const popup = window.open(`/auth/google/login?${param}`, 'googleLogin', 'width=600,height=700');
    if (!popup) {
        showMessage('Popup blocked — allow popups for Google sign-in.', 'warning');
        return;
    }
    showMessage('Opening Google sign-in…', 'info');
}

function setupDocumentDropzone() {
    const dropzone = document.querySelector('.file-drop.tall');
    const input = document.getElementById('docUpload');
    if (!dropzone || !input) return;

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const file = e.dataTransfer?.files?.[0];
        if (file) uploadDocument(file);
    });

    // Clicking the label opens the native file picker (browser default).
    // This listener handles the case where the user picks a file that way,
    // since only drag/drop was wired to uploadDocument() before.
    input.addEventListener('change', () => {
        const file = input.files?.[0];
        if (file) uploadDocument(file);
    });
}

async function uploadDocument(file) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }
    const allowed = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (file.type && !allowed.includes(file.type)) {
        showMessage('Upload a PDF, PNG, or JPEG file.', 'warning');
        return;
    }
    const form = new FormData();
    form.append('file', file);
    if (apiKey) form.append('api_key', apiKey);
    if (sessionId) form.append('session_id', sessionId);
    showMessage(`Processing ${file.name}…`, 'info');
    try {
        const res = await fetch('/api/doc-expense', { method: 'POST', body: form });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || data.message || 'Failed to parse document');
        const extracted = data.extracted || {};
        const merchant = extracted.merchant || 'Unknown';
        const amount = extracted.amount != null ? formatCurrency(extracted.amount, extracted.currency) : '';
        showMessage(data.message || `Bill parsed: ${merchant} ${amount}`, 'success');
        const docInput = document.getElementById('docUpload');
        if (docInput) docInput.value = '';
        loadExpenses();
        loadDashboardData();
    } catch (error) {
        console.error(error);
        showMessage(error.message || 'Document extraction failed', 'error');
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
            loadGmailSyncStatus();
            break;
        case 'income':
            loadIncome();
            break;
        case 'budget':
            loadBudgetStatus();
            break;
        case 'bills':
            loadUpcomingBills();
            loadBillNotificationSettings();
            break;
        case 'bank':
            if (typeof loadPlaidStatus === 'function') loadPlaidStatus();
            if (typeof loadBankConnections === 'function') loadBankConnections();
            if (typeof initIndiaBankSync === 'function') initIndiaBankSync();
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
        case 'recurring':
            loadRecurringExpenses();
            break;
        case 'reports':
            break;
        case 'taxes':
            break;
        case 'credit':
            loadCreditScore(true);
            break;
        case 'currency':
            break;
        case 'audit':
            refreshAuditView();
            break;
        case 'chat':
            break;
        default:
            break;
    }
}

// Helper to get auth payload (session_id or api_key)
function getAuthPayload() {
    if (apiKey) return { api_key: apiKey };
    if (sessionId) return { session_id: sessionId };
    return null;
}

function isUserAuthenticated() {
    // Allow guest sessions (sessionId exists) OR Google-linked OR API key
    return !!(apiKey || sessionLinked || (userProfile && userProfile.email) || sessionId);
}
async function restPost(path, payload = {}) {
    const auth = getAuthPayload();
    if (!auth) throw new Error('Session required. Please refresh the page.');
    const response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...auth, ...payload }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : data.message;
        throw new Error(detail || `Request failed (${response.status})`);
    }
    return data;
}

async function restGet(path, params = {}) {
    const auth = getAuthPayload();
    if (!auth) throw new Error('Session required. Please refresh the page.');
    const query = new URLSearchParams({ ...auth, ...params });
    const response = await fetch(`${path}?${query.toString()}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : data.message;
        throw new Error(detail || `Request failed (${response.status})`);
    }
    return data;
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

let expenseViewPeriod = '30d';
let spendBreakdownOpen = false;

const EXPENSE_SOURCE_LABELS = {
    gmail_sync: 'Gmail',
    rag_upload: 'Bill upload',
    plaid: 'Bank',
};

function formatExpenseSource(source) {
    if (!source) return '';
    return EXPENSE_SOURCE_LABELS[source] || source.replace(/_/g, ' ');
}

const EXPENSE_PERIOD_LABELS = {
    '30d': 'Last 30 days',
    '6m': 'Last 6 months',
    'all': 'All time',
};

function setExpensePeriod(period) {
    expenseViewPeriod = period;
    document.querySelectorAll('#expensePeriodPills .period-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.period === period);
    });
    loadExpenses();
}

function toggleSpendBreakdown(forceOpen) {
    const body = document.getElementById('spendBreakdownBody');
    const chevron = document.getElementById('spendBreakdownChevron');
    if (!body) return;
    spendBreakdownOpen = typeof forceOpen === 'boolean' ? forceOpen : !spendBreakdownOpen;
    body.hidden = !spendBreakdownOpen;
    chevron?.classList.toggle('open', spendBreakdownOpen);
}

async function loadExpenses(showToastOnRefresh = false) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;

    try {
        const result = await callMCPTool('get_expenses_overview', {
            ...auth,
            period: expenseViewPeriod,
        });

        if (result?.status === 'error') {
            showMessage(result.message || 'Error loading expenses', 'error');
            return;
        }

        renderExpenseOverview(result);
        if (showToastOnRefresh) showMessage('Expenses updated', 'success');
    } catch (error) {
        console.error('Error loading expenses:', error);
        showMessage('Error loading expenses', 'error');
    }
}

function refreshExpensesView() {
    loadExpenses();
}

async function loadGmailSyncStatus() {
    const bar = document.getElementById('gmailSyncBar');
    const statusEl = document.getElementById('gmailSyncStatusText');
    const connectLink = document.getElementById('gmailConnectLink');
    const syncBtn = document.getElementById('gmailSyncBtn');
    if (!bar || !statusEl) return;

    const auth = getAuthPayload();
    if (!auth) {
        statusEl.textContent = 'Sign in to import bills from Gmail.';
        bar.className = 'gmail-sync-bar warn';
        if (connectLink) connectLink.hidden = false;
        if (syncBtn) syncBtn.disabled = true;
        return;
    }

    try {
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/gmail/status?${params}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Could not check Gmail status');

        bar.className = 'gmail-sync-bar';
        if (connectLink) connectLink.hidden = true;
        if (syncBtn) syncBtn.disabled = false;

        if (!data.connected) {
            statusEl.textContent = 'Connect Gmail to auto-import payment receipts and order bills.';
            bar.classList.add('warn');
            if (connectLink) connectLink.hidden = false;
            if (syncBtn) syncBtn.disabled = true;
            return;
        }

        if (!data.gmail_read_enabled) {
            statusEl.textContent = `${data.email || 'Google account'} connected — reconnect to enable bill import.`;
            bar.classList.add('warn');
            if (connectLink) {
                connectLink.hidden = false;
                connectLink.textContent = 'Reconnect Gmail';
            }
            if (syncBtn) syncBtn.disabled = true;
            return;
        }

        const lastSync = data.last_sync_at
            ? `Last sync: ${new Date(data.last_sync_at).toLocaleString()}`
            : 'Not synced yet';
        statusEl.textContent = `${data.email || 'Gmail'} ready · ${data.synced_count || 0} bills imported · ${lastSync}`;
        bar.classList.add('ready');
    } catch (err) {
        console.error(err);
        statusEl.textContent = 'Gmail status unavailable.';
        bar.classList.add('warn');
    }
}

async function syncGmailBills() {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;

    const syncBtn = document.getElementById('gmailSyncBtn');
    if (syncBtn) {
        syncBtn.classList.add('syncing');
        syncBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning Gmail…';
    }

    try {
        const res = await fetch('/api/gmail/sync-bills', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, max_messages: 50 }),
        });
        const data = await res.json();
        if (!res.ok) {
            if (res.status === 400 && (data.detail || '').toLowerCase().includes('connect')) {
                showMessage('Connect Gmail first to import bills.', 'warning');
                loadGmailSyncStatus();
                return;
            }
            throw new Error(data.detail || data.message || 'Gmail sync failed');
        }

        const purged = data.purged || 0;
        const imported = data.imported || 0;
        let detail = data.message || `Imported ${imported} bill(s)`;
        if (data.skip_reasons?.promotional) {
            detail += ` (${data.skip_reasons.promotional} promo emails blocked)`;
        }
        showMessage(detail, purged || imported ? 'success' : 'info');
        await loadGmailSyncStatus();
        loadExpenses();
        loadDashboardData();
    } catch (err) {
        console.error(err);
        showMessage(err.message || 'Could not import Gmail bills', 'error');
    } finally {
        if (syncBtn) {
            syncBtn.classList.remove('syncing');
            syncBtn.innerHTML = '<i class="fas fa-envelope"></i> Import from Gmail';
        }
    }
}

async function purgeGmailPromos() {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;

    const btn = document.getElementById('gmailPurgeBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cleaning…';
    }

    try {
        const res = await fetch('/api/gmail/purge-promos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(auth),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message || 'Cleanup failed');

        const removed = data.removed || 0;
        showMessage(
            removed
                ? `Removed ${removed} invalid Gmail import(s) (promos / unconfirmed payments).`
                : 'No invalid Gmail imports found.',
            removed ? 'success' : 'info'
        );
        loadExpenses();
        loadDashboardData();
    } catch (err) {
        console.error(err);
        showMessage(err.message || 'Could not clean Gmail imports', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-broom"></i> Clean bad imports';
        }
    }
}

function renderExpenseOverview(data = {}) {
    const total = data.total || 0;
    const count = data.count || 0;
    const periodLabel = EXPENSE_PERIOD_LABELS[expenseViewPeriod] || 'Custom range';

    const totalEl = document.getElementById('expenseTotalAmount');
    const countEl = document.getElementById('expenseTransactionCount');
    const periodEl = document.getElementById('expensePeriodLabel');
    if (totalEl) totalEl.textContent = formatCurrency(total);
    if (countEl) countEl.textContent = `${count} transaction${count === 1 ? '' : 's'}`;
    if (periodEl) periodEl.textContent = periodLabel;

    renderSpendBreakdown(data.categories || []);
    displayExpenses(data.expenses || []);
}

function renderSpendBreakdown(categories = []) {
    const container = document.getElementById('spendBreakdownList');
    if (!container) return;

    if (!categories.length) {
        container.innerHTML = '<p class="empty-state compact">No spending in this period yet.</p>';
        return;
    }

    container.innerHTML = categories.map(cat => {
        const safeCat = escapeHtml(cat.category || 'Other');
        const items = (cat.items || []).map(item => {
            const noteHtml = item.note
                ? `<span class="spend-item-note">${escapeHtml(item.note)}</span>`
                : '';
            const source = item.source
                ? `<span class="spend-item-source">${escapeHtml(formatExpenseSource(item.source))}</span>`
                : '';
            return `
                <li class="spend-item">
                    <span class="spend-item-date">${formatDisplayDate(item.date || '')}</span>
                    <span class="spend-item-merchant">${escapeHtml(item.merchant || '—')}</span>
                    <span class="spend-item-amount">${formatCurrency(item.amount || 0)}</span>
                    ${noteHtml}
                    ${source}
                </li>`;
        }).join('');

        return `
            <details class="spend-category">
                <summary>
                    <span class="spend-cat-name">${safeCat}</span>
                    <span class="spend-cat-meta">
                        <span class="spend-cat-total">${formatCurrency(cat.total || 0)}</span>
                        <span class="spend-cat-count">${cat.count || 0} bill${(cat.count || 0) === 1 ? '' : 's'}</span>
                    </span>
                </summary>
                <ul class="spend-items">${items}</ul>
            </details>`;
    }).join('');
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
        container.innerHTML = '<p class="empty-state">No expenses in this period. Add one or upload a bill.</p>';
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
        const expenseId = expense.id || expense._id || '';
        
        return `
        <div class="expense-item">
            <div class="expense-body">
                <div class="expense-header">
                    <span class="expense-category">${escapeHtml(category)}</span>
                    <span class="expense-amount">${formatCurrency(amount, currency)}</span>
                </div>
                <div class="expense-merchant">${escapeHtml(merchant)}</div>
                ${note ? `<div class="expense-note">${escapeHtml(note)}</div>` : ''}
            </div>
            <div class="expense-meta">
                <span class="expense-date">${formatDisplayDate(date)}</span>
                ${expense.metadata?.source ? `<span class="expense-tag">${escapeHtml(formatExpenseSource(expense.metadata.source))}</span>` : ''}
                ${expenseId ? `
                <div class="expense-actions">
                    <button class="btn-icon" title="Delete" onclick="confirmDeleteExpense('${escapeHtml(expenseId)}')"><i class="fas fa-trash"></i></button>
                </div>` : ''}
            </div>
        </div>
        `;
    }).join('');
}

async function confirmDeleteExpense(expenseId) {
    if (!confirm('Delete this expense?')) return;
    await deleteExpense(expenseId);
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

function formatChatError(error, statusCode) {
    const raw = (error && error.message) ? error.message : String(error || '');
    if (error && error.name === 'AbortError') {
        return 'The AI is taking too long. Please try again in a moment.';
    }
    if (raw === 'Failed to fetch' || raw.includes('NetworkError')) {
        return 'Cannot reach the server. Start it with: python run_app.py --skip-install then refresh this page (Ctrl+Shift+R).';
    }
    if (statusCode === 401 || raw === 'Authentication required') {
        return 'Your session expired. Click your profile menu and sign in with Google again.';
    }
    if (statusCode === 429 || raw.toLowerCase().includes('rate limit')) {
        return 'Too many requests. Please wait a minute and try again.';
    }
    return raw || 'Sorry, something went wrong.';
}

async function processChatMessage(message) {
    const placeholder = addChatMessage('Processing...', 'bot');
    const auth = getAuthPayload();
    chatHistory.push({ role: 'user', content: message });
    if (chatHistory.length > 40) {
        chatHistory = chatHistory.slice(-40);
    }
    try {
        if (!apiKey && !sessionLinked) {
            await refreshSessionStatus(false);
        }
        if (!apiKey && !sessionLinked) {
            throw new Error('Authentication required');
        }
        const payload = {
            message,
            session_id: sessionId || auth?.session_id || null,
            api_key: apiKey || auth?.api_key || null,
            history: chatHistory
        };
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);
        let res;
        try {
            res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal
            });
        } finally {
            clearTimeout(timeoutId);
        }
        const responseText = await res.text();
        let data = {};
        if (responseText) {
            try {
                data = JSON.parse(responseText);
            } catch (parseError) {
                throw new Error(res.ok ? 'Invalid server response' : `Server error (${res.status})`);
            }
        }
        if (!res.ok) {
            const detail = data.detail;
            const detailText = typeof detail === 'string'
                ? detail
                : Array.isArray(detail)
                    ? detail.map((item) => item.msg || item).join(', ')
                    : 'Chat failed';
            const err = new Error(detailText);
            err.statusCode = res.status;
            throw err;
        }
        chatHistory.push({ role: 'assistant', content: data.reply });
        updateChatMessage(placeholder, data.reply || 'Done.');
        renderToolResults(placeholder, data.tool_results);
        refreshDataForTools(data.tool_results);
    } catch (error) {
        console.error('Error processing chat:', error);
        updateChatMessage(placeholder, formatChatError(error, error.statusCode));
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
    if (names.some(t => ['add_expense', 'quick_add_expense', 'delete_expense', 'update_expense', 'document_expense_from_rag', 'add_expense_multicurrency'].includes(t))) {
        loadExpenses();
        loadDashboardData();
    }
    
    // Budget
    if (names.some(t => ['set_budget', 'check_budget_status'].includes(t))) {
        loadBudgetStatus();
        loadDashboardData();
    }
    
    // Income
    if (names.some(t => ['add_income', 'list_income', 'get_income_summary'].includes(t))) {
        loadIncome();
        loadDashboardData();
    }
    
    // Recurring Expenses
    if (names.some(t => ['add_recurring_expense', 'generate_monthly_recurring'].includes(t))) {
        loadExpenses();
        loadDashboardData();
        if (typeof loadRecurringExpenses === 'function') {
            loadRecurringExpenses();
        }
    }
    
    // Bill Reminders
    if (names.some(t => ['add_bill_reminder', 'get_upcoming_bills'].includes(t)) && typeof loadUpcomingBills === 'function') {
        loadUpcomingBills();
    }
    
    // Savings Goals
    if (names.some(t => ['set_savings_goal', 'track_savings_progress'].includes(t))) {
        loadSavingsGoals();
        loadDashboardData();
    }
    
    // Investments
    if (names.some(t => ['add_investment', 'get_investment_portfolio', 'update_investment_value'].includes(t))) {
        loadInvestments();
        loadDashboardData();
    }
    
    // Debts
    if (names.some(t => ['add_debt', 'get_debt_summary', 'record_debt_payment'].includes(t))) {
        loadDebts();
        loadDashboardData();
    }
    
    // Financial Reports
    if (names.includes('generate_financial_report')) {
        loadDashboardData();
        if (typeof loadFinancialReports === 'function') {
            loadFinancialReports();
        }
    }
    
    // Tax Estimation
    if (names.includes('estimate_taxes')) {
        if (typeof loadTaxEstimates === 'function') {
            loadTaxEstimates();
        }
    }
    
    // Credit Score
    if (names.some(t => ['record_credit_score', 'get_credit_score_trend'].includes(t))) {
        if (typeof loadCreditScore === 'function') {
            loadCreditScore();
        }
    }
    
    // Stock queries (yahoo_finance, get_stock_returns, get_stock_return_one_year) - no UI refresh needed, just display in chat
    
    // Currency
    if (names.includes('set_base_currency')) {
        const result = toolResults.find(r => r.tool === 'set_base_currency');
        const currency = result?.result?.base_currency;
        if (currency) {
            applyBaseCurrency(currency);
        }
    }
    
    // Bank Integration
    if (names.some(t => ['connect_bank_account', 'sync_bank_transactions'].includes(t))) {
        loadExpenses();
        loadDashboardData();
    }
    
    // User Management (no UI refresh needed)
    // - register_user, login_user, get_user_info, create_user_role, share_account_access
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
            if (showToast) showMessage('Account linked! You can start using TrackExpensio.', 'success');
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
        if (userProfile && userProfile.email) {
            sessionLinked = true;
            toggleAuthOverlay(false);
            localStorage.setItem('googleLinked', 'true');
        }
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
    // Hide overlay if user has ANY session (guest or Google)
    if (show && !apiKey && !sessionId) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

function ensureAuthenticated(showToast = true) {
    if (isUserAuthenticated()) {
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

let marketLookupInFlight = false;

async function lookupTicker(event) {
    event?.preventDefault();
    if (marketLookupInFlight) return;

    const form = event?.target?.closest?.('form.market-form')
        || document.querySelector('.market-widget .market-form');
    const widget = form?.closest('.market-widget');
    const symbolInput = form?.querySelector('.market-symbol-input');
    const container = widget?.querySelector('.market-result');
    const submitBtn = form?.querySelector('button[type="submit"]');

    const symbol = symbolInput?.value.trim();
    if (!symbol) {
        showMessage('Enter a ticker or company name (e.g. AAPL, Netflix, INFY.NS)', 'warning');
        return;
    }
    const auth = getAuthPayload();
    if (!auth) {
        showMessage('Session required. Please refresh the page.', 'warning');
        return;
    }

    marketLookupInFlight = true;
    if (submitBtn) submitBtn.disabled = true;
    if (container) container.textContent = 'Fetching market data…';
    try {
        const response = await restGet('/api/market/quote', { symbol });
        const snapshot = response.data || response;
        if (!snapshot || !snapshot.price) {
            const msg = `No data for "${symbol}". Try AAPL, NFLX, or INFY.NS`;
            if (container) container.innerHTML = `<div class="error-message">${msg}</div>`;
            showMessage(msg, 'error');
            return;
        }
        if (container) container.innerHTML = renderMarketResult(snapshot);
        const label = snapshot.symbol || symbol;
        showMessage(`Fetched data for ${label}`, 'success');
    } catch (error) {
        console.error('Yahoo Finance error:', error);
        const errorMsg = error.message || 'Unable to fetch ticker. Please try again.';
        if (container) container.innerHTML = `<div class="error-message">${escapeHtml(errorMsg)}</div>`;
        showMessage(errorMsg, 'error');
    } finally {
        marketLookupInFlight = false;
        if (submitBtn) submitBtn.disabled = false;
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
            <p style="margin-top:12px;">
                <a href="/stocks/page?ticker=${encodeURIComponent(snapshot.symbol || '')}" style="color:#58a6ff;text-decoration:none;font-weight:600;">
                    Open Smart Stock Analyzer (RSI, MACD, chart) →
                </a>
            </p>
        </div>
    `;
}

async function loadUpcomingBills() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const data = await callMCPTool('get_upcoming_bills', { ...auth, days_ahead: 60 });
        renderBillList(data);
    } catch (error) {
        console.error('Error loading bills:', error);
        showMessage('Error loading bill reminders', 'error');
    }
}

function showAddBillModal() {
    if (!ensureAuthenticated()) return;
    const dateInput = document.getElementById('billDueDate');
    if (dateInput && !dateInput.value) {
        const d = new Date();
        d.setDate(d.getDate() + 7);
        dateInput.value = d.toISOString().split('T')[0];
    }
    showModal('addBillModal');
}

async function submitBillReminder(event) {
    event.preventDefault();
    if (!ensureAuthenticated()) return;
    await addBillReminder(
        document.getElementById('billName').value,
        document.getElementById('billAmount').value,
        document.getElementById('billDueDate').value,
        document.getElementById('billFrequency').value,
        'Bills'
    );
    closeModal('addBillModal');
    document.getElementById('addBillForm')?.reset();
}

async function loadBillNotificationSettings() {
    const auth = getAuthPayload();
    const statusEl = document.getElementById('billEmailStatus');
    if (!auth || !statusEl) return;
    try {
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/user/notification-settings?${params}`);
        const data = await res.json();
        const settings = data.settings || {};
        const enabled = document.getElementById('billEmailEnabled');
        const emailInput = document.getElementById('billAlertEmail');
        if (enabled) enabled.checked = settings.bill_email_enabled !== false;
        if (emailInput) {
            emailInput.placeholder = data.profile_email
                ? `Default: ${data.profile_email}`
                : 'Alert email address';
            if (settings.alert_email) emailInput.value = settings.alert_email;
        }
        if (!data.smtp_configured) {
            statusEl.innerHTML = '<span class="warning">SMTP not configured — add SMTP_* to .env to enable emails.</span>';
        } else {
            statusEl.textContent = 'Reminders send at 7, 3, 1, and 0 days before due date (hourly check).';
        }
    } catch (e) {
        console.error(e);
    }
}

async function saveBillNotificationSettings() {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const res = await fetch('/api/user/notification-settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...auth,
                bill_email_enabled: document.getElementById('billEmailEnabled')?.checked,
                alert_email: document.getElementById('billAlertEmail')?.value || null,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Save failed');
        showMessage('Notification settings saved', 'success');
    } catch (e) {
        showMessage(e.message || 'Could not save settings', 'error');
    }
}

async function runBillRemindersNow() {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/bills/reminders/run?${params}`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        if (data.status === 'skipped') {
            showMessage(data.message || 'SMTP not configured', 'warning');
            return;
        }
        showMessage(`Sent ${data.sent || 0} reminder email(s)`, 'success');
    } catch (e) {
        showMessage(e.message || 'Could not send reminders', 'error');
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
        updateSavingsGoalButton(data);
    } catch (error) {
        console.error('Error loading savings goals:', error);
    }
}

function updateSavingsGoalButton(data = {}) {
    const btn = document.getElementById('addSavingsGoalBtn');
    if (!btn) return;
    const goals = data.goals || [];
    const hasGoals = goals.length > 0;
    btn.innerHTML = hasGoals
        ? '<i class="fas fa-plus"></i> Add Another Goal'
        : '<i class="fas fa-plus"></i> Add Goal';
}

function showAddSavingsGoalModal(goal = null) {
    if (!ensureAuthenticated()) return;
    const form = document.getElementById('addSavingsGoalForm');
    const title = document.getElementById('savingsGoalModalTitle');
    const idInput = document.getElementById('savingsGoalId');
    const nameInput = document.getElementById('savingsGoalName');
    const numberInput = document.getElementById('savingsGoalNumber');
    const targetInput = document.getElementById('savingsGoalTarget');
    const currentInput = document.getElementById('savingsGoalCurrent');
    const dateInput = document.getElementById('savingsGoalDate');

    form?.reset();
    if (idInput) idInput.value = goal?.goal_id || goal?.id || '';
    if (nameInput) nameInput.value = goal?.goal_name || '';
    if (numberInput) numberInput.value = goal?.goal_number || '';
    if (targetInput) targetInput.value = goal?.target_amount ?? '';
    if (currentInput) currentInput.value = goal?.current_amount ?? 0;
    if (dateInput) {
        if (goal?.target_date) {
            dateInput.value = goal.target_date.split('T')[0];
        } else {
            const d = new Date();
            d.setMonth(d.getMonth() + 6);
            dateInput.value = d.toISOString().split('T')[0];
        }
    }
    if (title) {
        title.innerHTML = goal
            ? '<i class="fas fa-bullseye"></i> Edit Savings Goal'
            : '<i class="fas fa-bullseye"></i> Add Savings Goal';
    }
    showModal('addSavingsGoalModal');
}

async function submitSavingsGoal(event) {
    event.preventDefault();
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;

    const goalId = document.getElementById('savingsGoalId')?.value?.trim();
    const payload = {
        ...auth,
        goal_name: document.getElementById('savingsGoalName').value.trim(),
        goal_number: document.getElementById('savingsGoalNumber').value.trim(),
        target_amount: parseFloat(document.getElementById('savingsGoalTarget').value),
        current_amount: parseFloat(document.getElementById('savingsGoalCurrent').value) || 0,
        target_date: document.getElementById('savingsGoalDate').value,
    };

    try {
        const tool = goalId ? 'set_savings_goal' : 'add_savings_goal';
        if (goalId) payload.goal_id = goalId;
        const result = await callMCPTool(tool, payload);
        if (result.status === 'success') {
            showMessage(result.message || 'Savings goal saved', 'success');
            closeModal('addSavingsGoalModal');
            document.getElementById('addSavingsGoalForm')?.reset();
            loadSavingsGoals();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to save savings goal', 'error');
        }
    } catch (error) {
        console.error('Error saving savings goal:', error);
        showMessage('Error saving savings goal', 'error');
    }
}

function editSavingsGoal(goalId) {
    const goal = (window._savingsGoalsCache || []).find(
        g => (g.goal_id || g.id) === goalId
    );
    if (goal) showAddSavingsGoalModal(goal);
}

async function deleteSavingsGoal(goalId, goalName) {
    if (!ensureAuthenticated()) return;
    const label = goalName ? `"${goalName}"` : 'this goal';
    if (!confirm(`Remove ${label}?`)) return;
    const auth = getAuthPayload();
    if (!auth) return;

    try {
        const result = await callMCPTool('delete_savings_goal', { ...auth, goal_id: goalId });
        if (result.status === 'success') {
            showMessage(result.message || 'Goal removed', 'success');
            loadSavingsGoals();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to remove goal', 'error');
        }
    } catch (error) {
        console.error('Error deleting savings goal:', error);
        showMessage('Error removing savings goal', 'error');
    }
}

function renderSavingsGoals(data = {}) {
    const container = document.getElementById('savingsGoalsList');
    if (!container) return;
    
    const goals = data.goals || data.savings_goals || [];
    window._savingsGoalsCache = goals;
    if (!goals.length) {
        container.innerHTML = '<p class="empty-state">No savings goals yet. Click Add Goal to create one.</p>';
        return;
    }
    
    container.innerHTML = goals.map(goal => {
        const goalId = goal.goal_id || goal.id || '';
        const target = goal.target_amount || 0;
        const current = goal.current_amount || 0;
        const progress = goal.progress_percentage != null
            ? goal.progress_percentage
            : (target > 0 ? (current / target) * 100 : 0);
        const remaining = goal.remaining != null ? goal.remaining : Math.max(0, target - current);
        const goalNumber = goal.goal_number ? `<span class="goal-number">#${escapeHtml(String(goal.goal_number))}</span>` : '';
        const safeId = escapeHtml(goalId);
        const safeName = escapeHtml(goal.goal_name || 'Savings Goal');
        
        return `
        <div class="savings-goal-item" data-goal-id="${safeId}">
            <div class="goal-header">
                <div class="goal-title-row">
                    <h3>${safeName} ${goalNumber}</h3>
                    <div class="goal-actions">
                        <button class="btn-icon" title="Edit" onclick="editSavingsGoal('${safeId}')">
                            <i class="fas fa-pen"></i>
                        </button>
                        <button class="btn-icon" title="Remove" onclick="deleteSavingsGoal('${safeId}', '${safeName.replace(/'/g, "\\'")}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <span class="goal-amount">${formatCurrency(target)}</span>
            </div>
            <div class="goal-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.min(100, progress)}%"></div>
                </div>
                <div class="progress-text">
                    ${formatCurrency(current)} / ${formatCurrency(target)} (${Number(progress).toFixed(1)}%)
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
    const container = document.getElementById('debtList');
    if (!container) return;
    
    const debts = data.debts || [];
    if (!debts.length) {
        container.innerHTML = '<p class="empty-state">No debts recorded yet.</p>';
        return;
    }
    
    container.innerHTML = debts.map(debt => {
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
    if (!ensureAuthenticated()) return;
    const dateInput = document.getElementById('expenseDate');
    if (dateInput && !dateInput.value) {
        dateInput.value = new Date().toISOString().split('T')[0];
    }
    showModal('addExpenseModal');
}

function showAddIncomeModal() {
    if (!ensureAuthenticated()) return;
    const dateInput = document.getElementById('incomeDate');
    if (dateInput && !dateInput.value) {
        dateInput.value = new Date().toISOString().split('T')[0];
    }
    showModal('addIncomeModal');
}

function showSetBudgetModal() {
    if (!ensureAuthenticated()) return;
    showModal('setBudgetModal');
}

async function addIncome(event) {
    event.preventDefault();
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const result = await restPost('/api/income', {
            date: document.getElementById('incomeDate').value,
            amount: parseFloat(document.getElementById('incomeAmount').value),
            source: document.getElementById('incomeSource').value,
            category: document.getElementById('incomeCategory').value,
            note: document.getElementById('incomeNote').value || '',
        });
        if (result?.status === 'error') {
            showMessage(result.message || 'Failed to add income', 'error');
            return;
        }
        showMessage('Income added successfully!', 'success');
        closeModal('addIncomeModal');
        document.getElementById('addIncomeForm').reset();
        loadIncome();
        loadDashboardData();
    } catch (error) {
        showMessage(error.message || 'Error adding income', 'error');
    }
}

async function submitBudget(event) {
    event.preventDefault();
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const result = await restPost('/api/budget', {
            category: document.getElementById('budgetCategory').value,
            amount: parseFloat(document.getElementById('budgetAmount').value),
        });
        if (result?.status === 'error') {
            showMessage(result.message || 'Failed to save budget', 'error');
            return;
        }
        showMessage('Budget saved!', 'success');
        closeModal('setBudgetModal');
        document.getElementById('setBudgetForm').reset();
        loadBudgetStatus();
        loadDashboardData();
    } catch (error) {
        showMessage(error.message || 'Error saving budget', 'error');
    }
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
        date: document.getElementById('expenseDate').value,
        amount: parseFloat(document.getElementById('expenseAmount').value),
        category: document.getElementById('expenseCategory').value,
        merchant: document.getElementById('expenseMerchant').value,
        note: document.getElementById('expenseNote').value
    };
    
    try {
        const result = await restPost('/api/expenses', formData);
        if (result?.status === 'error') {
            showMessage(result.message || 'Failed to add expense', 'error');
            return;
        }
        showMessage('Expense added successfully!', 'success');
        closeModal('addExpenseModal');
        document.getElementById('addExpenseForm').reset();
        loadExpenses();
        loadDashboardData();
    } catch (error) {
        console.error('Error adding expense:', error);
        showMessage(error.message || 'Error adding expense', 'error');
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

// Additional MCP Tool Functions - Ensuring all model functions are accessible

// Delete Expense
async function deleteExpense(expenseId) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('delete_expense', {
            ...auth,
            expense_id: expenseId
        });
        if (result.status === 'success') {
            showMessage('Expense deleted successfully', 'success');
            loadExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to delete expense', 'error');
        }
    } catch (error) {
        console.error('Error deleting expense:', error);
        showMessage('Error deleting expense', 'error');
    }
}

// Update Expense
async function updateExpense(expenseId, updates) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('update_expense', {
            ...auth,
            expense_id: expenseId,
            ...updates
        });
        if (result.status === 'success') {
            showMessage('Expense updated successfully', 'success');
            loadExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to update expense', 'error');
        }
    } catch (error) {
        console.error('Error updating expense:', error);
        showMessage('Error updating expense', 'error');
    }
}

// AI Insights
async function loadAIInsights(startDate = null, endDate = null) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        if (!startDate || !endDate) {
            const today = new Date();
            startDate = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
            endDate = today.toISOString().split('T')[0];
        }
        
        const result = await callMCPTool('ai_insights', {
            ...auth,
            start_date: startDate,
            end_date: endDate
        });
        return result;
    } catch (error) {
        console.error('Error loading AI insights:', error);
        return null;
    }
}

// Recurring Expenses
async function loadRecurringExpenses() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    const container = document.getElementById('recurringList');
    if (!auth || !container) return;
    
    try {
        const today = new Date();
        const startDate = new Date(today.getFullYear(), today.getMonth() - 6, 1).toISOString().split('T')[0];
        const endDate = today.toISOString().split('T')[0];
        const expenses = await callMCPTool('list_expenses', { ...auth, start_date: startDate, end_date: endDate });
        const recurring = Array.isArray(expenses) ? expenses.filter(e => e.metadata?.recurring) : [];
        if (!recurring.length) {
            container.innerHTML = '<p class="empty-state">No recurring expenses yet. Ask AI: "Add recurring rent 15000 on 1st".</p>';
            return;
        }
        container.innerHTML = recurring.map(e => `
            <div class="expense-item">
                <div class="expense-header">
                    <span class="expense-category">${e.category || 'Recurring'}</span>
                    <span class="expense-amount">${formatCurrency(e.amount)}</span>
                </div>
                <div class="expense-merchant">${e.merchant || e.note || '—'}</div>
                <div class="expense-date">${formatDisplayDate(e.date)}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading recurring expenses:', error);
        container.innerHTML = '<p class="empty-state">Could not load recurring expenses.</p>';
    }
}

async function loadFinancialReports(generate = false) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    const container = document.getElementById('reportsOutput');
    if (!auth || !container) return;
    if (!generate) return;
    
    try {
        container.innerHTML = '<p class="empty-state">Generating report…</p>';
        const today = new Date();
        const startDate = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
        const endDate = today.toISOString().split('T')[0];
        const result = await callMCPTool('generate_financial_report', { ...auth, start_date: startDate, end_date: endDate });
        container.innerHTML = `<pre class="report-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    } catch (error) {
        console.error('Error loading financial report:', error);
        container.innerHTML = '<p class="empty-state">Could not generate report.</p>';
    }
}

async function loadTaxEstimates(runEstimate = false) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    const container = document.getElementById('taxOutput');
    if (!auth || !container) return;
    if (!runEstimate) return;
    
    try {
        const jurisdiction = document.getElementById('taxJurisdiction')?.value || 'IN';
        container.innerHTML = '<p class="empty-state">Calculating…</p>';
        const result = await callMCPTool('estimate_taxes', {
            ...auth,
            tax_year: new Date().getFullYear(),
            jurisdiction,
        });
        container.innerHTML = `<pre class="report-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    } catch (error) {
        console.error('Error loading tax estimates:', error);
        container.innerHTML = '<p class="empty-state">Could not estimate taxes.</p>';
    }
}

async function loadCreditScore(refresh = false) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    const container = document.getElementById('creditOutput');
    if (!auth || !container) return;
    
    try {
        const result = await callMCPTool('get_credit_score_trend', auth);
        const scores = result?.scores || result?.trend || [];
        if (!scores || (Array.isArray(scores) && scores.length === 0)) {
            if (refresh) container.innerHTML = '<p class="empty-state">No credit scores on file.</p>';
            return;
        }
        container.innerHTML = `<pre class="report-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    } catch (error) {
        console.error('Error loading credit score:', error);
        if (refresh) container.innerHTML = '<p class="empty-state">Could not load credit data.</p>';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Quick Add Expense (simplified version)
async function quickAddExpense(amount, description, merchant = '') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('quick_add_expense', {
            ...auth,
            amount: parseFloat(amount),
            description: description,
            merchant: merchant
        });
        if (result.status === 'success') {
            showMessage('Expense added successfully', 'success');
            loadExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add expense', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding expense:', error);
        showMessage('Error adding expense', 'error');
        return null;
    }
}

// Set Budget
async function setBudget(category, amount) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('set_budget', {
            ...auth,
            category: category,
            amount: parseFloat(amount)
        });
        if (result.status === 'success') {
            showMessage(`Budget set for ${category}`, 'success');
            loadBudgetStatus();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to set budget', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error setting budget:', error);
        showMessage('Error setting budget', 'error');
        return null;
    }
}

// Add Income
async function addIncome(date, amount, source, category = 'Salary', note = '') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('add_income', {
            ...auth,
            date: date,
            amount: parseFloat(amount),
            source: source,
            category: category,
            note: note
        });
        if (result.status === 'success') {
            showMessage('Income added successfully', 'success');
            loadIncome();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add income', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding income:', error);
        showMessage('Error adding income', 'error');
        return null;
    }
}

// Set Savings Goal (programmatic / chat helper)
async function setSavingsGoal(goalName, targetAmount, targetDate, currentAmount = 0, goalNumber = '', goalId = null) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const payload = {
            ...auth,
            goal_name: goalName,
            target_amount: parseFloat(targetAmount),
            target_date: targetDate,
            current_amount: parseFloat(currentAmount),
            goal_number: goalNumber || '',
        };
        if (goalId) payload.goal_id = goalId;
        const tool = goalId ? 'set_savings_goal' : 'add_savings_goal';
        const result = await callMCPTool(tool, payload);
        if (result.status === 'success') {
            showMessage('Savings goal set successfully', 'success');
            loadSavingsGoals();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to set savings goal', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error setting savings goal:', error);
        showMessage('Error setting savings goal', 'error');
        return null;
    }
}

// Add Debt
async function addDebt(creditorName, totalAmount, interestRate, minimumPayment, dueDate, debtType = 'Credit Card') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('add_debt', {
            ...auth,
            creditor_name: creditorName,
            total_amount: parseFloat(totalAmount),
            interest_rate: parseFloat(interestRate),
            minimum_payment: parseFloat(minimumPayment),
            due_date: dueDate,
            debt_type: debtType
        });
        if (result.status === 'success') {
            showMessage('Debt added successfully', 'success');
            loadDebts();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add debt', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding debt:', error);
        showMessage('Error adding debt', 'error');
        return null;
    }
}

// Record Debt Payment
async function recordDebtPayment(debtId, amount, paymentDate) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('record_debt_payment', {
            ...auth,
            debt_id: debtId,
            amount: parseFloat(amount),
            payment_date: paymentDate
        });
        if (result.status === 'success') {
            showMessage('Debt payment recorded', 'success');
            loadDebts();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to record payment', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error recording debt payment:', error);
        showMessage('Error recording payment', 'error');
        return null;
    }
}

// Add Investment
async function addInvestment(investmentName, investmentType, amount, purchaseDate, currentValue = null) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const params = {
            ...auth,
            investment_name: investmentName,
            investment_type: investmentType,
            amount: parseFloat(amount),
            purchase_date: purchaseDate
        };
        if (currentValue !== null) {
            params.current_value = parseFloat(currentValue);
        }
        
        const result = await callMCPTool('add_investment', params);
        if (result.status === 'success') {
            showMessage('Investment added successfully', 'success');
            loadInvestments();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add investment', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding investment:', error);
        showMessage('Error adding investment', 'error');
        return null;
    }
}

// Update Investment Value
async function updateInvestmentValue(investmentId, currentValue, updateDate = null) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const params = {
            ...auth,
            investment_id: investmentId,
            current_value: parseFloat(currentValue)
        };
        if (updateDate) {
            params.update_date = updateDate;
        }
        
        const result = await callMCPTool('update_investment_value', params);
        if (result.status === 'success') {
            showMessage('Investment value updated', 'success');
            loadInvestments();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to update investment', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error updating investment:', error);
        showMessage('Error updating investment', 'error');
        return null;
    }
}

// Add Bill Reminder
async function addBillReminder(billName, amount, dueDate, frequency = 'monthly', category = 'Bills') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('add_bill_reminder', {
            ...auth,
            bill_name: billName,
            amount: parseFloat(amount),
            due_date: dueDate,
            frequency: frequency,
            category: category
        });
        if (result.status === 'success') {
            showMessage('Bill reminder added', 'success');
            loadUpcomingBills();
        } else {
            showMessage(result.message || 'Failed to add bill reminder', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding bill reminder:', error);
        showMessage('Error adding bill reminder', 'error');
        return null;
    }
}

// Add Recurring Expense
async function addRecurringExpense(amount, category, frequency, merchant = '') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('add_recurring_expense', {
            ...auth,
            amount: parseFloat(amount),
            category: category,
            frequency: frequency,
            merchant: merchant
        });
        if (result.status === 'success') {
            showMessage('Recurring expense added', 'success');
            loadRecurringExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add recurring expense', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding recurring expense:', error);
        showMessage('Error adding recurring expense', 'error');
        return null;
    }
}

// Generate Monthly Recurring
async function generateMonthlyRecurring() {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('generate_monthly_recurring', auth);
        if (result.status === 'success') {
            showMessage('Monthly recurring expenses generated', 'success');
            loadExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to generate recurring expenses', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error generating monthly recurring:', error);
        showMessage('Error generating recurring expenses', 'error');
        return null;
    }
}

// Record Credit Score
async function recordCreditScore(score, date, creditBureau = 'General') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('record_credit_score', {
            ...auth,
            score: parseInt(score),
            date: date,
            credit_bureau: creditBureau
        });
        if (result.status === 'success') {
            showMessage('Credit score recorded', 'success');
            loadCreditScore();
        } else {
            showMessage(result.message || 'Failed to record credit score', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error recording credit score:', error);
        showMessage('Error recording credit score', 'error');
        return null;
    }
}

// Add Expense Multi-Currency
async function addExpenseMultiCurrency(date, amount, currency, category, note = '', merchant = '') {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('add_expense_multicurrency', {
            ...auth,
            date: date,
            amount: parseFloat(amount),
            currency: currency,
            category: category,
            note: note,
            merchant: merchant
        });
        if (result.status === 'success') {
            showMessage('Multi-currency expense added', 'success');
            loadExpenses();
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to add expense', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error adding multi-currency expense:', error);
        showMessage('Error adding expense', 'error');
        return null;
    }
}

// Set Base Currency
async function setBaseCurrency(currency) {
    if (!ensureAuthenticated()) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('set_base_currency', {
            ...auth,
            currency: currency
        });
        if (result.status === 'success') {
            const newCurrency = result.base_currency || currency;
            applyBaseCurrency(newCurrency);
            showMessage(`Base currency set to ${newCurrency}`, 'success');
            loadDashboardData();
        } else {
            showMessage(result.message || 'Failed to set base currency', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error setting base currency:', error);
        showMessage('Error setting base currency', 'error');
        return null;
    }
}

// Get Stock Returns
async function getStockReturns(symbol, years = null, fromListing = false) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const params = {
            ...auth,
            symbol: symbol.toUpperCase()
        };
        if (years !== null) {
            params.years = parseInt(years);
        }
        if (fromListing) {
            params.from_listing = true;
        }
        
        const result = await callMCPTool('get_stock_returns', params);
        return result;
    } catch (error) {
        console.error('Error getting stock returns:', error);
        return null;
    }
}

// Get Stock Return One Year
async function getStockReturnOneYear(symbol) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const result = await callMCPTool('get_stock_return_one_year', {
            ...auth,
            symbol: symbol.toUpperCase()
        });
        return result;
    } catch (error) {
        console.error('Error getting stock return:', error);
        return null;
    }
}

// Get Help
async function getHelp() {
    try {
        const result = await callMCPTool('get_help', {});
        return result;
    } catch (error) {
        console.error('Error getting help:', error);
        return null;
    }
}

// Audit Logs — state
let auditOffset = 0;
const AUDIT_PAGE_SIZE = 40;

function resetAuditOffset() {
    auditOffset = 0;
}

function buildAuditQueryParams() {
    const auth = getAuthPayload();
    if (!auth) return null;
    const params = new URLSearchParams(auth);
    const startDate = document.getElementById('auditStartDate')?.value;
    const endDate = document.getElementById('auditEndDate')?.value;
    const action = document.getElementById('auditActionFilter')?.value;
    const category = document.getElementById('auditCategoryFilter')?.value;
    const severity = document.getElementById('auditSeverityFilter')?.value;
    const source = document.getElementById('auditSourceFilter')?.value;
    const success = document.getElementById('auditSuccessFilter')?.value;
    const search = document.getElementById('auditSearch')?.value?.trim();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    if (action) params.set('action', action);
    if (category) params.set('category', category);
    if (severity) params.set('severity', severity);
    if (source) params.set('source', source);
    if (success !== '') params.set('success', success);
    if (search) params.set('search', search);
    params.set('limit', String(AUDIT_PAGE_SIZE));
    params.set('offset', String(auditOffset));
    return params;
}

async function refreshAuditView() {
    await Promise.all([loadAuditDashboard(), loadAuditLogs()]);
}

async function loadAuditDashboard() {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    const hours = document.getElementById('auditPeriodHours')?.value || '168';
    const params = new URLSearchParams({ ...auth, hours });
    try {
        const res = await fetch(`/api/audit/dashboard?${params}`);
        if (!res.ok) throw new Error('dashboard failed');
        const data = await res.json();
        renderAuditDashboard(data.dashboard || {});
    } catch (e) {
        console.error('Audit dashboard:', e);
    }
}

function renderAuditDashboard(d) {
    const statsEl = document.getElementById('auditStatsGrid');
    const timelineEl = document.getElementById('auditTimeline');
    const alertsEl = document.getElementById('auditAlerts');
    if (!statsEl) return;

    statsEl.innerHTML = `
        <div class="audit-stat-card"><span class="label">Total events</span><span class="value">${d.total_events ?? 0}</span></div>
        <div class="audit-stat-card fail"><span class="label">Failed</span><span class="value">${d.failed_events ?? 0}</span></div>
        <div class="audit-stat-card"><span class="label">Success rate</span><span class="value">${d.success_rate_pct ?? 100}%</span></div>
        <div class="audit-stat-card critical"><span class="label">Critical</span><span class="value">${d.critical_events ?? 0}</span></div>
        <div class="audit-stat-card"><span class="label">Unique IPs</span><span class="value">${d.unique_ips ?? 0}</span></div>
    `;

    if (alertsEl) {
        const alerts = d.security_alerts || [];
        alertsEl.innerHTML = alerts.length
            ? alerts.map(a => `
                <div class="audit-alert audit-alert-${a.level}">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>${escapeHtml(a.message)}</span>
                </div>`).join('')
            : '';
    }

    if (timelineEl && d.timeline?.length) {
        const max = Math.max(...d.timeline.map(t => t.count), 1);
        timelineEl.innerHTML = `
            <h3 class="audit-chart-title">Activity timeline</h3>
            <div class="audit-bars">
                ${d.timeline.map(t => `
                    <div class="audit-bar-wrap" title="${t.date}: ${t.count} events (${t.failures} failed)">
                        <div class="audit-bar" style="height:${Math.max(4, (t.count / max) * 100)}%"></div>
                        <span class="audit-bar-label">${t.date.slice(5)}</span>
                    </div>
                `).join('')}
            </div>`;
    } else if (timelineEl) {
        timelineEl.innerHTML = '';
    }

    const summaryEl = document.getElementById('auditSummary');
    if (summaryEl && d.by_category) {
        const chips = Object.entries(d.by_category)
            .map(([k, v]) => `<span class="audit-chip audit-cat-${k}">${k}: ${v}</span>`)
            .join('');
        summaryEl.innerHTML = chips ? `<div class="audit-chips">${chips}</div>` : '';
    }
}

async function loadAuditLogs(append = false) {
    if (!ensureAuthenticated()) return;
    const listEl = document.getElementById('auditLogList');
    const pageEl = document.getElementById('auditPagination');
    if (!listEl) return;

    if (!append) {
        listEl.innerHTML = '<p class="empty-state">Loading activity log…</p>';
    }

    const params = buildAuditQueryParams();
    if (!params) return;

    try {
        const res = await fetch(`/api/audit/logs?${params}`);
        if (!res.ok) throw new Error('logs failed');
        const result = await res.json();
        const logs = result.logs || [];

        if (!logs.length && !append) {
            listEl.innerHTML = '<p class="empty-state">No activity recorded for this period.</p>';
            if (pageEl) pageEl.innerHTML = '';
            return;
        }

        const html = logs.map(log => renderAuditEntry(log)).join('');
        if (append) {
            listEl.insertAdjacentHTML('beforeend', html);
        } else {
            listEl.innerHTML = html;
        }

        if (pageEl) {
            const shown = auditOffset + logs.length;
            pageEl.innerHTML = `
                <span class="audit-page-info">Showing ${shown} of ${result.total ?? shown}</span>
                ${result.has_more ? `<button class="btn-secondary" onclick="loadMoreAuditLogs()">Load more</button>` : ''}
            `;
        }
    } catch (error) {
        console.error('Error loading audit logs:', error);
        listEl.innerHTML = '<p class="empty-state">Could not load audit logs.</p>';
    }
}

function loadMoreAuditLogs() {
    auditOffset += AUDIT_PAGE_SIZE;
    loadAuditLogs(true);
}

function renderAuditEntry(log) {
    const ok = log.success !== false;
    const sev = log.severity || 'info';
    const changes = log.changes
        ? `<pre class="audit-details audit-changes">${escapeHtml(JSON.stringify(log.changes, null, 2))}</pre>`
        : '';
    const details = log.details && Object.keys(log.details).length
        ? `<details class="audit-expand"><summary>Details</summary><pre class="audit-details">${escapeHtml(JSON.stringify(log.details, null, 2))}</pre></details>`
        : '';
    const tool = log.tool_name ? `<span class="audit-tool" title="MCP tool">⚙ ${escapeHtml(log.tool_name)}</span>` : '';
    const ip = log.ip_address ? `<span title="IP">${escapeHtml(log.ip_address)}</span>` : '';
    const dur = log.duration_ms != null ? `<span>${log.duration_ms}ms</span>` : '';
    const req = log.request_id ? `<span class="audit-req" title="Request ID">${log.request_id.slice(0, 8)}…</span>` : '';

    return `
    <article class="audit-entry audit-sev-${sev} ${ok ? 'audit-ok' : 'audit-fail'}">
        <div class="audit-entry-header">
            <span class="audit-action">
                <span class="audit-cat-badge">${log.category || 'system'}</span>
                ${log.action_label || log.action}
            </span>
            <span class="audit-time">${formatAuditTime(log.timestamp)}</span>
        </div>
        <div class="audit-meta">
            <span class="audit-severity-badge">${sev}</span>
            <span class="audit-source">${log.source || 'system'}</span>
            ${tool}
            ${log.resource_type ? `<span>${log.resource_type}${log.resource_id ? ' #' + String(log.resource_id).slice(0, 12) : ''}</span>` : ''}
            ${ip}
            ${dur}
            ${req}
            <span class="audit-status">${ok ? '✓' : '✗'}</span>
        </div>
        ${log.error_message ? `<p class="audit-error">${escapeHtml(log.error_message)}</p>` : ''}
        ${changes}
        ${details}
    </article>`;
}

function formatAuditTime(iso) {
    if (!iso) return '';
    try {
        return new Date(iso).toLocaleString();
    } catch {
        return iso;
    }
}

async function exportAuditCsv() {
    if (!ensureAuthenticated()) return;
    const params = buildAuditQueryParams();
    if (!params) return;
    params.delete('limit');
    params.delete('offset');
    try {
        const res = await fetch(`/api/audit/export?${params}`);
        if (!res.ok) throw new Error('export failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audit_export_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showMessage('Audit log exported', 'success');
        refreshAuditView();
    } catch (e) {
        showMessage('Export failed', 'error');
    }
}

async function getAuditLogs(startDate = null, endDate = null, action = null, limit = 100) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    try {
        const params = { ...auth, limit };
        if (startDate) params.start_date = startDate;
        if (endDate) params.end_date = endDate;
        if (action) params.action = action;
        return await callMCPTool('get_audit_logs', params);
    } catch (error) {
        console.error('Error getting audit logs:', error);
        return null;
    }
}

async function showHelpModal() {
    const result = await getHelp();
    const lines = result?.commands || [];
    const preview = result?.message || 'TrackExpensio Help';
    showMessage(lines.length ? `${preview} — ${lines.length} commands (see browser console)` : preview, 'info');
    if (result) console.info('TrackExpensio help:', result);
}

async function runCurrencyConversion() {
    if (!ensureAuthenticated()) return;
    const amount = parseFloat(document.getElementById('fxAmount')?.value);
    const from = document.getElementById('fxFrom')?.value?.trim();
    const to = document.getElementById('fxTo')?.value?.trim();
    const out = document.getElementById('currencyOutput');
    if (!amount || !from || !to || !out) {
        showMessage('Enter amount and currencies', 'warning');
        return;
    }
    const result = await convertCurrencyWithTiming(amount, from, to);
    out.innerHTML = result
        ? `<pre class="report-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`
        : '<p class="empty-state">Conversion failed.</p>';
}

// Currency Conversion with Timing
async function convertCurrencyWithTiming(amount, fromCurrency, toCurrency, date = null) {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const params = {
            ...auth,
            amount: parseFloat(amount),
            from_currency: fromCurrency.toUpperCase(),
            to_currency: toCurrency.toUpperCase()
        };
        if (date) params.date = date;
        
        const result = await callMCPTool('convert_currency_with_timing', params);
        return result;
    } catch (error) {
        console.error('Error converting currency:', error);
        return null;
    }
}

// Enhanced Tax Estimation with Multi-Jurisdiction
async function estimateTaxesEnhanced(taxYear = null, jurisdiction = 'IN') {
    if (!ensureAuthenticated(false)) return;
    const auth = getAuthPayload();
    if (!auth) return;
    
    try {
        const params = {
            ...auth,
            jurisdiction: jurisdiction.toUpperCase()
        };
        if (taxYear) params.tax_year = parseInt(taxYear);
        
        const result = await callMCPTool('estimate_taxes', params);
        return result;
    } catch (error) {
        console.error('Error estimating taxes:', error);
        return null;
    }
}

// Expose handlers for inline HTML onclick attributes and browser caching edge cases
Object.assign(window, {
    openQuickAction,
    showAddExpenseModal,
    showAddIncomeModal,
    showSetBudgetModal,
    showAddBillModal,
    showAddSavingsGoalModal,
    closeModal,
    addExpense,
    addIncome,
    submitBudget,
    initiateGmailConnect,
    refreshSessionStatus,
    sendChatMessage,
    sendExample,
    setExpensePeriod,
    syncGmailBills,
    purgeGmailPromos,
    loadExpenses,
    toggleSpendBreakdown,
    saveBillNotificationSettings,
    runBillRemindersNow,
    syncBankTransactions,
    syncLast30Days,
    loadAccountBalances,
    loadFinancialReports,
    loadTaxEstimates,
    runCurrencyConversion,
    exportAuditCsv,
    refreshAuditView,
    resetAuditOffset,
    loadAuditLogs,
    lookupTicker,
});
