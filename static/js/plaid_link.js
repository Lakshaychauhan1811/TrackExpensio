/**
 * Plaid Link integration for TrackExpensio bank sync.
 */

async function loadPlaidStatus() {
    const el = document.getElementById('plaidStatus');
    if (!el) return;
    try {
        const res = await fetch('/api/integrations/status');
        const data = await res.json();
        const plaid = data.plaid || {};
        const email = data.notifications?.email;
        if (!plaid.configured) {
            el.innerHTML = `
                <p class="hint-text warning">
                    <i class="fas fa-info-circle"></i>
                    Plaid is not configured. Add <code>PLAID_CLIENT_ID</code> and <code>PLAID_SECRET</code> to your <code>.env</code> file
                    (use <code>PLAID_ENV=sandbox</code> for testing).
                </p>`;
            const btn = document.getElementById('plaidLinkBtn');
            if (btn) btn.disabled = true;
            return;
        }
        const regions = (plaid.active_country_codes || plaid.supported_country_codes || ['US']).join(', ');
        el.innerHTML = `
            <p class="hint-text success">
                <i class="fas fa-check-circle"></i>
                Plaid ready (${plaid.env || 'sandbox'}). Regions: <strong>${regions}</strong>.
                Click <strong>Connect Bank</strong> to link an account.
            </p>
            ${plaid.india_supported === false ? `
            <p class="hint-text warning">
                <i class="fas fa-globe-asia"></i>
                Indian banks are not available via Plaid — use manual entry, Gmail sync, or bill upload (see below).
            </p>` : ''}
            ${email ? '' : '<p class="hint-text">Set SMTP_* in .env for bill email alerts.</p>'}`;
    } catch (e) {
        el.innerHTML = '<p class="hint-text">Could not load integration status.</p>';
    }
}

async function openPlaidLink() {
    if (typeof ensureAuthenticated === 'function' && !ensureAuthenticated()) return;
    const auth = typeof getAuthPayload === 'function' ? getAuthPayload() : null;
    if (!auth) {
        if (typeof showMessage === 'function') showMessage('Sign in required', 'warning');
        return;
    }
    if (typeof Plaid === 'undefined') {
        if (typeof showMessage === 'function') showMessage('Plaid script failed to load', 'error');
        return;
    }

    try {
        if (typeof showMessage === 'function') showMessage('Opening Plaid Link…', 'info');
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/bank/plaid/link-token?${params}`);
        const data = await res.json();
        if (!res.ok || !data.link_token) {
            throw new Error(data.message || data.detail || 'Could not get link token');
        }

        const handler = Plaid.create({
            token: data.link_token,
            onSuccess: async (publicToken, metadata) => {
                const institution = metadata.institution || {};
                const account = (metadata.accounts && metadata.accounts[0]) || {};
                const exchangeRes = await fetch('/api/bank/plaid/exchange', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ...auth,
                        public_token: publicToken,
                        bank_name: institution.name || 'Linked Bank',
                        account_type: account.subtype || account.type || 'Checking',
                        account_number_last4: account.mask || '0000',
                    }),
                });
                const exchangeData = await exchangeRes.json();
                if (!exchangeRes.ok) {
                    throw new Error(exchangeData.detail || exchangeData.message || 'Exchange failed');
                }
                if (typeof showMessage === 'function') {
                    showMessage(exchangeData.message || 'Bank connected!', 'success');
                }
                if (typeof loadBankConnections === 'function') await loadBankConnections();
                if (typeof loadAccountBalances === 'function') loadAccountBalances();
                if (typeof loadExpenses === 'function') loadExpenses();
                if (typeof loadDashboardData === 'function') loadDashboardData();
            },
            onExit: (err) => {
                if (err && err.error_message && typeof showMessage === 'function') {
                    showMessage(err.error_message, 'warning');
                }
            },
        });
        handler.open();
    } catch (error) {
        console.error('Plaid Link error:', error);
        if (typeof showMessage === 'function') showMessage(error.message || 'Plaid Link failed', 'error');
    }
}

async function loadBankConnections() {
    const listEl = document.getElementById('bankConnectionsList');
    const panel = document.getElementById('bankSyncPanel');
    const select = document.getElementById('bankConnectionSelect');
    if (!listEl) return;

    const auth = typeof getAuthPayload === 'function' ? getAuthPayload() : null;
    if (!auth) {
        listEl.innerHTML = '<p class="empty-state">Sign in to connect a bank.</p>';
        return;
    }

    try {
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/bank/connections?${params}`);
        const data = await res.json();
        const connections = data.connections || [];

        if (!connections.length) {
            listEl.innerHTML = '<p class="empty-state">No banks connected yet. Use Connect Bank above.</p>';
            if (panel) panel.style.display = 'none';
            return;
        }

        listEl.innerHTML = connections.map(c => `
            <div class="bank-connection-card">
                <div>
                    <strong>${escapeHtml(c.bank_name || 'Bank')}</strong>
                    <span class="bank-meta">${escapeHtml(c.account_type || '')} · •••• ${escapeHtml(c.account_number_last4 || '')}</span>
                </div>
                ${c.plaid_item_id ? '<span class="bank-badge">Plaid</span>' : ''}
            </div>
        `).join('');

        if (panel && select) {
            panel.style.display = 'block';
            select.innerHTML = connections.map(c =>
                `<option value="${c.id}">${c.bank_name} (${c.account_number_last4})</option>`
            ).join('');
            const today = new Date();
            const start = document.getElementById('bankSyncStart');
            const end = document.getElementById('bankSyncEnd');
            if (start && !start.value) {
                const d = new Date(today.getFullYear(), today.getMonth(), 1);
                start.value = d.toISOString().split('T')[0];
            }
            if (end && !end.value) {
                end.value = today.toISOString().split('T')[0];
            }
        }
    } catch (e) {
        console.error(e);
        listEl.innerHTML = '<p class="empty-state">Could not load bank connections.</p>';
    }
}

async function syncLast30Days() {
    const select = document.getElementById('bankConnectionSelect');
    const end = document.getElementById('bankSyncEnd');
    const start = document.getElementById('bankSyncStart');
    if (!select?.value) {
        showMessage('Connect a bank first', 'warning');
        return;
    }
    const today = new Date();
    const d = new Date(today);
    d.setDate(d.getDate() - 30);
    if (start) start.value = d.toISOString().split('T')[0];
    if (end) end.value = today.toISOString().split('T')[0];
    await syncBankTransactions();
}

async function loadAccountBalances() {
    if (typeof ensureAuthenticated === 'function' && !ensureAuthenticated()) return;
    const auth = getAuthPayload();
    const connectionId = document.getElementById('bankConnectionSelect')?.value;
    const listEl = document.getElementById('bankBalancesList');
    if (!auth || !connectionId) {
        showMessage('Select a connected bank first', 'warning');
        return;
    }
    if (listEl) listEl.innerHTML = '<p class="hint-text">Loading balances…</p>';
    try {
        const params = new URLSearchParams({ ...auth, connection_id: connectionId });
        const res = await fetch(`/api/bank/plaid/balances?${params}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        const accounts = data.accounts || [];
        if (!accounts.length) {
            if (listEl) listEl.innerHTML = '<p class="hint-text">No accounts returned.</p>';
            return;
        }
        if (listEl) {
            listEl.innerHTML = accounts.map(a => {
                const cur = a.currency === 'USD' ? '$' : (a.currency === 'INR' ? '₹' : a.currency + ' ');
                const bal = a.balance_current != null ? `${cur}${Number(a.balance_current).toLocaleString()}` : '—';
                return `<div class="bank-connection-card">
                    <div><strong>${escapeHtml(a.name)}</strong>
                    <span class="bank-meta">${escapeHtml(a.subtype || a.type || '')} · •••• ${escapeHtml(a.mask || '')}</span></div>
                    <span class="bank-badge">${bal}</span>
                </div>`;
            }).join('');
        }
    } catch (e) {
        if (listEl) listEl.innerHTML = '';
        showMessage(e.message || 'Could not load balances', 'error');
    }
}

async function syncBankTransactions() {
    if (typeof ensureAuthenticated === 'function' && !ensureAuthenticated()) return;
    const auth = getAuthPayload();
    const connectionId = document.getElementById('bankConnectionSelect')?.value;
    const start = document.getElementById('bankSyncStart')?.value;
    const end = document.getElementById('bankSyncEnd')?.value;
    const resultEl = document.getElementById('bankSyncResult');
    if (!auth || !connectionId || !start || !end) {
        showMessage('Select connection and date range', 'warning');
        return;
    }
    if (resultEl) resultEl.textContent = 'Syncing…';
    try {
        const res = await fetch('/api/bank/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...auth,
                connection_id: connectionId,
                start_date: start,
                end_date: end,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        const msg = data.message || `Synced ${data.transactions_synced ?? 0} transactions`;
        if (resultEl) resultEl.textContent = data.note ? `${msg} — ${data.note}` : msg;
        showMessage(msg, 'success');
        loadExpenses();
        loadDashboardData();
    } catch (e) {
        if (resultEl) resultEl.textContent = '';
        showMessage(e.message || 'Sync failed', 'error');
    }
}

function escapeHtml(text) {
    if (typeof window.escapeHtml === 'function' && window.escapeHtml !== escapeHtml) {
        return window.escapeHtml(text);
    }
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

Object.assign(window, {
    openPlaidLink,
    loadBankConnections,
    syncBankTransactions,
    syncLast30Days,
    loadAccountBalances,
    loadPlaidStatus,
});
