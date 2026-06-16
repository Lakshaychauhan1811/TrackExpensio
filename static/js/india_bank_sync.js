/**
 * Indian bank sync: Account Aggregator, Gmail transaction emails, SMS parsing.
 */

function indiaEscapeHtml(text) {
    if (typeof escapeHtml === 'function') return escapeHtml(text);
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

function indiaAuthPayload() {
    if (typeof getAuthPayload === 'function') return getAuthPayload();
    return null;
}

function indiaEnsureAuth(showToast = true) {
    if (typeof ensureAuthenticated === 'function') return ensureAuthenticated(showToast);
    return true;
}

function indiaShowMessage(msg, type) {
    if (typeof showMessage === 'function') showMessage(msg, type);
    else alert(msg);
}

async function loadAAStatus() {
    const el = document.getElementById('aaStatus');
    const listEl = document.getElementById('aaConnectionsList');
    if (!el) return;
    try {
        const res = await fetch('/api/aa/status');
        const data = await res.json();
        const mode = data.mode === 'sandbox' ? 'Sandbox demo' : 'Live';
        el.innerHTML = `
            <p class="hint-text success">
                <i class="fas fa-check-circle"></i>
                Account Aggregator ready (${mode}) — providers: ${(data.providers || []).join(', ')}
            </p>
            <p class="hint-text">Flow: Connect Bank → AA Consent → Approve → Fetch Transactions → MongoDB</p>`;
    } catch (e) {
        el.innerHTML = '<p class="hint-text">AA status unavailable.</p>';
    }
    await loadAAConnections();
}

async function loadAAConnections() {
    const listEl = document.getElementById('aaConnectionsList');
    if (!listEl) return;
    if (!indiaEnsureAuth(false)) {
        listEl.innerHTML = '<p class="empty-state">Sign in with Google to use Account Aggregator.</p>';
        return;
    }
    const auth = indiaAuthPayload();
    if (!auth) {
        listEl.innerHTML = '<p class="empty-state">Session required. Please refresh the page.</p>';
        return;
    }
    try {
        const params = new URLSearchParams(auth);
        const res = await fetch(`/api/aa/connections?${params}`);
        const data = await res.json();
        const connections = data.connections || [];
        if (!connections.length) {
            listEl.innerHTML = '<p class="empty-state">No Indian banks linked via Account Aggregator yet.</p>';
            return;
        }
        listEl.innerHTML = connections.map(c => {
            const cid = indiaEscapeHtml(c.consent_id || '');
            const id = indiaEscapeHtml(c.id || '');
            return `
            <div class="bank-connection-card">
                <div>
                    <strong>${indiaEscapeHtml(c.bank_name || 'Bank')}</strong>
                    <span class="bank-meta">${indiaEscapeHtml(c.provider || 'aa')} · ${indiaEscapeHtml(c.status || '')}</span>
                </div>
                <div class="aa-card-actions">
                    ${c.status === 'active' ? `
                    <button type="button" class="btn-secondary btn-sm" data-aa-sync="${id}">
                        <i class="fas fa-sync"></i> Sync
                    </button>` : `
                    <button type="button" class="btn-secondary btn-sm" data-aa-approve="${cid}">
                        Approve Consent
                    </button>`}
                </div>
            </div>`;
        }).join('');
        listEl.querySelectorAll('[data-aa-sync]').forEach(btn => {
            btn.addEventListener('click', () => syncAABank(btn.dataset.aaSync));
        });
        listEl.querySelectorAll('[data-aa-approve]').forEach(btn => {
            btn.addEventListener('click', () => approveAAConsent(btn.dataset.aaApprove));
        });
    } catch (e) {
        listEl.innerHTML = '<p class="empty-state">Could not load AA connections.</p>';
    }
}

async function startAAConsent() {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    if (!auth) return;
    const provider = document.getElementById('aaProvider')?.value || 'setu';
    const bank = document.getElementById('aaBankSelect')?.value || 'HDFC Bank';
    const modal = document.getElementById('aaConsentModal');
    try {
        const res = await fetch('/api/aa/consent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, provider, bank_name: bank }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message || 'Consent failed');
        window._pendingAAConsentId = data.consent_id;
        if (modal) {
            document.getElementById('aaConsentBank').textContent = data.bank_name;
            document.getElementById('aaConsentProvider').textContent = data.provider;
            modal.classList.add('active');
        } else {
            await approveAAConsent(data.consent_id);
        }
        indiaShowMessage(data.message || 'AA consent created', 'success');
        await loadAAConnections();
    } catch (e) {
        indiaShowMessage(e.message || 'Failed to start AA consent', 'error');
    }
}

async function approveAAConsent(consentId) {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    const cid = consentId || window._pendingAAConsentId;
    if (!auth || !cid) return;
    try {
        const res = await fetch('/api/aa/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, consent_id: cid, approved: true }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        closeModal('aaConsentModal');
        window._pendingAAConsentId = null;
        indiaShowMessage(data.message || 'Bank linked via Account Aggregator', 'success');
        await loadAAConnections();
        if (typeof loadDashboardData === 'function') loadDashboardData();
    } catch (e) {
        indiaShowMessage(e.message || 'Consent approval failed', 'error');
    }
}

async function syncAABank(connectionId) {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    if (!auth || !connectionId) return;
    indiaShowMessage('Fetching transactions from Account Aggregator…', 'info');
    try {
        const res = await fetch('/api/aa/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, connection_id: connectionId }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        indiaShowMessage(data.message || `Synced ${data.imported} transactions`, 'success');
        if (typeof loadExpenses === 'function') loadExpenses();
        if (typeof loadDashboardData === 'function') loadDashboardData();
        await loadAAConnections();
    } catch (e) {
        indiaShowMessage(e.message || 'AA sync failed', 'error');
    }
}

async function syncGmailBankTransactions() {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    if (!auth) return;
    const btn = document.getElementById('gmailBankSyncBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning bank emails…';
    }
    try {
        const res = await fetch('/api/gmail/sync-bank-transactions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, max_messages: 50 }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        const out = document.getElementById('gmailBankResult');
        if (out) {
            out.innerHTML = (data.items || []).slice(0, 8).map(i =>
                `<div class="sms-preview-item">${indiaEscapeHtml(i.date)} · ${indiaEscapeHtml(i.merchant)} · ₹${i.amount}</div>`
            ).join('') || '<span class="hint-text">No new bank emails found.</span>';
        }
        indiaShowMessage(data.message || `Imported ${data.imported} transactions`, 'success');
        if (typeof loadExpenses === 'function') loadExpenses();
        if (typeof loadDashboardData === 'function') loadDashboardData();
    } catch (e) {
        indiaShowMessage(e.message || 'Gmail bank sync failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-envelope-open-text"></i> Scan Bank Emails';
        }
    }
}

async function previewSMSExpenses() {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    const text = document.getElementById('smsInput')?.value?.trim();
    const preview = document.getElementById('smsPreview');
    if (!auth || !text) {
        indiaShowMessage('Paste an SMS alert first', 'warning');
        return;
    }
    try {
        const res = await fetch('/api/sms/parse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, sms_text: text, save: false }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        const items = data.parsed || [];
        if (preview) {
            preview.innerHTML = items.map(p => `
                <div class="sms-preview-item">
                    <strong>${indiaEscapeHtml(p.merchant)}</strong> · ₹${p.amount} · ${indiaEscapeHtml(p.category)}
                    <div class="hint-text">${indiaEscapeHtml(p.note || '').slice(0, 80)}</div>
                </div>
            `).join('');
        }
        indiaShowMessage(`Parsed ${items.length} transaction(s)`, 'success');
    } catch (e) {
        if (preview) preview.innerHTML = '';
        indiaShowMessage(e.message || 'SMS parse failed', 'error');
    }
}

async function saveSMSExpenses() {
    if (!indiaEnsureAuth()) return;
    const auth = indiaAuthPayload();
    const text = document.getElementById('smsInput')?.value?.trim();
    if (!auth || !text) {
        indiaShowMessage('Paste an SMS alert first', 'warning');
        return;
    }
    try {
        const res = await fetch('/api/sms/parse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...auth, sms_text: text, save: true }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message);
        indiaShowMessage(data.message || `Saved ${data.imported} expense(s)`, 'success');
        document.getElementById('smsInput').value = '';
        document.getElementById('smsPreview').innerHTML = '';
        if (typeof loadExpenses === 'function') loadExpenses();
        if (typeof loadDashboardData === 'function') loadDashboardData();
    } catch (e) {
        indiaShowMessage(e.message || 'Failed to save SMS expenses', 'error');
    }
}

function initIndiaBankSync() {
    loadAAStatus();
}

Object.assign(window, {
    loadAAStatus,
    loadAAConnections,
    startAAConsent,
    approveAAConsent,
    syncAABank,
    syncGmailBankTransactions,
    previewSMSExpenses,
    saveSMSExpenses,
    initIndiaBankSync,
});
