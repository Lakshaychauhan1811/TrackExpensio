"""
FIXED Gmail Expense Detector for India
=======================================
The original scanner was too strict.
This version catches ALL Indian bank/payment emails correctly.

Replace your existing gmail_expense_detector.py with this file.
"""

import os, re, base64, logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── TOKEN STORE (use MongoDB in production) ───────────────────────────────────
TOKEN_STORE = {}  # {user_id: token_data}

# ── GMAIL SCOPES ──────────────────────────────────────────────────────────────
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ── AMOUNT PATTERNS (Indian formats) ─────────────────────────────────────────
AMOUNT_PATTERNS = [
    r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)',
    r'([\d,]+(?:\.\d{1,2})?)\s*(?:Rs\.?|INR|₹)',
    r'(?:amount|amt)\s*(?:of|:)?\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'(?:debited|deducted|charged|spent|paid)\s+(?:with\s+)?(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)\s+(?:has been|is|was)',
    r'transaction\s+of\s+(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'payment\s+of\s+(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'([\d,]+(?:\.\d{1,2})?)\s+(?:has been\s+)?debited',
]

# ── MERCHANT PATTERNS ─────────────────────────────────────────────────────────
MERCHANT_PATTERNS = [
    r'(?:at|to|towards|for|merchant[:\s]+|payee[:\s]+|paid to[:\s]+)\s*([A-Za-z0-9][A-Za-z0-9\s&\'\-\.]{1,40}?)(?:\s+on|\s+for|\s+via|\s+ref|\.|,|\n|$)',
    r'(?:UPI|NEFT|IMPS|POS)\s*[-/]\s*([A-Za-z0-9][A-Za-z0-9\s&\'\-]{2,30}?)(?:\s+on|\s+ref|\.|,|\n|$)',
    r'purchase\s+at\s+([A-Za-z0-9][A-Za-z0-9\s&\'\-]{2,30}?)(?:\s+on|\.|,|\n|$)',
    r'txn\s+(?:at|to)\s+([A-Za-z0-9][A-Za-z0-9\s&\'\-]{2,30}?)(?:\s+on|\.|,|\n|$)',
]

# ── DEBIT KEYWORDS ────────────────────────────────────────────────────────────
DEBIT_KEYWORDS = [
    'debited', 'deducted', 'spent', 'payment', 'purchase',
    'charged', 'withdrawn', 'debit', 'paid', 'transaction',
    'upi payment', 'neft', 'imps', 'pos transaction',
    'bill payment', 'auto debit', 'mandate',
]

CREDIT_KEYWORDS = [
    'credited', 'received', 'cashback', 'refund',
    'reward', 'interest credited', 'salary',
]

# ── BANK SENDERS (Gmail filter) ───────────────────────────────────────────────
BANK_SENDER_PATTERNS = [
    'hdfc', 'sbi', 'icici', 'axis', 'kotak', 'yes bank',
    'paytm', 'phonepe', 'gpay', 'amazon pay', 'mobikwik',
    'alerts', 'notify', 'noreply', 'donotreply',
    'transaction', 'statement', 'bank',
]

# ── CATEGORY MAP ──────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    'swiggy': 'Food', 'zomato': 'Food', 'dominos': 'Food', 'kfc': 'Food',
    'mcdonald': 'Food', 'subway': 'Food', 'pizza': 'Food', 'blinkit': 'Food',
    'bigbasket': 'Food', 'dunzo': 'Food', 'zepto': 'Food', 'instamart': 'Food',
    'amazon': 'Shopping', 'flipkart': 'Shopping', 'myntra': 'Shopping',
    'meesho': 'Shopping', 'nykaa': 'Shopping', 'ajio': 'Shopping',
    'uber': 'Transportation', 'ola': 'Transportation', 'rapido': 'Transportation',
    'metro': 'Transportation', 'irctc': 'Transportation', 'petrol': 'Transportation',
    'fuel': 'Transportation', 'indigo': 'Transportation', 'spicejet': 'Transportation',
    'airtel': 'Bills', 'jio': 'Bills', 'bsnl': 'Bills', 'vi ': 'Bills',
    'vodafone': 'Bills', 'electricity': 'Bills', 'gas': 'Bills',
    'broadband': 'Bills', 'dth': 'Bills', 'recharge': 'Bills',
    'netflix': 'Entertainment', 'hotstar': 'Entertainment', 'spotify': 'Entertainment',
    'prime': 'Entertainment', 'zee5': 'Entertainment', 'pvr': 'Entertainment',
    'inox': 'Entertainment', 'bookmyshow': 'Entertainment',
    'hospital': 'Healthcare', 'pharmacy': 'Healthcare', 'apollo': 'Healthcare',
    'medplus': 'Healthcare', 'netmeds': 'Healthcare', 'health': 'Healthcare',
    'clinic': 'Healthcare', 'doctor': 'Healthcare',
    'school': 'Education', 'college': 'Education', 'udemy': 'Education',
    'coursera': 'Education', 'fees': 'Education', 'tuition': 'Education',
    'makemytrip': 'Travel', 'goibibo': 'Travel', 'oyo': 'Travel', 'hotel': 'Travel',
}


def _safe_decode(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _get_body(msg: dict) -> str:
    """Extract all text from a Gmail message."""
    parts = []

    def walk(payload):
        mime = payload.get('mimeType', '')
        if mime == 'text/plain':
            data = payload.get('body', {}).get('data', '')
            if data:
                parts.append(_safe_decode(data))
        elif mime == 'text/html':
            data = payload.get('body', {}).get('data', '')
            if data:
                html = _safe_decode(data)
                # Strip HTML tags
                clean = re.sub(r'<[^>]+>', ' ', html)
                parts.append(clean)
        for part in payload.get('parts', []):
            walk(part)

    walk(msg.get('payload', {}))
    return ' '.join(parts)


def _get_header(msg: dict, name: str) -> str:
    for h in msg.get('payload', {}).get('headers', []):
        if h['name'].lower() == name.lower():
            return h['value']
    return ''


def _extract_amount(text: str) -> Optional[float]:
    for pattern in AMOUNT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                amount = float(match.group(1).replace(',', ''))
                if 1 <= amount <= 10_000_000:  # Rs 1 to 1 crore
                    return amount
            except (ValueError, IndexError):
                continue
    return None


def _extract_merchant(text: str, subject: str) -> str:
    combined = subject + ' ' + text
    for pattern in MERCHANT_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            m = match.group(1).strip()
            if len(m) > 2 and m.lower() not in (
                'your', 'the', 'this', 'has', 'been', 'from', 'with',
                'using', 'account', 'bank', 'card', 'rupees', 'inr',
            ):
                return m.title()[:50]

    # Fallback: check known merchants in text
    text_lower = (text + ' ' + subject).lower()
    for keyword in CATEGORY_MAP:
        if keyword in text_lower:
            return keyword.title()

    # Fallback: extract from subject
    subject_clean = re.sub(
        r'(?i)(alert|notification|transaction|debit|credit|bank|hdfc|sbi|icici|axis|kotak)', 
        '', subject
    ).strip()
    if subject_clean and len(subject_clean) > 3:
        return subject_clean[:40]

    return 'Bank Transaction'


def _guess_category(text: str, merchant: str) -> str:
    combined = (text + ' ' + merchant).lower()
    for keyword, cat in CATEGORY_MAP.items():
        if keyword in combined:
            return cat
    return 'Other'


def _is_debit(text: str, subject: str) -> bool:
    combined = (text + ' ' + subject).lower()
    has_debit  = any(k in combined for k in DEBIT_KEYWORDS)
    has_credit = any(k in combined for k in CREDIT_KEYWORDS)
    # If both found, trust debit keywords more
    if has_debit and not has_credit:
        return True
    if has_credit and not has_debit:
        return False
    return has_debit  # If both, lean toward debit


def _get_date(msg: dict) -> str:
    date_str = _get_header(msg, 'date')
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')


# ── GMAIL AUTH ────────────────────────────────────────────────────────────────

def get_gmail_auth_url(user_id: str, redirect_uri: str) -> str:
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=GMAIL_SCOPES,
        redirect_uri=redirect_uri,
    )
    url, _ = flow.authorization_url(
        access_type='offline',
        state=user_id,
        prompt='consent',
        include_granted_scopes='true',
    )
    return url


def handle_gmail_callback(code: str, user_id: str, redirect_uri: str) -> dict:
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=GMAIL_SCOPES,
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    TOKEN_STORE[user_id] = {
        'token':         creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri':     creds.token_uri,
        'client_id':     creds.client_id,
        'client_secret': creds.client_secret,
        'scopes':        list(creds.scopes),
    }
    return {'status': 'connected'}


def _get_service(user_id: str):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    data = TOKEN_STORE.get(user_id)
    if not data:
        raise ValueError('Gmail not connected for this user.')

    creds = Credentials(**{k: data[k] for k in
        ['token','refresh_token','token_uri','client_id','client_secret','scopes']})
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_STORE[user_id]['token'] = creds.token

    return build('gmail', 'v1', credentials=creds)


# ── MAIN FETCH FUNCTION ───────────────────────────────────────────────────────

def fetch_gmail_expenses(user_id: str, days_back: int = 30, max_emails: int = 100) -> dict:
    """
    Fetch and parse all bank transaction emails from Gmail.
    FIXED: Much more aggressive search + better Indian format parsing.
    """
    try:
        service = _get_service(user_id)
    except ValueError as e:
        return {'error': str(e), 'connected': False}

    after = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')

    # FIXED: Much broader search query — catches more bank emails
    queries = [
        # Query 1: Bank transaction keywords
        f'after:{after} (debited OR deducted OR "has been paid" OR "payment of" OR "transaction of" OR "spent on" OR "UPI payment" OR "NEFT" OR "IMPS")',
        # Query 2: Bank sender domains
        f'after:{after} (from:alerts OR from:notify OR from:noreply OR from:transaction OR from:bank)',
        # Query 3: Common bank keywords in subject
        f'after:{after} subject:(debit OR transaction OR payment OR alert OR spent)',
    ]

    seen_ids = set()
    all_messages = []

    for q in queries:
        try:
            result = service.users().messages().list(
                userId='me', q=q, maxResults=max_emails
            ).execute()
            for m in result.get('messages', []):
                if m['id'] not in seen_ids:
                    seen_ids.add(m['id'])
                    all_messages.append(m)
        except Exception as e:
            logger.warning(f'Query failed: {q} — {e}')

    expenses = []
    skipped  = 0
    errors   = 0

    for msg_ref in all_messages[:max_emails]:
        try:
            msg     = service.users().messages().get(
                userId='me', id=msg_ref['id'], format='full'
            ).execute()
            subject = _get_header(msg, 'subject')
            sender  = _get_header(msg, 'from').lower()
            body    = _get_body(msg)
            full    = subject + ' ' + body

            # Skip non-debit emails
            if not _is_debit(body, subject):
                skipped += 1
                continue

            amount = _extract_amount(full)
            if not amount:
                skipped += 1
                continue

            merchant = _extract_merchant(body, subject)
            category = _guess_category(full, merchant)
            date     = _get_date(msg)

            # Determine bank from sender
            bank = 'Unknown Bank'
            for b in ['hdfc','sbi','icici','axis','kotak','paytm','phonepe']:
                if b in sender:
                    bank = b.upper()
                    break

            expenses.append({
                'merchant':      merchant,
                'amount':        round(amount, 2),
                'category':      category,
                'date':          date,
                'source':        'gmail',
                'bank':          bank,
                'email_subject': subject[:100],
                'currency':      'INR',
            })

        except Exception as e:
            logger.warning(f'Error parsing email: {e}')
            errors += 1

    total  = sum(e['amount'] for e in expenses)
    by_cat: dict = {}
    for e in expenses:
        by_cat[e['category']] = round(by_cat.get(e['category'], 0) + e['amount'], 2)

    return {
        'connected':      True,
        'emails_scanned': len(all_messages),
        'expenses_found': len(expenses),
        'skipped':        skipped,
        'errors':         errors,
        'expenses':       expenses,
        'total_amount':   round(total, 2),
        'by_category':    by_cat,
        'period_days':    days_back,
        'scanned_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def is_gmail_connected(user_id: str) -> bool:
    return user_id in TOKEN_STORE


def disconnect_gmail(user_id: str) -> bool:
    if user_id in TOKEN_STORE:
        del TOKEN_STORE[user_id]
        return True
    return False


def parse_sms_expense(sms_text: str) -> Optional[dict]:
    if not _is_debit(sms_text, ''):
        return None
    amount = _extract_amount(sms_text)
    if not amount:
        return None
    merchant = _extract_merchant(sms_text, '')
    return {
        'merchant': merchant,
        'amount':   round(amount, 2),
        'category': _guess_category(sms_text, merchant),
        'date':     datetime.now().strftime('%Y-%m-%d'),
        'source':   'sms',
        'currency': 'INR',
    }
