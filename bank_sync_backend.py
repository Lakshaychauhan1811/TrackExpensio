"""
TrackExpensio — Complete Bank Sync Backend
==========================================
Add ALL of these endpoints to your api.py

IMPORTS TO ADD AT TOP OF api.py:
---------------------------------
from bank_sync_backend import setup_bank_routes
setup_bank_routes(app)

OR copy-paste each endpoint directly into api.py
"""

import os, re, base64, logging, csv, io, json
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── IN-MEMORY TOKEN STORE (replace with MongoDB in production) ────────────────
GMAIL_TOKENS   = {}   # {user_id: {token, refresh_token, ...}}
PLAID_TOKENS   = {}   # {user_id: access_token}
AA_CONSENTS    = {}   # {user_id: consent_id}


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SMSRequest(BaseModel):
    sms_text: str

class ExpensesImport(BaseModel):
    expenses: list

class AAConsentRequest(BaseModel):
    phone: str

class PlaidExchangeRequest(BaseModel):
    public_token: str


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

AMOUNT_PATTERNS = [
    r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)',
    r'([\d,]+(?:\.\d{1,2})?)\s*(?:Rs\.?|INR|₹)',
    r'(?:amount|amt)\s*(?:of|:)?\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'(?:debited|deducted|charged|spent|paid)\s+(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    r'([\d,]+(?:\.\d{1,2})?)\s+(?:has been\s+)?debited',
    r'transaction\s+of\s+(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
]

CATEGORY_MAP = {
    'swiggy':'Food','zomato':'Food','dominos':'Food','kfc':'Food','mcdonald':'Food',
    'blinkit':'Food','bigbasket':'Food','dunzo':'Food','zepto':'Food','instamart':'Food',
    'amazon':'Shopping','flipkart':'Shopping','myntra':'Shopping','meesho':'Shopping',
    'nykaa':'Shopping','ajio':'Shopping','snapdeal':'Shopping',
    'uber':'Transportation','ola':'Transportation','rapido':'Transportation',
    'metro':'Transportation','irctc':'Transportation','petrol':'Transportation',
    'fuel':'Transportation','indigo':'Transportation','spicejet':'Transportation',
    'airtel':'Bills','jio':'Bills','bsnl':'Bills','vodafone':'Bills',
    'electricity':'Bills','gas':'Bills','broadband':'Bills','recharge':'Bills',
    'netflix':'Entertainment','hotstar':'Entertainment','spotify':'Entertainment',
    'prime':'Entertainment','pvr':'Entertainment','bookmyshow':'Entertainment',
    'hospital':'Healthcare','pharmacy':'Healthcare','apollo':'Healthcare',
    'medplus':'Healthcare','netmeds':'Healthcare','clinic':'Healthcare',
    'school':'Education','college':'Education','udemy':'Education',
    'coursera':'Education','fees':'Education','tuition':'Education',
    'makemytrip':'Travel','goibibo':'Travel','oyo':'Travel','hotel':'Travel',
}

DEBIT_WORDS  = ['debited','deducted','spent','payment','purchase','charged',
                'withdrawn','debit','paid','transaction','upi payment','neft','imps',
                'bill payment','auto debit','mandate','pos']
CREDIT_WORDS = ['credited','received','cashback','refund','reward','interest credited','salary']


def _extract_amount(text: str) -> Optional[float]:
    for p in AMOUNT_PATTERNS:
        for m in re.finditer(p, text, re.IGNORECASE):
            try:
                a = float(m.group(1).replace(',',''))
                if 1 <= a <= 10_000_000:
                    return a
            except (ValueError, IndexError):
                continue
    return None


def _extract_merchant(text: str, subject: str = '') -> str:
    combined = subject + ' ' + text
    patterns = [
        r'(?:at|to|towards|for|merchant[:\s]+|payee[:\s]+|paid to[:\s]+)\s*([A-Za-z0-9][A-Za-z0-9\s&\'\-\.]{1,40}?)(?:\s+on|\s+for|\.|,|\n|$)',
        r'(?:UPI|NEFT|IMPS|POS)\s*[-/]\s*([A-Za-z0-9][A-Za-z0-9\s&\'\-]{2,30}?)(?:\s+on|\.|,|\n|$)',
    ]
    skip = {'your','the','this','has','been','from','with','using','account',
            'bank','card','rupees','inr','rs','amount','transaction'}
    for p in patterns:
        m = re.search(p, combined, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) > 2 and name.lower() not in skip:
                return name.title()[:50]

    text_lower = combined.lower()
    for kw in CATEGORY_MAP:
        if kw in text_lower:
            return kw.title()

    return 'Bank Transaction'


def _guess_category(text: str, merchant: str = '') -> str:
    combined = (text + ' ' + merchant).lower()
    for kw, cat in CATEGORY_MAP.items():
        if kw in combined:
            return cat
    return 'Other'


def _is_debit(text: str, subject: str = '') -> bool:
    combined = (text + ' ' + subject).lower()
    has_debit  = any(k in combined for k in DEBIT_WORDS)
    has_credit = any(k in combined for k in CREDIT_WORDS)
    if has_credit and not has_debit:
        return False
    return has_debit


def _safe_b64(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _get_user(request: Request) -> str:
    return (request.session.get('session_id')
            or request.session.get('user_id')
            or 'guest')


# ══════════════════════════════════════════════════════════════════════════════
# GMAIL — AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _gmail_service(user_id: str):
    """Build authenticated Gmail service."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        data = GMAIL_TOKENS.get(user_id)
        if not data:
            raise ValueError('Gmail not connected')

        creds = Credentials(
            token=data['token'],
            refresh_token=data.get('refresh_token'),
            token_uri=data.get('token_uri','https://oauth2.googleapis.com/token'),
            client_id=data['client_id'],
            client_secret=data['client_secret'],
            scopes=data.get('scopes',['https://www.googleapis.com/auth/gmail.readonly']),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            GMAIL_TOKENS[user_id]['token'] = creds.token

        return build('gmail','v1',credentials=creds)
    except ImportError:
        raise HTTPException(503, detail='Gmail libraries not installed. Run: pip install google-auth google-auth-oauthlib google-api-python-client')


def _gmail_body(msg: dict) -> str:
    parts = []
    def walk(payload):
        mime = payload.get('mimeType','')
        if mime == 'text/plain':
            d = payload.get('body',{}).get('data','')
            if d: parts.append(_safe_b64(d))
        elif mime == 'text/html':
            d = payload.get('body',{}).get('data','')
            if d: parts.append(re.sub(r'<[^>]+>',' ',_safe_b64(d)))
        for p in payload.get('parts',[]):
            walk(p)
    walk(msg.get('payload',{}))
    return ' '.join(parts)


def _gmail_header(msg: dict, name: str) -> str:
    for h in msg.get('payload',{}).get('headers',[]):
        if h['name'].lower() == name.lower():
            return h['value']
    return ''


def _gmail_date(msg: dict) -> str:
    ds = _gmail_header(msg, 'date')
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(ds).strftime('%Y-%m-%d')
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')


# ══════════════════════════════════════════════════════════════════════════════
# CSV BANK STATEMENT PARSERS
# ══════════════════════════════════════════════════════════════════════════════

BANK_FORMATS = {
    'hdfc': {
        'date':'Date','narration':'Narration',
        'debit':'Withdrawal Amt.','credit':'Deposit Amt.',
        'date_format':'%d/%m/%y',
    },
    'sbi': {
        'date':'Txn Date','narration':'Description',
        'debit':'Debit','credit':'Credit',
        'date_format':'%d %b %Y',
    },
    'icici': {
        'date':'Transaction Date','narration':'Transaction Remarks',
        'debit':'Withdrawal Amount (INR )','credit':'Deposit Amount (INR )',
        'date_format':'%d/%m/%Y',
    },
    'axis': {
        'date':'Tran Date','narration':'Particulars',
        'debit':'Debit','credit':'Credit',
        'date_format':'%d-%m-%Y',
    },
    'kotak': {
        'date':'Transaction Date','narration':'Description',
        'debit':'Debit Amount','credit':'Credit Amount',
        'date_format':'%d-%m-%Y',
    },
}


def _clean_narration(n: str) -> str:
    n = re.sub(r'^(UPI[/-]|NEFT[/-]|IMPS[/-]|POS\s+|ATM[/-]|BIL[/-]|\d{6,}/)', '', n, flags=re.IGNORECASE).strip()
    parts = re.split(r'[/\-|]', n)
    return parts[0].strip().title()[:50] if parts else 'Bank Transaction'


def _parse_csv_content(content: str, bank: str = 'hdfc') -> list:
    fmt    = BANK_FORMATS.get(bank.lower(), BANK_FORMATS['hdfc'])
    lines  = content.strip().splitlines()
    start  = 0
    for i, line in enumerate(lines):
        if fmt['date'].lower() in line.lower():
            start = i; break

    reader   = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    expenses = []

    for row in reader:
        try:
            debit_str = row.get(fmt['debit'],'').strip().replace(',','')
            if not debit_str or debit_str in ['0','0.00','-','']:
                continue
            amount = float(debit_str)
            if amount <= 0:
                continue

            date_str = row.get(fmt['date'],'').strip()
            try:
                date = datetime.strptime(date_str, fmt['date_format']).strftime('%Y-%m-%d')
            except Exception:
                date = datetime.now().strftime('%Y-%m-%d')

            narration = row.get(fmt['narration'],'Unknown').strip()
            merchant  = _clean_narration(narration)
            category  = _guess_category(narration, merchant)

            expenses.append({
                'merchant': merchant,
                'amount':   round(amount, 2),
                'date':     date,
                'category': category,
                'source':   f'bank_csv_{bank}',
                'currency': 'INR',
                'narration': narration[:100],
            })
        except Exception:
            continue

    return expenses


# ══════════════════════════════════════════════════════════════════════════════
# MOCK DATA
# ══════════════════════════════════════════════════════════════════════════════

def _mock_indian_txns() -> list:
    return [
        {'merchant':'Swiggy','amount':349,'category':'Food','date':'2026-09-06','bank':'HDFC','currency':'INR'},
        {'merchant':'Amazon India','amount':1299,'category':'Shopping','date':'2026-09-05','bank':'SBI','currency':'INR'},
        {'merchant':'Uber India','amount':156.5,'category':'Transportation','date':'2026-09-04','bank':'ICICI','currency':'INR'},
        {'merchant':'Jio Recharge','amount':349,'category':'Bills','date':'2026-09-03','bank':'Axis','currency':'INR'},
        {'merchant':'Zomato','amount':278,'category':'Food','date':'2026-09-02','bank':'Kotak','currency':'INR'},
        {'merchant':'Apollo Pharmacy','amount':523,'category':'Healthcare','date':'2026-09-01','bank':'HDFC','currency':'INR'},
        {'merchant':'Netflix','amount':649,'category':'Entertainment','date':'2026-08-31','bank':'SBI','currency':'INR'},
        {'merchant':'Blinkit','amount':432,'category':'Food','date':'2026-08-30','bank':'ICICI','currency':'INR'},
        {'merchant':'IRCTC','amount':1820,'category':'Travel','date':'2026-08-29','bank':'Axis','currency':'INR'},
        {'merchant':'Petrol Pump','amount':1500,'category':'Transportation','date':'2026-08-28','bank':'HDFC','currency':'INR'},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SETUP FUNCTION — call this in api.py
# ══════════════════════════════════════════════════════════════════════════════

def setup_bank_routes(app: FastAPI):
    """
    Call this in your api.py:
        from bank_sync_backend import setup_bank_routes
        setup_bank_routes(app)
    """

    # ── PAGE ──────────────────────────────────────────────────────────────────
    @app.get('/bank-sync', response_class=HTMLResponse)
    async def bank_sync_page(request: Request):
        """Serve the Bank Sync HTML page."""
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory='templates')
        try:
            return templates.TemplateResponse('bank_sync.html', {
                'request': request,
                'user': {'name': request.session.get('user_name', 'User')},
            })
        except Exception:
            # Fallback: read file directly
            try:
                with open('templates/bank_sync.html') as f:
                    return HTMLResponse(f.read())
            except FileNotFoundError:
                return HTMLResponse('<h2>Bank Sync page not found. Copy bank_sync_frontend.html to templates/bank_sync.html</h2>')


    # ══════════════════════════════════════════════════════
    # GMAIL ENDPOINTS
    # ══════════════════════════════════════════════════════

    @app.get('/bank/gmail/connect')
    async def gmail_connect(request: Request):
        """Redirect user to Google OAuth."""
        user_id      = _get_user(request)
        redirect_uri = str(request.base_url).rstrip('/') + '/bank/gmail/callback'

        try:
            from google_auth_oauthlib.flow import Flow
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError('credentials.json not found')

            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/gmail.readonly'],
                redirect_uri=redirect_uri,
            )
            url, _ = flow.authorization_url(
                access_type='offline',
                state=user_id,
                prompt='consent',
                include_granted_scopes='true',
            )
            return RedirectResponse(url=url)

        except FileNotFoundError:
            # Guide user to set up credentials
            return HTMLResponse("""
            <html><body style="font-family:Arial;padding:40px;background:#0d1117;color:#e6edf3;">
            <h2 style="color:#58a6ff;">Gmail Setup Required</h2>
            <p>To connect Gmail, you need a Google Cloud credentials file.</p>
            <ol style="line-height:2.2;">
              <li>Go to <a href="https://console.cloud.google.com" target="_blank" style="color:#58a6ff;">console.cloud.google.com</a></li>
              <li>Create a project → Enable <strong>Gmail API</strong></li>
              <li>OAuth Consent Screen → Add scope: <code>gmail.readonly</code></li>
              <li>Create OAuth 2.0 credentials (Web Application)</li>
              <li>Add redirect URI: <code>http://127.0.0.1:8080/bank/gmail/callback</code></li>
              <li>Download <strong>credentials.json</strong> → place in project root</li>
              <li>Restart the server and try again</li>
            </ol>
            <a href="/bank-sync" style="color:#58a6ff;">← Back to Bank Sync</a>
            </body></html>""")

        except ImportError:
            raise HTTPException(503, detail='Install: pip install google-auth google-auth-oauthlib google-api-python-client')


    @app.get('/bank/gmail/callback')
    async def gmail_callback(request: Request, code: str = '', state: str = '', error: str = ''):
        """Handle Google OAuth callback."""
        if error:
            return RedirectResponse(url=f'/bank-sync?error=gmail_{error}')

        user_id      = state or _get_user(request)
        redirect_uri = str(request.base_url).rstrip('/') + '/bank/gmail/callback'

        try:
            from google_auth_oauthlib.flow import Flow
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/gmail.readonly'],
                redirect_uri=redirect_uri,
            )
            flow.fetch_token(code=code)
            creds = flow.credentials

            GMAIL_TOKENS[user_id] = {
                'token':         creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri':     creds.token_uri,
                'client_id':     creds.client_id,
                'client_secret': creds.client_secret,
                'scopes':        list(creds.scopes or []),
                'email':         '',  # filled on first use
            }

            # Try to get email address
            try:
                from googleapiclient.discovery import build
                svc   = build('gmail','v1',credentials=creds)
                profile = svc.users().getProfile(userId='me').execute()
                GMAIL_TOKENS[user_id]['email'] = profile.get('emailAddress','')
            except Exception:
                pass

            return RedirectResponse(url='/bank-sync?gmail=connected')

        except Exception as e:
            logger.error(f'Gmail callback error: {e}')
            return RedirectResponse(url=f'/bank-sync?error=gmail_callback')


    @app.get('/bank/gmail/status')
    async def gmail_status(request: Request):
        """Check if Gmail is connected."""
        uid = _get_user(request)
        tok = GMAIL_TOKENS.get(uid, {})
        return {
            'connected': uid in GMAIL_TOKENS,
            'email':     tok.get('email', ''),
        }


    @app.get('/bank/gmail/sync')
    async def gmail_sync(request: Request, days: int = 30, min_amount: float = 0):
        """
        Scan Gmail for bank transaction emails and parse expenses.
        """
        uid = _get_user(request)

        if uid not in GMAIL_TOKENS:
            raise HTTPException(401, detail='Gmail not connected. Visit /bank/gmail/connect')

        try:
            service = _gmail_service(uid)
        except Exception as e:
            raise HTTPException(503, detail=str(e))

        after = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')

        # Multiple queries to maximize coverage of Indian bank emails
        queries = [
            f'after:{after} (debited OR deducted OR "has been paid" OR "UPI payment" OR "NEFT" OR "IMPS" OR "POS transaction")',
            f'after:{after} (from:alerts OR from:notify OR from:noreply OR from:transaction) (debit OR payment OR spent)',
            f'after:{after} subject:(debit OR transaction OR payment OR alert OR spent OR debited)',
        ]

        seen, all_msgs = set(), []
        for q in queries:
            try:
                res = service.users().messages().list(userId='me', q=q, maxResults=100).execute()
                for m in res.get('messages', []):
                    if m['id'] not in seen:
                        seen.add(m['id'])
                        all_msgs.append(m)
            except Exception as e:
                logger.warning(f'Gmail query failed: {e}')

        expenses, skipped, errors = [], 0, 0

        for ref in all_msgs[:150]:
            try:
                msg     = service.users().messages().get(userId='me', id=ref['id'], format='full').execute()
                subject = _gmail_header(msg, 'subject')
                sender  = _gmail_header(msg, 'from').lower()
                body    = _gmail_body(msg)
                full    = subject + ' ' + body

                if not _is_debit(body, subject):
                    skipped += 1; continue

                amount = _extract_amount(full)
                if not amount or amount < min_amount:
                    skipped += 1; continue

                merchant = _extract_merchant(body, subject)
                category = _guess_category(full, merchant)
                date     = _gmail_date(msg)

                bank = 'Unknown'
                for b in ['hdfc','sbi','icici','axis','kotak','paytm','phonepe','amazon pay']:
                    if b in sender or b in subject.lower():
                        bank = b.upper(); break

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
                logger.warning(f'Email parse error: {e}')
                errors += 1

        total   = sum(e['amount'] for e in expenses)
        by_cat: dict = {}
        for e in expenses:
            by_cat[e['category']] = round(by_cat.get(e['category'], 0) + e['amount'], 2)

        return {
            'connected':      True,
            'emails_scanned': len(all_msgs),
            'expenses_found': len(expenses),
            'skipped':        skipped,
            'errors':         errors,
            'expenses':       expenses,
            'total_amount':   round(total, 2),
            'by_category':    by_cat,
            'period_days':    days,
            'scanned_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


    @app.post('/bank/gmail/add-expenses')
    async def gmail_add_expenses(request: Request, body: ExpensesImport):
        """Add Gmail-detected expenses to TrackExpensio."""
        uid   = _get_user(request)
        added, failed = 0, []

        for exp in body.expenses:
            try:
                # Call your existing add_expense MCP tool
                from main import TOOL_REGISTRY
                await TOOL_REGISTRY['add_expense'](
                    user_id=uid,
                    amount=exp['amount'],
                    category=exp.get('category','Other'),
                    merchant=exp.get('merchant','Unknown'),
                    note=f"Auto-imported from Gmail: {exp.get('email_subject','')}",
                    date=exp.get('date',''),
                    currency=exp.get('currency','INR'),
                )
                added += 1
            except Exception as e:
                logger.warning(f'Add expense failed: {e}')
                failed.append({'merchant': exp.get('merchant'), 'error': str(e)})

        return {'status':'success', 'added':added, 'failed':len(failed), 'errors':failed}


    @app.delete('/bank/gmail/disconnect')
    async def gmail_disconnect(request: Request):
        """Remove Gmail connection."""
        uid = _get_user(request)
        GMAIL_TOKENS.pop(uid, None)
        return {'status':'disconnected'}


    # ══════════════════════════════════════════════════════
    # SMS PARSER ENDPOINT
    # ══════════════════════════════════════════════════════

    @app.post('/bank/sms/parse')
    async def sms_parse(body: SMSRequest):
        """
        Parse a bank SMS into an expense.
        POST /bank/sms/parse
        {"sms_text": "INR 599 debited from A/C XX1234 at Zomato"}
        """
        sms = body.sms_text.strip()
        if not sms:
            raise HTTPException(400, detail='sms_text is required')

        if not _is_debit(sms):
            return {'parsed': False, 'message': 'No debit transaction found in this SMS'}

        amount = _extract_amount(sms)
        if not amount:
            return {'parsed': False, 'message': 'Could not extract amount from SMS'}

        merchant = _extract_merchant(sms)
        category = _guess_category(sms, merchant)

        # Try to extract date from SMS
        date = datetime.now().strftime('%Y-%m-%d')
        date_pats = [
            r'(\d{2}[/-]\w{3}[/-]\d{2,4})',
            r'(\d{2}[/-]\d{2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
        ]
        for dp in date_pats:
            dm = re.search(dp, sms, re.IGNORECASE)
            if dm:
                try:
                    import dateparser
                    parsed = dateparser.parse(dm.group(1))
                    if parsed:
                        date = parsed.strftime('%Y-%m-%d')
                except Exception:
                    pass
                break

        return {
            'parsed': True,
            'expense': {
                'merchant': merchant,
                'amount':   round(amount, 2),
                'category': category,
                'date':     date,
                'source':   'sms',
                'currency': 'INR',
                'raw_sms':  sms[:200],
            }
        }


    # ══════════════════════════════════════════════════════
    # CSV UPLOAD ENDPOINTS
    # ══════════════════════════════════════════════════════

    @app.post('/bank/csv/upload')
    async def csv_upload(
        file: UploadFile = File(...),
        bank: str = Form('hdfc'),
        min_amount: float = Form(0),
    ):
        """
        Upload a bank statement CSV.
        POST /bank/csv/upload  (multipart form)
        Fields: file (.csv), bank (hdfc/sbi/icici/axis/kotak), min_amount
        """
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, detail='Please upload a .csv file')

        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')

        expenses = _parse_csv_content(text, bank=bank)

        if min_amount > 0:
            expenses = [e for e in expenses if e['amount'] >= min_amount]

        total   = sum(e['amount'] for e in expenses)
        by_cat: dict = {}
        for e in expenses:
            by_cat[e['category']] = round(by_cat.get(e['category'], 0) + e['amount'], 2)

        return {
            'status':      'success',
            'bank':        bank.upper(),
            'expenses':    expenses,
            'count':       len(expenses),
            'total':       round(total, 2),
            'by_category': by_cat,
            'parsed_at':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


    @app.post('/bank/csv/add-expenses')
    async def csv_add_expenses(request: Request, body: ExpensesImport):
        """Add CSV-parsed expenses to TrackExpensio."""
        uid   = _get_user(request)
        added = 0
        for exp in body.expenses:
            try:
                from main import TOOL_REGISTRY
                await TOOL_REGISTRY['add_expense'](
                    user_id=uid,
                    amount=exp['amount'],
                    category=exp.get('category','Other'),
                    merchant=exp.get('merchant','Unknown'),
                    note=f"Imported from {exp.get('source','bank').upper()}",
                    date=exp.get('date',''),
                    currency=exp.get('currency','INR'),
                )
                added += 1
            except Exception as e:
                logger.warning(f'Add expense failed: {e}')

        return {'status':'success', 'added':added, 'total':len(body.expenses)}


    # ══════════════════════════════════════════════════════
    # PLAID ENDPOINTS
    # ══════════════════════════════════════════════════════

    def _plaid_client():
        """Build Plaid API client from env vars."""
        try:
            import plaid
            from plaid.api import plaid_api
            env = os.getenv('PLAID_ENV','sandbox')
            host = {
                'sandbox':    plaid.Environment.Sandbox,
                'development':plaid.Environment.Development,
                'production': plaid.Environment.Production,
            }.get(env, plaid.Environment.Sandbox)

            cfg = plaid.Configuration(
                host=host,
                api_key={
                    'clientId': os.getenv('PLAID_CLIENT_ID',''),
                    'secret':   os.getenv('PLAID_SECRET',''),
                }
            )
            return plaid_api.PlaidApi(plaid.ApiClient(cfg))
        except ImportError:
            return None


    @app.post('/plaid/create-link-token')
    async def plaid_link_token(request: Request):
        """Create a Plaid Link token for the frontend."""
        uid    = _get_user(request)
        client = _plaid_client()

        if not client or not os.getenv('PLAID_CLIENT_ID'):
            # Return demo indicator
            return {'demo': True, 'message': 'Plaid not configured. Add PLAID_CLIENT_ID and PLAID_SECRET to .env'}

        try:
            from plaid.model.link_token_create_request import LinkTokenCreateRequest
            from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
            from plaid.model.products import Products
            from plaid.model.country_code import CountryCode

            req = LinkTokenCreateRequest(
                products=[Products('transactions')],
                client_name='TrackExpensio',
                country_codes=[CountryCode('US'), CountryCode('IN')],
                language='en',
                user=LinkTokenCreateRequestUser(client_user_id=uid),
            )
            resp = client.link_token_create(req)
            return {'link_token': resp.link_token, 'expiration': str(resp.expiration)}

        except Exception as e:
            raise HTTPException(500, detail=f'Plaid error: {e}')


    @app.post('/plaid/exchange-token')
    async def plaid_exchange_token(request: Request, body: PlaidExchangeRequest):
        """Exchange Plaid public_token for permanent access_token."""
        uid    = _get_user(request)
        client = _plaid_client()

        if not client:
            raise HTTPException(503, detail='Plaid not configured')

        try:
            from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
            req  = ItemPublicTokenExchangeRequest(public_token=body.public_token)
            resp = client.item_public_token_exchange(req)
            PLAID_TOKENS[uid] = resp.access_token
            return {'status':'connected', 'item_id': resp.item_id}
        except Exception as e:
            raise HTTPException(500, detail=f'Exchange error: {e}')


    @app.get('/plaid/sync-transactions')
    async def plaid_sync(request: Request, days: int = 30):
        """Fetch real transactions from connected Plaid account."""
        uid   = _get_user(request)
        token = PLAID_TOKENS.get(uid)

        if not token:
            raise HTTPException(400, detail='No bank connected via Plaid')

        client = _plaid_client()
        if not client:
            raise HTTPException(503, detail='Plaid not configured')

        try:
            from plaid.model.transactions_get_request import TransactionsGetRequest
            from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions

            end   = datetime.now().date()
            start = end - timedelta(days=days)

            req  = TransactionsGetRequest(
                access_token=token,
                start_date=start,
                end_date=end,
                options=TransactionsGetRequestOptions(count=100),
            )
            resp = client.transactions_get(req)
            txns = []
            for t in resp.transactions:
                if t.amount > 0:  # positive = debit
                    txns.append({
                        'merchant': t.merchant_name or t.name,
                        'amount':   round(float(t.amount), 2),
                        'date':     str(t.date),
                        'category': (t.personal_finance_category.primary
                                     if t.personal_finance_category
                                     else (t.category[0] if t.category else 'Other')),
                        'source':   'plaid',
                        'currency': t.iso_currency_code or 'USD',
                        'pending':  t.pending,
                    })
            return {'status':'success', 'transactions':txns, 'count':len(txns)}

        except Exception as e:
            raise HTTPException(500, detail=f'Plaid sync error: {e}')


    @app.get('/plaid/balances')
    async def plaid_balances(request: Request):
        """Get current account balances from Plaid."""
        uid   = _get_user(request)
        token = PLAID_TOKENS.get(uid)
        if not token:
            raise HTTPException(400, detail='No bank connected')

        client = _plaid_client()
        if not client:
            raise HTTPException(503, detail='Plaid not configured')

        try:
            from plaid.model.accounts_get_request import AccountsGetRequest
            req  = AccountsGetRequest(access_token=token)
            resp = client.accounts_get(req)
            return {
                'accounts': [{
                    'name':     a.name,
                    'type':     str(a.type),
                    'balance':  a.balances.current,
                    'currency': a.balances.iso_currency_code,
                    'mask':     a.mask,
                } for a in resp.accounts]
            }
        except Exception as e:
            raise HTTPException(500, detail=f'Balance error: {e}')


    # ══════════════════════════════════════════════════════
    # SETU ACCOUNT AGGREGATOR ENDPOINTS
    # ══════════════════════════════════════════════════════

    @app.post('/bank/aa/consent')
    async def aa_consent(request: Request, body: AAConsentRequest):
        """Create Setu AA consent request."""
        uid   = _get_user(request)
        phone = body.phone.strip()

        if len(phone) != 10 or not phone.isdigit():
            raise HTTPException(400, detail='Enter a valid 10-digit mobile number')

        client_id = os.getenv('SETU_CLIENT_ID','')
        secret    = os.getenv('SETU_CLIENT_SECRET','')

        if not client_id or not secret:
            # Demo mode
            return {
                'status':       'demo_mode',
                'message':      'Setu not configured. Add SETU_CLIENT_ID and SETU_CLIENT_SECRET to .env for real data.',
                'redirect_url': f'/bank/aa-demo?user_id={uid}',
                'setup_guide':  'https://setu.co/data/account-aggregator',
            }

        try:
            import requests as req_lib
            now = datetime.utcnow()
            payload = {
                'Detail': {
                    'consentStart':  now.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    'consentExpiry': '2030-01-01T00:00:00.000Z',
                    'consentMode':   'STORE',
                    'fetchType':     'PERIODIC',
                    'consentTypes':  ['TRANSACTIONS','PROFILE','SUMMARY'],
                    'fiTypes':       ['DEPOSIT'],
                    'DataConsumer':  {'id': 'setu-fiu-id'},
                    'Customer':      {'id': f'{phone}@setu-sandbox'},
                    'Purpose': {
                        'code': '101',
                        'refUri': 'https://api.rebit.org.in/aa/purpose/101.xml',
                        'text': 'Personal expense tracking',
                        'Category': {'type': 'Personal Finance'},
                    },
                    'FIDataRange': {
                        'from': (now - timedelta(days=90)).strftime('%Y-%m-%dT00:00:00.000Z'),
                        'to':   now.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    },
                    'DataLife':  {'unit':'YEAR','value':1},
                    'Frequency': {'unit':'MONTH','value':1},
                },
                'redirectUrl': str(request.base_url).rstrip('/') + f'/bank/aa-callback?user_id={uid}',
            }
            resp = req_lib.post(
                'https://dg-sandbox.setu.co/api/v1/consents',
                headers={
                    'Content-Type':    'application/json',
                    'x-client-id':     client_id,
                    'x-client-secret': secret,
                },
                json=payload,
                timeout=15,
            )
            data = resp.json()
            consent_id = data.get('id','')
            AA_CONSENTS[uid] = consent_id

            return {
                'status':       'pending',
                'consent_id':   consent_id,
                'redirect_url': data.get('url',''),
                'message':      'Redirect user to redirect_url to approve consent',
            }

        except Exception as e:
            logger.error(f'Setu consent error: {e}')
            return {
                'status':       'demo_mode',
                'message':      f'Setu error: {e}. Using demo data.',
                'redirect_url': f'/bank/aa-demo?user_id={uid}',
            }


    @app.get('/bank/aa-callback')
    async def aa_callback(request: Request, user_id: str = '', demo: bool = False):
        """Handle Setu AA callback after user approves consent."""
        uid        = user_id or _get_user(request)
        consent_id = AA_CONSENTS.get(uid,'')

        if demo or not consent_id:
            return RedirectResponse(url='/bank-sync?aa=demo')

        # In real implementation: fetch transactions using consent_id
        return RedirectResponse(url='/bank-sync?aa=connected')


    @app.get('/bank/aa-demo')
    async def aa_demo(request: Request):
        """Return mock Indian bank transactions for demo."""
        txns  = _mock_indian_txns()
        total = sum(t['amount'] for t in txns)
        return {
            'status':       'demo',
            'transactions': txns,
            'count':        len(txns),
            'total':        total,
            'message':      'Demo data. Add SETU_CLIENT_ID + SETU_CLIENT_SECRET for real bank data.',
        }


    @app.get('/bank/aa/demo-sync')
    async def aa_demo_sync():
        """Quick endpoint to get demo Indian transactions."""
        txns = _mock_indian_txns()
        return {
            'status':       'demo',
            'transactions': txns,
            'count':        len(txns),
            'total':        sum(t['amount'] for t in txns),
        }


    logger.info('✅ Bank sync routes registered: Gmail, CSV, Plaid, Setu AA, SMS')
    return app
