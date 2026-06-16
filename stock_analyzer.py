"""
TrackExpensio — Smart Stock Analyzer
=====================================
Live candlestick chart, Bollinger Bands, RSI, MACD, buy/sell/hold signals,
30-day linear regression projection, Serper news, support/resistance levels.

Install: pip install yfinance plotly pandas numpy scikit-learn requests
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots


COMPANY_TICKER_MAP = {
    "indian oil": "IOC.NS",
    "ioc": "IOC.NS",
    "indian oil corporation": "IOC.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "wipro": "WIPRO.NS",
    "ongc": "ONGC.NS",
    "ntpc": "NTPC.NS",
    "nhpc": "NHPC.NS",
    "coal india": "COALINDIA.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "meta": "META",
}


def resolve_ticker(query: str) -> str:
    """Convert company name to ticker symbol."""
    query_clean = (query or "").strip().lower()
    if not query_clean:
        return ""

    if query_clean in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[query_clean]

    for name, ticker in COMPANY_TICKER_MAP.items():
        if name in query_clean or query_clean in name:
            return ticker

    if len(query_clean) <= 20 and " " not in query_clean:
        return query_clean.upper()

    try:
        results = yf.Search(query, max_results=1)
        quotes = results.quotes
        if quotes:
            return quotes[0].get("symbol", query_clean.upper())
    except Exception:
        pass

    return query_clean.upper()


def _safe(val):
    if val is None:
        return None
    try:
        if isinstance(val, float) and (val != val):
            return None
        if hasattr(val, "item"):
            return val.item()
        return val
    except Exception:
        return None


def _currency_symbol(currency: str) -> str:
    return (
        "₹" if currency == "INR" else
        "£" if currency == "GBP" else
        "€" if currency == "EUR" else
        "¥" if currency == "JPY" else
        "$"
    )


def compute_rsi(closes, period=14):
    delta = np.diff(closes.astype(float))
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(len(gains))
    avg_loss = np.zeros(len(gains))
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rs = np.where(avg_loss == 0, 100, avg_gain / (avg_loss + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    pad = np.full(len(closes) - len(rsi), np.nan)
    return np.concatenate([pad, rsi])


def compute_ema(arr, span):
    result = np.zeros(len(arr))
    k = 2 / (span + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def compute_macd(closes, fast=12, slow=26, signal=9):
    closes = closes.astype(float)
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    macd = ema_fast - ema_slow
    sig = compute_ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def compute_bollinger(closes, period=20, std=2.0):
    closes = closes.astype(float)
    sma = np.array([
        np.mean(closes[max(0, i - period):i]) if i >= period else np.nan
        for i in range(len(closes))
    ])
    rolling_std = np.array([
        np.std(closes[max(0, i - period):i]) if i >= period else np.nan
        for i in range(len(closes))
    ])
    return sma, sma + std * rolling_std, sma - std * rolling_std


def compute_sr(closes, n=5):
    highs, lows = [], []
    for i in range(n, len(closes) - n):
        window = closes[i - n : i + n + 1]
        if closes[i] == max(window):
            highs.append(float(closes[i]))
        if closes[i] == min(window):
            lows.append(float(closes[i]))
    return {
        "resistance": round(max(highs[-3:]), 2) if highs else None,
        "support": round(min(lows[-3:]), 2) if lows else None,
    }


def project_prices(closes, days=30):
    from sklearn.linear_model import LinearRegression

    closes = closes.astype(float)
    x = np.arange(len(closes)).reshape(-1, 1)
    model = LinearRegression().fit(x, closes)
    fx = np.arange(len(closes), len(closes) + days).reshape(-1, 1)
    proj = model.predict(fx)
    std = np.std(closes - model.predict(x))
    return {
        "projected": [round(float(v), 2) for v in proj],
        "upper_band": [round(float(v + 1.5 * std), 2) for v in proj],
        "lower_band": [round(float(v - 1.5 * std), 2) for v in proj],
        "trend": "bullish" if model.coef_[0] > 0 else "bearish",
        "slope": round(float(model.coef_[0]), 4),
        "r_squared": round(float(model.score(x, closes)), 3),
    }


def generate_signals(closes, rsi, macd_line, signal_line, sma, upper_bb, lower_bb, currency_symbol="$"):
    def v(arr):
        return float(arr[-1]) if not np.isnan(arr[-1]) else 50

    price = float(closes[-1])
    r = v(rsi)
    m = float(macd_line[-1])
    s = float(signal_line[-1])
    sma_v = v(sma)
    ub = v(upper_bb) if not np.isnan(upper_bb[-1]) else price * 1.02
    lb = v(lower_bb) if not np.isnan(lower_bb[-1]) else price * 0.98

    buy = 0
    sell = 0
    reasons = []

    if r < 30:
        buy += 30
        reasons.append(f"RSI {r:.1f} — Oversold zone. Strong buy signal.")
    elif r < 45:
        buy += 15
        reasons.append(f"RSI {r:.1f} — Approaching oversold.")
    elif r > 70:
        sell += 30
        reasons.append(f"RSI {r:.1f} — Overbought zone. Consider selling.")
    elif r > 55:
        sell += 15
        reasons.append(f"RSI {r:.1f} — Approaching overbought.")
    else:
        reasons.append(f"RSI {r:.1f} — Neutral. No clear signal.")

    if m > s:
        buy += 20
        reasons.append("MACD above signal line. Bullish momentum.")
    else:
        sell += 20
        reasons.append("MACD below signal line. Bearish momentum.")

    if price > sma_v:
        buy += 15
        reasons.append(f"Price above 20-day SMA {currency_symbol}{sma_v:.2f}. Bullish trend.")
    else:
        sell += 15
        reasons.append(f"Price below 20-day SMA {currency_symbol}{sma_v:.2f}. Bearish trend.")

    if price < lb:
        buy += 25
        reasons.append("Below lower Bollinger Band. High probability bounce zone.")
    elif price > ub:
        sell += 25
        reasons.append("Above upper Bollinger Band. Potential reversal ahead.")
    else:
        pct = (price - lb) / (ub - lb) * 100
        reasons.append(f"Price at {pct:.0f}% inside Bollinger Bands. Normal range.")

    if len(closes) >= 5:
        chg = (closes[-1] - closes[-5]) / closes[-5] * 100
        if chg > 3:
            sell += 10
            reasons.append(f"Up {chg:.1f}% in 5 days. Short-term overbought.")
        elif chg < -3:
            buy += 10
            reasons.append(f"Down {abs(chg):.1f}% in 5 days. Potential bounce entry.")

    total = max(buy + sell, 1)
    if buy > sell and buy > 30:
        sig = "BUY"
        conf = min(95, int(buy / total * 100))
        color = "#00ff88"
    elif sell > buy and sell > 30:
        sig = "SELL"
        conf = min(95, int(sell / total * 100))
        color = "#ff4444"
    else:
        sig = "HOLD"
        conf = 50
        color = "#ffaa00"

    return {
        "signal": sig,
        "confidence": conf,
        "color": color,
        "reasons": reasons,
        "buy_score": buy,
        "sell_score": sell,
    }


def fetch_stock_news(ticker, company_name=""):
    key = os.getenv("SERPER_API_KEY", "")
    query = f"{company_name or ticker} stock news"
    if not key:
        return [{
            "title": "Add SERPER_API_KEY to .env for live news",
            "source": "TrackExpensio",
            "url": "#",
            "snippet": "",
            "sentiment": "neutral",
            "date": "",
        }]
    try:
        r = requests.post(
            "https://google.serper.dev/news",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": 8, "tbs": "qdr:d"},
            timeout=10,
        )
        items = r.json().get("news", [])[:8]
        news = []
        for item in items:
            t = item.get("title", "").lower()
            pos = ["surge", "rally", "gain", "rise", "up", "high", "beat", "record", "bull", "growth"]
            neg = ["drop", "fall", "down", "loss", "decline", "miss", "low", "sell", "bear", "crash"]
            sent = "positive" if any(w in t for w in pos) else "negative" if any(w in t for w in neg) else "neutral"
            news.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("link", "#"),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "sentiment": sent,
            })
        return news
    except Exception as e:
        return [{
            "title": f"News error: {e}",
            "source": "",
            "url": "#",
            "snippet": "",
            "sentiment": "neutral",
            "date": "",
        }]


def build_chart(tkr, ticker):
    hist = tkr.history(period="6mo", interval="1d")
    if hist.empty:
        return "{}"

    dates = hist.index.strftime("%Y-%m-%d").tolist()
    opens = hist["Open"].values.astype(float)
    highs = hist["High"].values.astype(float)
    lows = hist["Low"].values.astype(float)
    closes = hist["Close"].values.astype(float)
    vols = hist["Volume"].values

    rsi_v = compute_rsi(closes)
    macd_l, sig_l, mh = compute_macd(closes)
    sma, ubb, lbb = compute_bollinger(closes)
    proj = project_prices(closes, 30)
    proj_dates = [
        (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, 31)
    ]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker.upper()} — Price Chart", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=ubb, line=dict(color="rgba(100,180,255,0.5)", width=1, dash="dot"), showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lbb,
            fill="tonexty",
            fillcolor="rgba(100,180,255,0.07)",
            line=dict(color="rgba(100,180,255,0.5)", width=1, dash="dot"),
            name="Bollinger Bands",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sma, name="SMA 20", line=dict(color="#ffaa00", width=1.5)),
        row=1,
        col=1,
    )

    pd_dates = [dates[-1]] + proj_dates
    fig.add_trace(
        go.Scatter(
            x=pd_dates,
            y=[closes[-1]] + proj["upper_band"],
            line=dict(color="rgba(160,100,255,0.3)", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pd_dates,
            y=[closes[-1]] + proj["lower_band"],
            fill="tonexty",
            fillcolor="rgba(160,100,255,0.08)",
            line=dict(color="rgba(160,100,255,0.3)", width=1),
            name="30-Day Forecast",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pd_dates,
            y=[closes[-1]] + proj["projected"],
            line=dict(color="#aa66ff", width=2, dash="dash"),
            name="Projected",
        ),
        row=1,
        col=1,
    )

    vc = ["#00ff88" if closes[i] >= opens[i] else "#ff4444" for i in range(len(dates))]
    fig.add_trace(go.Bar(x=dates, y=vols, marker_color=vc, showlegend=False), row=2, col=1)

    fig.add_trace(
        go.Scatter(x=dates, y=rsi_v, name="RSI", line=dict(color="#00d4ff", width=1.5)),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ff4444", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff88", row=3, col=1)

    hc = ["#00ff88" if v >= 0 else "#ff4444" for v in mh]
    fig.add_trace(go.Bar(x=dates, y=mh, marker_color=hc, showlegend=False), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=macd_l, name="MACD", line=dict(color="#00d4ff", width=1.5)),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sig_l, name="Signal", line=dict(color="#ff9900", width=1.5)),
        row=4,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter"),
        height=820,
        margin=dict(l=40, r=40, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        ax = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(**{ax: dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True)})

    return fig.to_json()


def analyze_stock(ticker: str) -> dict:
    original = (ticker or "").strip()
    if not original:
        return {"error": "Ticker is required"}

    ticker = resolve_ticker(original)
    if not ticker or len(ticker) > 25:
        return {"error": f"Could not find ticker for: '{original}'"}

    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info or {}
        hist = tkr.history(period="6mo", interval="1d")

        if hist.empty:
            return {"error": f"No data for '{ticker}'. Check the symbol."}

        closes = hist["Close"].values.astype(float)

        rsi_v = compute_rsi(closes)
        macd_l, sig_l, _mh = compute_macd(closes)
        sma, ubb, lbb = compute_bollinger(closes)
        sr = compute_sr(closes)
        proj = project_prices(closes, 30)

        currency = (
            info.get("currency")
            or info.get("financialCurrency")
            or ("INR" if ".NS" in ticker or ".BO" in ticker else "USD")
        )
        currency_symbol = _currency_symbol(currency)

        sig = generate_signals(closes, rsi_v, macd_l, sig_l, sma, ubb, lbb, currency_symbol)
        chart = build_chart(tkr, ticker)
        news = fetch_stock_news(ticker, info.get("longName", ""))

        price = _safe(info.get("currentPrice") or closes[-1])
        prev = _safe(info.get("previousClose") or (closes[-2] if len(closes) > 1 else None))
        change = round(price - prev, 2) if price and prev else None
        chg_pct = round(change / prev * 100, 2) if change and prev else None

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "currency": currency,
            "currency_symbol": currency_symbol,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": price,
            "prev_close": prev,
            "change": change,
            "change_pct": chg_pct,
            "day_high": _safe(info.get("dayHigh")),
            "day_low": _safe(info.get("dayLow")),
            "week_52_high": _safe(info.get("fiftyTwoWeekHigh")),
            "week_52_low": _safe(info.get("fiftyTwoWeekLow")),
            "market_cap": _safe(info.get("marketCap")),
            "volume": _safe(info.get("volume")),
            "pe_ratio": _safe(info.get("trailingPE")),
            "dividend_yield": _safe(info.get("dividendYield")),
            "beta": _safe(info.get("beta")),
            "rsi": round(float(rsi_v[-1]), 2) if not np.isnan(rsi_v[-1]) else None,
            "macd": round(float(macd_l[-1]), 4),
            "sma_20": round(float(sma[-1]), 2) if not np.isnan(sma[-1]) else None,
            "bb_upper": round(float(ubb[-1]), 2) if not np.isnan(ubb[-1]) else None,
            "bb_lower": round(float(lbb[-1]), 2) if not np.isnan(lbb[-1]) else None,
            "support": sr["support"],
            "resistance": sr["resistance"],
            "signal": sig,
            "projection": proj,
            "news": news,
            "chart_json": chart,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
