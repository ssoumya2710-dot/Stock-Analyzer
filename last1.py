import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
USD_TO_INR = 83          # approximate conversion rate

# 20 popular tickers split into categories
TICKERS = {
    "🇺🇸 US Stocks": [
        ("AAPL",  "Apple Inc."),
        ("MSFT",  "Microsoft Corp."),
        ("GOOGL", "Alphabet Inc."),
        ("AMZN",  "Amazon.com Inc."),
        ("TSLA",  "Tesla Inc."),
        ("META",  "Meta Platforms"),
        ("NVDA",  "NVIDIA Corp."),
        ("NFLX",  "Netflix Inc."),
    ],
    "🇮🇳 Indian Stocks (NSE)": [
        ("TCS.NS",        "Tata Consultancy"),
        ("INFY.NS",       "Infosys Ltd."),
        ("RELIANCE.NS",   "Reliance Industries"),
        ("HDFCBANK.NS",   "HDFC Bank"),
        ("WIPRO.NS",      "Wipro Ltd."),
        ("TATAMOTORS.NS", "Tata Motors"),
        ("ITC.NS",        "ITC Ltd."),
        ("SBIN.NS",       "State Bank of India"),
    ],
    "🌍 Commodities / ETFs": [
        ("GC=F",  "Gold Futures"),
        ("CL=F",  "Crude Oil Futures"),
        ("SPY",   "S&P 500 ETF"),
        ("BTC-USD","Bitcoin / USD"),
    ],
}

# ─────────────────────────────────────────────
#  SIDEBAR – navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Smart Stock Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📋 Stock Tickers", "🔍 Analysis"],
        index=0,
    )
    st.markdown("---")
    st.caption("Data powered by Yahoo Finance · Prices delayed ~15 min")

# ═══════════════════════════════════════════════════════════
#  PAGE 1 – STOCK TICKERS
# ═══════════════════════════════════════════════════════════
if page == "📋 Stock Tickers":

    st.title("📋 Live Stock Tickers")
    st.markdown("Prices for **20 popular tickers** across US stocks, Indian stocks, and commodities.")
    st.markdown("---")

    for category, stocks in TICKERS.items():

        st.subheader(category)

        # Download all tickers in the category at once (fast)
        symbols = [sym for sym, _ in stocks]
        raw = yf.download(symbols, period="2d", auto_adjust=True, progress=False)

        # Build display rows
        rows = []
        for symbol, name in stocks:
            try:
                if len(symbols) == 1:
                    close_col = raw["Close"]
                else:
                    close_col = raw["Close"][symbol]

                close_col = close_col.dropna()
                if len(close_col) < 2:
                    raise ValueError("Not enough data")

                today_price = float(close_col.iloc[-1])
                prev_price  = float(close_col.iloc[-2])
                change      = today_price - prev_price
                pct_change  = (change / prev_price) * 100

                # Currency label
                if ".NS" in symbol:
                    price_str  = f"₹{today_price:,.2f}"
                    change_str = f"₹{change:+.2f} ({pct_change:+.2f}%)"
                else:
                    price_str  = f"${today_price:,.2f}  (₹{today_price * USD_TO_INR:,.0f})"
                    change_str = f"${change:+.2f} ({pct_change:+.2f}%)"

                trend = "🟢 Up" if change >= 0 else "🔴 Down"

                rows.append({
                    "Symbol":        symbol,
                    "Company / Asset": name,
                    "Price":         price_str,
                    "Change (1 day)": change_str,
                    "Trend":         trend,
                })

            except Exception:
                rows.append({
                    "Symbol":          symbol,
                    "Company / Asset": name,
                    "Price":           "—",
                    "Change (1 day)":  "—",
                    "Trend":           "⚠️ No data",
                })

        df = pd.DataFrame(rows)

        # Colour "Trend" column with st.dataframe styling
        def colour_trend(val):
            if "Up" in val:
                return "color: green; font-weight: bold"
            elif "Down" in val:
                return "color: red; font-weight: bold"
            return ""

        styled = df.style.applymap(colour_trend, subset=["Trend"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown(" ")   # spacer between categories


# ═══════════════════════════════════════════════════════════
#  PAGE 2 – ANALYSIS
# ═══════════════════════════════════════════════════════════
else:

    st.title("🔍 Stock Analysis & Prediction")
    st.markdown("Enter any stock symbol to get a **7-day price prediction** using Linear Regression.")
    st.markdown("---")

    # ── Inputs ──────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        stock = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="e.g. AAPL · TCS.NS · GC=F",
            help="Use .NS suffix for NSE-listed Indian stocks",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)   # vertical align
        analyze = st.button("🚀 Analyze", use_container_width=True)

    # ── Analysis ─────────────────────────────────────────────
    if analyze:

        with st.spinner(f"Fetching data for **{stock}** …"):
            data = yf.download(stock, period="1y", auto_adjust=True, progress=False)

        if data is None or data.empty:
            st.error("❌ Invalid symbol or no data available. Please try another ticker.")
            st.stop()

        # ── Clean & feature-engineer ─────────────────────────
        data = data.dropna()
        data["Days"] = np.arange(len(data))
        X = data[["Days"]]
        y = data["Close"]

        # ── Train model ───────────────────────────────────────
        model = LinearRegression()
        model.fit(X, y)

        future_days  = np.arange(len(data), len(data) + 7).reshape(-1, 1)
        predictions  = model.predict(future_days)

        current_price = float(data["Close"].iloc[-1])
        future_price  = float(predictions[-1])
        profit_loss   = future_price - current_price
        pct_change    = (profit_loss / current_price) * 100

        currency = "INR" if ".NS" in stock else "USD"

        # ── Helper: format price ──────────────────────────────
        def fmt(price, cur=currency):
            if cur == "INR":
                return f"₹{price:,.2f}"
            return f"${price:,.2f}  (≈ ₹{price * USD_TO_INR:,.0f})"

        # ── Section 1 – Key metrics ───────────────────────────
        st.subheader("📌 Key Metrics")
        m1, m2, m3 = st.columns(3)

        m1.metric(
            label="Current Price",
            value=fmt(current_price),
        )
        m2.metric(
            label="Predicted Price (7 days)",
            value=fmt(future_price),
            delta=f"{pct_change:+.2f}%",
        )
        m3.metric(
            label="Expected Change",
            value=fmt(abs(profit_loss)),
            delta="Profit" if profit_loss >= 0 else "Loss",
            delta_color="normal" if profit_loss >= 0 else "inverse",
        )

        st.markdown("---")

        # ── Section 2 – Recommendation ────────────────────────
        st.subheader("💡 Recommendation")

        if future_price > current_price:
            st.success("✅ **BUY** — Price is expected to rise over the next 7 days.")
        elif future_price < current_price:
            st.error("🔴 **SELL** — Price is expected to fall over the next 7 days.")
        else:
            st.warning("⚠️ **HOLD** — No significant change predicted.")

        if profit_loss > 0:
            st.info(f"📈 Estimated profit per unit held: **{fmt(profit_loss)}**")
        else:
            st.info(f"📉 Estimated loss per unit held: **{fmt(abs(profit_loss))}**")

        st.markdown("---")

        # ── Section 3 – Chart ─────────────────────────────────
        st.subheader("📈 Price Trend + 7-Day Forecast")

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        # Historical line
        ax.plot(
            data["Days"].values,
            data["Close"].values,
            color="#00bfff",
            linewidth=1.8,
            label="Historical Close",
        )
        # Predicted dashed line
        ax.plot(
            range(len(data), len(data) + 7),
            predictions,
            color="#ff6b35",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=5,
            label="7-Day Forecast",
        )

        # Vertical separator
        ax.axvline(x=len(data) - 1, color="#888", linestyle=":", linewidth=1)

        ax.set_xlabel("Trading Days (past year)", color="#cccccc")
        ax.set_ylabel(f"Price ({'₹' if currency == 'INR' else '$'})", color="#cccccc")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"₹{x:,.0f}" if currency == "INR" else f"${x:,.0f}"
            )
        )

        fig.tight_layout()
        st.pyplot(fig)

        # ── Section 4 – Raw data ──────────────────────────────
        st.markdown("---")
        with st.expander("📋 View Recent Historical Data (last 10 days)"):
            display_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
            st.dataframe(
                data[display_cols].tail(10).sort_index(ascending=False),
                use_container_width=True,
            )

        st.caption(
            "⚠️ This prediction uses a simple Linear Regression model for educational purposes only. "
            "It is NOT financial advice. Always consult a qualified advisor before investing."
        )