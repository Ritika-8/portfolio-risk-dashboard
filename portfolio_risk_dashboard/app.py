import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.stock_data import fetch_stock_data, fetch_benchmark, NSE_STOCKS
from analytics.returns import (
    compute_daily_returns, compute_cumulative_returns,
    compute_annualised_return, compute_annualised_volatility,
    compute_portfolio_returns, compute_rolling_volatility,
    compute_correlation_matrix,
)
from analytics.risk import (
    compute_sharpe_ratio, compute_var, compute_beta,
    compute_max_drawdown, monte_carlo_simulation, compute_sortino_ratio,
)
from analytics.optimiser import optimise_portfolio, generate_efficient_frontier
from nlp.sentiment import get_stock_sentiment
from nlp.ai_summary import generate_risk_summary

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Indian Stock Portfolio Risk & Intelligence Dashboard")
st.caption("Investment analytics · Risk modelling · AI insights · NSE stocks · 100% Free")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Build Your Portfolio")

    selected_stocks = st.multiselect(
        "Select NSE Stocks (min 2)",
        options=list(NSE_STOCKS.keys()),
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        format_func=lambda x: f"{x} — {NSE_STOCKS[x]}",
    )

    period_options = {"1 Year": 1, "2 Years": 2, "3 Years": 3}
    period_choice = st.selectbox(
        "History Period",
        options=list(period_options.keys()),
        index=1,
    )
    period = period_options[period_choice]

    initial_investment = st.number_input(
        "Investment Amount (₹)",
        value=100000,
        step=10000,
        min_value=10000,
    )

    st.subheader("Portfolio Weights")
    weights = {}
    if selected_stocks:
        equal_weight = round(1.0 / len(selected_stocks), 3)
        for stock in selected_stocks:
            weights[stock] = st.slider(stock, 0.0, 1.0, equal_weight, 0.01)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — normalising to 1.0")
            weights = {k: v / total for k, v in weights.items()}
        else:
            st.success(f"Weights sum: {total:.2f} ✓")

    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── GUARD ──────────────────────────────────────────────────────────────────────
if len(selected_stocks) < 2:
    st.info("👈 Select at least 2 NSE stocks from the sidebar to begin.")
    st.stop()

# ── FETCH DATA ────────────────────────────────────────────────────────────────
if run or "prices" not in st.session_state:
    with st.spinner("Fetching live NSE market data... (15-20 seconds)"):
        prices = fetch_stock_data(selected_stocks, period_years=period)
        benchmark = fetch_benchmark(period_years=period)
        st.session_state["prices"] = prices
        st.session_state["benchmark"] = benchmark
        st.session_state.pop("sentiments", None)
        st.session_state.pop("ai_summary", None)

prices = st.session_state["prices"]
benchmark = st.session_state["benchmark"]

# ── GUARD: empty data ─────────────────────────────────────────────────────────
if prices.empty:
    st.error("Could not fetch stock data. Check your internet and try again. Also try: pip install --upgrade yfinance")
    st.stop()

available_stocks = [s for s in selected_stocks if s in prices.columns]
if len(available_stocks) < 2:
    st.error("Not enough stocks returned data. Try different stocks.")
    st.stop()

prices = prices[available_stocks]
weights_array = np.array([weights[s] for s in available_stocks])
if weights_array.sum() > 0:
    weights_array = weights_array / weights_array.sum()

# ── CALCULATIONS ──────────────────────────────────────────────────────────────
returns = compute_daily_returns(prices)
port_returns = compute_portfolio_returns(returns, weights_array)
bench_returns = benchmark.pct_change().dropna() if not benchmark.empty else pd.Series(dtype=float)
aligned_bench = bench_returns.reindex(port_returns.index).fillna(0)

var_metrics = compute_var(port_returns)
ann_ret = compute_annualised_return(returns)
ann_vol = compute_annualised_volatility(returns)
port_ann_vol = float(ann_vol.values.dot(weights_array))
port_ann_ret = float(ann_ret.values.dot(weights_array))

portfolio_metrics = {
    "sharpe_ratio": compute_sharpe_ratio(port_returns),
    "sortino_ratio": compute_sortino_ratio(port_returns),
    "var_95": var_metrics["var_95"],
    "var_99": var_metrics["var_99"],
    "cvar_95": var_metrics["cvar_95"],
    "beta": compute_beta(port_returns, aligned_bench),
    "max_drawdown": compute_max_drawdown(port_returns),
    "volatility": port_ann_vol,
    "annual_return": port_ann_ret,
}

cum_port = compute_cumulative_returns(port_returns.to_frame("Portfolio"))
total_return_pct = float(cum_port.iloc[-1, 0]) * 100
portfolio_value = initial_investment * (1 + float(cum_port.iloc[-1, 0]))

# ══ SECTION 1: SNAPSHOT ══════════════════════════════════════════════════════
st.subheader("Portfolio Snapshot")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Portfolio Value", f"₹{portfolio_value:,.0f}", f"{total_return_pct:+.1f}%")
c2.metric("Annual Return", f"{port_ann_ret:.1%}")
c3.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}", ">1 is good")
c4.metric("VaR 95% (1-day)", f"{portfolio_metrics['var_95']:.2%}")
c5.metric("Beta vs Nifty", f"{portfolio_metrics['beta']:.2f}")
c6.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2%}")
st.divider()

# ══ SECTION 2: PERFORMANCE ═══════════════════════════════════════════════════
st.subheader("Performance Analysis")
col1, col2 = st.columns(2)

with col1:
    cum_stocks = compute_cumulative_returns(returns) * 100
    cum_p = compute_cumulative_returns(port_returns.to_frame("Portfolio")) * 100
    all_cum = pd.concat([cum_stocks, cum_p], axis=1)
    if not benchmark.empty:
        cum_b = compute_cumulative_returns(aligned_bench.to_frame("Nifty 50")) * 100
        all_cum = pd.concat([all_cum, cum_b], axis=1)
    fig = px.line(all_cum, title="Cumulative Returns (%)",
                  labels={"value": "Return (%)", "index": "Date", "variable": "Stock"})
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    roll_vol = compute_rolling_volatility(returns, window=30) * 100
    fig2 = px.line(roll_vol, title="30-Day Rolling Volatility (%)",
                   labels={"value": "Volatility (%)", "index": "Date", "variable": "Stock"})
    fig2.update_layout(height=360)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══ SECTION 3: RISK ANALYTICS ════════════════════════════════════════════════
st.subheader("Risk Analytics")
r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("**Value at Risk**")
    var_df = pd.DataFrame({
        "Metric": ["VaR 95% (1-day)", "VaR 99% (1-day)", "CVaR (Expected Shortfall)"],
        "% Loss": [f"{var_metrics['var_95']:.2%}", f"{var_metrics['var_99']:.2%}", f"{var_metrics['cvar_95']:.2%}"],
        f"On ₹{initial_investment/1000:.0f}K": [
            f"₹{var_metrics['var_95']*initial_investment:,.0f}",
            f"₹{var_metrics['var_99']*initial_investment:,.0f}",
            f"₹{var_metrics['cvar_95']*initial_investment:,.0f}",
        ],
    })
    st.dataframe(var_df, hide_index=True, use_container_width=True)
    st.markdown("**Other Metrics**")
    other_df = pd.DataFrame({
        "Metric": ["Sortino Ratio", "Annual Volatility", "Beta vs Nifty", "Max Drawdown"],
        "Value": [
            f"{portfolio_metrics['sortino_ratio']:.2f}",
            f"{portfolio_metrics['volatility']:.2%}",
            f"{portfolio_metrics['beta']:.2f}",
            f"{portfolio_metrics['max_drawdown']:.2%}",
        ],
    })
    st.dataframe(other_df, hide_index=True, use_container_width=True)

with r2:
    corr = compute_correlation_matrix(returns)
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                     title="Correlation Matrix", zmin=-1, zmax=1)
    fig3.update_layout(height=360)
    st.plotly_chart(fig3, use_container_width=True)

with r3:
    stock_sharpe = ann_ret / ann_vol
    risk_table = pd.DataFrame({
        "Stock": available_stocks,
        "Ann. Return": [f"{ann_ret[s]:.1%}" for s in available_stocks],
        "Volatility": [f"{ann_vol[s]:.1%}" for s in available_stocks],
        "Sharpe": [f"{stock_sharpe[s]:.2f}" for s in available_stocks],
        "Weight": [f"{weights[s]:.1%}" for s in available_stocks],
    })
    st.markdown("**Per-Stock Metrics**")
    st.dataframe(risk_table, hide_index=True, use_container_width=True)
    fig_dist = px.histogram(port_returns, nbins=60, title="Return Distribution",
                            labels={"value": "Daily Return"},
                            color_discrete_sequence=["steelblue"])
    fig_dist.add_vline(x=-var_metrics["var_95"], line_dash="dash",
                       line_color="red", annotation_text="VaR 95%")
    fig_dist.update_layout(height=230, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ══ SECTION 4: MONTE CARLO ═══════════════════════════════════════════════════
st.subheader("Monte Carlo Simulation — 500 Scenarios, 1 Year")
sims = monte_carlo_simulation(port_returns, num_simulations=500,
                               num_days=252, initial_value=initial_investment)
final_vals = sims[:, -1]

fig4 = go.Figure()
for i in range(0, 500, 5):
    fig4.add_trace(go.Scatter(y=sims[i], mode="lines",
                              line=dict(width=0.4, color="rgba(100,149,237,0.12)"),
                              showlegend=False))
p5 = np.percentile(sims, 5, axis=0)
p50 = np.percentile(sims, 50, axis=0)
p95 = np.percentile(sims, 95, axis=0)
fig4.add_trace(go.Scatter(y=p95, mode="lines", name="Best 5%", line=dict(color="green", width=2.5)))
fig4.add_trace(go.Scatter(y=p50, mode="lines", name="Median", line=dict(color="royalblue", width=2.5)))
fig4.add_trace(go.Scatter(y=p5, mode="lines", name="Worst 5%", line=dict(color="red", width=2.5)))
fig4.add_hline(y=initial_investment, line_dash="dot", line_color="orange",
               annotation_text=f"Initial ₹{initial_investment:,}")
fig4.update_layout(
    title=f"Median: ₹{np.median(final_vals):,.0f}  |  Best 5%: ₹{np.percentile(final_vals,95):,.0f}  |  Worst 5%: ₹{np.percentile(final_vals,5):,.0f}",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value (₹)", height=420)
st.plotly_chart(fig4, use_container_width=True)
st.divider()

# ══ SECTION 5: OPTIMISER ═════════════════════════════════════════════════════
st.subheader("Portfolio Optimisation — Maximum Sharpe Ratio")
with st.spinner("Optimising portfolio weights..."):
    optimal = optimise_portfolio(returns)
    frontier = generate_efficient_frontier(returns, num_portfolios=300)

opt1, opt2 = st.columns([1, 2])
with opt1:
    opt_df = pd.DataFrame([
        {"Stock": k, "Optimal": f"{v:.1%}", "Current": f"{weights.get(k,0):.1%}",
         "Change": f"{(v - weights.get(k,0)):+.1%}"}
        for k, v in optimal["weights"].items()
    ])
    st.dataframe(opt_df, hide_index=True, use_container_width=True)
    st.metric("Expected Return", f"{optimal['expected_annual_return']}%")
    st.metric("Expected Volatility", f"{optimal['expected_annual_volatility']}%")
    st.metric("Optimal Sharpe", f"{optimal['sharpe_ratio']}")

with opt2:
    fig5 = px.scatter(frontier, x="volatility", y="return", color="sharpe",
                      color_continuous_scale="Viridis",
                      title="Efficient Frontier — 300 Random Portfolios",
                      labels={"volatility": "Volatility (%)", "return": "Return (%)", "sharpe": "Sharpe"})
    fig5.add_scatter(x=[optimal["expected_annual_volatility"]], y=[optimal["expected_annual_return"]],
                     mode="markers", marker=dict(color="red", size=16, symbol="star"),
                     name="Optimal Portfolio")
    fig5.update_layout(height=380)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ══ SECTION 6: SENTIMENT ═════════════════════════════════════════════════════
st.subheader("Company Sentiment Analysis — FinBERT + Wikipedia")
st.caption("Uses Wikipedia company descriptions + FinBERT AI model. No API key needed.")

if st.button("Analyse Sentiment", use_container_width=False):
    sentiment_results = {}
    progress = st.progress(0, text="Analysing...")
    for i, stock in enumerate(available_stocks):
        company = NSE_STOCKS.get(stock, stock)
        progress.progress((i + 1) / len(available_stocks), text=f"Analysing {stock}...")
        sentiment_results[stock] = get_stock_sentiment(stock, company)
    progress.empty()
    st.session_state["sentiments"] = sentiment_results
    st.success("Sentiment analysis complete!")

if "sentiments" in st.session_state:
    sentiments = st.session_state["sentiments"]
    sent_cols = st.columns(len(available_stocks))
    for i, stock in enumerate(available_stocks):
        s = sentiments[stock]
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(s["sentiment"], "⚪")
        sent_cols[i].metric(stock, f"{emoji} {s['sentiment'].upper()}", f"Score: {s['score']}")

    st.divider()

    # ══ SECTION 7: AI SUMMARY ════════════════════════════════════════════════
    st.subheader("AI Risk Commentary — Mistral 7B (Hugging Face Free)")
    st.caption("Uses Hugging Face free inference API. Get your free token at huggingface.co → Settings → Access Tokens")

    hf_key = st.text_input(
        "Hugging Face Token (optional — leave blank for rule-based summary)",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
    )
    if hf_key:
        import os
        os.environ["HF_API_KEY"] = hf_key

    if st.button("Generate Risk Summary", type="primary"):
        with st.spinner("Generating AI risk commentary..."):
            summary = generate_risk_summary(portfolio_metrics, sentiments)
            st.session_state["ai_summary"] = summary

    if "ai_summary" in st.session_state:
        st.info(st.session_state["ai_summary"])

st.divider()
st.caption("Educational purposes only. Not financial advice. Data from Yahoo Finance.")
