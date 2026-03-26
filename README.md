# 📊 Indian Stock Portfolio Risk & Intelligence Dashboard

> **A production-grade investment analytics dashboard for NSE-listed stocks — built with Python, Data Science, NLP, and AI.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔗 Live Demo

**[👉 Click here to open the live app](https://portfolio-risk-dashboard-twcjqkkeeruhafnyztpeyw.streamlit.app/)**

---

## 📌 What This Project Does

Most retail investors in India have no way to professionally analyse their stock portfolio risk. This dashboard solves that.

You pick any NSE stocks, set your weights and investment amount — and the app instantly computes **professional-grade risk metrics**, **portfolio optimisation**, **AI-powered sentiment analysis**, and **Monte Carlo simulations** — all in one interactive dashboard.

---

## 🖼️ Dashboard Preview

### Portfolio Snapshot
![Portfolio Snapshot](D:\portfolio_risk_dashboard\Screenshots\Screenshot-1.png)

### Risk Analytics
![Risk Analytics](D:\portfolio_risk_dashboard\Screenshots\Screenshot-2.png)

### Monte Carlo Simulation
![Monte Carlo](D:\portfolio_risk_dashboard\Screenshots\Screenshot-3.png)

### Efficient Frontier
![Efficient Frontier](D:\portfolio_risk_dashboard\Screenshots\Screenshot-4.png)

---

## ✨ Features

### 📈 Investment Analytics
- Live NSE stock prices via **yfinance** (no API key needed)
- Cumulative returns vs **Nifty 50 benchmark**
- 30-day rolling volatility per stock
- Adjustable portfolio weights with real-time normalisation

### ⚠️ Risk Analytics (7 Metrics)
| Metric | What It Tells You |
|--------|-------------------|
| **Sharpe Ratio** | Return earned per unit of risk. Above 1.0 = good |
| **Sortino Ratio** | Like Sharpe but only penalises downside volatility |
| **Value at Risk (VaR 95%)** | Maximum expected daily loss on 95% of days |
| **CVaR / Expected Shortfall** | Average loss on the worst 5% of days |
| **Beta vs Nifty 50** | How much portfolio moves relative to the market |
| **Maximum Drawdown** | Worst historical peak-to-trough loss |
| **Correlation Matrix** | How much stocks move together |

### 🎲 Monte Carlo Simulation
- 500 simulated portfolio trajectories over 1 year
- Uses **Geometric Brownian Motion** based on historical mean and volatility
- Shows 5th / 50th / 95th percentile outcomes in rupees

### ⚙️ Portfolio Optimisation
- **Maximum Sharpe Ratio** optimisation using **Scipy SLSQP**
- Efficient Frontier visualisation (300 random portfolios)
- Optimal vs current weights comparison
- Long-only constraints (no short selling)

### 📰 Sentiment Analysis
- Company descriptions fetched from **Wikipedia API** (free, no key needed)
- **ProsusAI/FinBERT** — BERT model fine-tuned on financial news
- Per-stock sentiment: Positive / Negative / Neutral with confidence score

### 🤖 AI Risk Commentary
- **Mistral-7B-Instruct** via Hugging Face free inference API
- Combines all quantitative metrics + sentiment into one plain-English paragraph
- Falls back to intelligent rule-based summary if no API key provided
- **No paid API required** — works with a free Hugging Face token

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard UI | Streamlit |
| Market Data | yfinance (Yahoo Finance) |
| Data Processing | Pandas, NumPy |
| Risk & Optimisation | Scipy, custom analytics modules |
| Visualisation | Plotly |
| Sentiment Analysis | ProsusAI/FinBERT (Hugging Face Transformers) |
| Company Data | Wikipedia REST API |
| AI Commentary | Mistral-7B-Instruct (Hugging Face Inference API) |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
portfolio_risk_dashboard/
│
├── app.py                        # Main Streamlit dashboard (7 sections)
│
├── data/
│   ├── __init__.py
│   └── stock_data.py             # yfinance NSE data fetching
│
├── analytics/
│   ├── __init__.py
│   ├── returns.py                # Daily returns, volatility, correlation
│   ├── risk.py                   # VaR, Sharpe, Beta, Drawdown, Monte Carlo
│   └── optimiser.py              # Scipy portfolio optimisation + Efficient Frontier
│
├── nlp/
│   ├── __init__.py
│   ├── sentiment.py              # Wikipedia + FinBERT sentiment pipeline
│   └── ai_summary.py            # Hugging Face Mistral-7B risk commentary
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── .gitignore                    # Excludes .env and cache files
└── README.md                     # This file
```

---

## ⚡ Quick Start — Run Locally

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Ritika-8/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First install takes 5–10 minutes. PyTorch (~700MB) downloads once and caches.

### 3. Set up environment variables (optional)
```bash
cp .env.example .env
```
Open `.env` and add your free Hugging Face token:
```
HF_API_KEY=hf_your_token_here
```
Get a free token at [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New Token (Read)

> The app works **without** a token too — it generates a rule-based risk summary instead.

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🎯 How to Use

1. **Select stocks** from the sidebar dropdown (2–10 NSE stocks)
2. **Set weights** using the sliders (they auto-normalise to 100%)
3. **Set your investment amount** in rupees
4. Click **Run Analysis** — data loads in 15–20 seconds
5. Scroll through 7 sections to explore your portfolio
6. Click **Analyse Sentiment** for FinBERT analysis
7. Click **Generate Risk Summary** for AI commentary

---

## 📊 Supported NSE Stocks

| Symbol | Company |
|--------|---------|
| RELIANCE | Reliance Industries |
| TCS | Tata Consultancy Services |
| INFY | Infosys |
| HDFCBANK | HDFC Bank |
| ICICIBANK | ICICI Bank |
| WIPRO | Wipro |
| SBIN | State Bank of India |
| BAJFINANCE | Bajaj Finance |
| TATAMOTORS | Tata Motors |
| ITC | ITC Limited |
| KOTAKBANK | Kotak Mahindra Bank |
| LT | Larsen & Toubro |
| HINDUNILVR | Hindustan Unilever |
| MARUTI | Maruti Suzuki |
| SUNPHARMA | Sun Pharmaceutical |
| ASIANPAINT | Asian Paints |
| TITAN | Titan Company |
| NESTLEIND | Nestle India |
| ULTRACEMCO | UltraTech Cement |

---

## 🧠 Key Concepts Explained

### Value at Risk (VaR)
VaR at 95% confidence means: *"On 95% of trading days, the portfolio will not lose more than X%."*
This project uses **Historical Simulation** — sorting past returns and taking the 5th percentile.

### Sharpe Ratio
Measures risk-adjusted return: `(Portfolio Return − Risk Free Rate) / Volatility`
Risk-free rate used = **6.5%** (approximate Indian 10Y Government Bond yield).

### Monte Carlo Simulation
Generates 500 random future scenarios using **Geometric Brownian Motion**, based on historical return mean and standard deviation. Shows the range of possible portfolio values after 1 year.

### Portfolio Optimisation
Uses **Scipy's SLSQP algorithm** to find the combination of weights that maximises the Sharpe Ratio. Constraints: weights sum to 1.0, each weight between 0 and 1 (long-only).

### FinBERT Sentiment
**ProsusAI/FinBERT** is a BERT model fine-tuned on financial news from Reuters and analyst reports. Unlike general-purpose models, it understands financial terminology like *"bearish"*, *"NPA"*, *"write-off"*, and *"yield curve"*.

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud** (free tier).

### Deploy your own copy
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository, branch `main`, file `app.py`
5. Click **Deploy**
6. Add `HF_API_KEY` in Settings → Secrets

---

## ⚠️ Disclaimer

This dashboard is for **educational and portfolio demonstration purposes only**.
It is **not financial advice**. Past performance does not guarantee future results.
Data is sourced from Yahoo Finance via yfinance and may be delayed.

---

## 👩‍💻 About the Author

**Ritika Bajaj**
MSc Data Science — King's College London
Data Science Intern — DRPCRD (CreatorOS)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/ritika-bajaj-b0929821b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Ritika-8)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
