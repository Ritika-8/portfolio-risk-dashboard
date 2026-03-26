import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Using Mistral-7B on Hugging Face free inference API
# Free tier: no billing, no credit card, just a free HF account token
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"


def generate_risk_summary(portfolio_metrics: dict, stock_sentiments: dict) -> str:
    """
    Generate AI risk commentary using Hugging Face free inference API.
    Uses Mistral-7B-Instruct — powerful open source model, completely free.

    Get your free token at: huggingface.co -> Settings -> Access Tokens -> New Token (read)
    Add to .env as: HF_API_KEY=hf_xxxxxxxxxxxx
    """

    sentiment_lines = "\n".join(
        f"- {sym}: {data['sentiment'].upper()} (score: {data['score']})"
        for sym, data in stock_sentiments.items()
    )

    prompt = f"""<s>[INST] You are a senior portfolio risk analyst at an Indian investment firm.
Analyse this NSE equity portfolio and write a 4-5 sentence professional risk commentary.

PORTFOLIO METRICS:
- Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f} (above 1 = good)
- Sortino Ratio: {portfolio_metrics.get('sortino_ratio', 0):.2f}
- VaR 95% (daily): {portfolio_metrics.get('var_95', 0):.2%}
- VaR 99% (daily): {portfolio_metrics.get('var_99', 0):.2%}
- Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}
- Beta vs Nifty 50: {portfolio_metrics.get('beta', 0):.2f}
- Annual Volatility: {portfolio_metrics.get('volatility', 0):.2%}
- Annual Return: {portfolio_metrics.get('annual_return', 0):.2%}

STOCK SENTIMENTS:
{sentiment_lines}

Write a concise professional commentary covering:
1. Overall risk level (Low/Moderate/High) with reason
2. Key risks based on the numbers
3. One specific recommendation for Indian equity context

Keep it under 5 sentences. Be direct and professional. [/INST]"""

    if not HF_API_KEY:
        return generate_rule_based_summary(portfolio_metrics, stock_sentiments)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "").strip()
                if text:
                    return text
        elif response.status_code == 503:
            return "Model is loading on Hugging Face servers (can take 20-30 seconds on first call). Please click Generate again in a moment."
        else:
            return generate_rule_based_summary(portfolio_metrics, stock_sentiments)
    except Exception as e:
        return generate_rule_based_summary(portfolio_metrics, stock_sentiments)


def generate_rule_based_summary(portfolio_metrics: dict, stock_sentiments: dict) -> str:
    """
    Fallback: generate a rule-based summary when AI is unavailable.
    No API needed — uses the numbers directly.
    """
    sharpe = portfolio_metrics.get("sharpe_ratio", 0)
    beta = portfolio_metrics.get("beta", 1)
    drawdown = portfolio_metrics.get("max_drawdown", 0)
    var_95 = portfolio_metrics.get("var_95", 0)
    annual_return = portfolio_metrics.get("annual_return", 0)
    volatility = portfolio_metrics.get("volatility", 0)

    # Determine risk level
    if sharpe >= 1.5 and abs(drawdown) < 0.15:
        risk_level = "LOW"
        risk_desc = "The portfolio demonstrates strong risk-adjusted returns with manageable drawdowns."
    elif sharpe >= 1.0 and abs(drawdown) < 0.25:
        risk_level = "MODERATE"
        risk_desc = "The portfolio shows acceptable risk-adjusted performance with moderate volatility."
    else:
        risk_level = "HIGH"
        risk_desc = "The portfolio carries elevated risk relative to its returns."

    # Beta commentary
    if beta > 1.3:
        beta_text = f"With a Beta of {beta:.2f}, this portfolio is significantly more volatile than the Nifty 50 — expect amplified swings in both directions."
    elif beta > 1.0:
        beta_text = f"A Beta of {beta:.2f} indicates slightly higher market sensitivity than the Nifty 50 benchmark."
    else:
        beta_text = f"A Beta of {beta:.2f} suggests this portfolio is more defensive than the broader Nifty 50 market."

    # Sharpe commentary
    if sharpe >= 2:
        sharpe_text = f"The Sharpe Ratio of {sharpe:.2f} is excellent, indicating strong returns for the risk taken."
    elif sharpe >= 1:
        sharpe_text = f"The Sharpe Ratio of {sharpe:.2f} is acceptable — the portfolio earns reasonable excess returns over the 6.5% risk-free rate."
    else:
        sharpe_text = f"The Sharpe Ratio of {sharpe:.2f} is below 1.0 — consider replacing weaker holdings to improve risk-adjusted returns."

    # Recommendation
    if beta > 1.2:
        recommendation = "Consider adding defensive stocks like FMCG or Pharma (HUL, Sun Pharma) to reduce overall Beta."
    elif sharpe < 1.0:
        recommendation = "Review and replace the lowest Sharpe stocks in the portfolio to improve overall risk-adjusted performance."
    else:
        recommendation = "The portfolio is well-positioned — consider a quarterly rebalance to maintain optimal weights."

    # Sentiment summary
    neg_stocks = [sym for sym, d in stock_sentiments.items() if d["sentiment"] == "negative"]
    sentiment_note = ""
    if neg_stocks:
        sentiment_note = f" Note: {', '.join(neg_stocks)} show negative sentiment signals — monitor for near-term headwinds."

    return (
        f"RISK LEVEL: {risk_level} — {risk_desc} "
        f"{beta_text} "
        f"{sharpe_text} "
        f"The 1-day VaR at 95% confidence is {var_95:.2%}, meaning on a bad day your portfolio could lose up to "
        f"this amount. {recommendation}.{sentiment_note}"
    )
