import requests
from transformers import pipeline

# Load FinBERT once and reuse
_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
    return _sentiment_pipeline


def fetch_wikipedia_summary(company_name: str) -> str:
    """
    Fetch company description from Wikipedia.
    Completely free — no API key needed.
    """
    try:
        name = company_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")
    except Exception:
        pass
    return ""


def get_stock_sentiment(symbol: str, company_name: str) -> dict:
    """
    Get sentiment for a stock using Wikipedia summary + FinBERT.
    No API key needed. Completely free.
    """
    text = fetch_wikipedia_summary(company_name)

    # Fallback if Wikipedia has no data
    if not text or len(text) < 30:
        return {
            "symbol": symbol,
            "sentiment": "neutral",
            "score": 0.5,
            "articles": [],
            "counts": {"positive": 0, "negative": 0, "neutral": 1},
        }

    pipe = get_sentiment_pipeline()
    try:
        output = pipe(text[:512])[0]
        label = output["label"].lower()
        score = round(output["score"], 2)
    except Exception:
        label, score = "neutral", 0.5

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    counts[label] = 1

    # Score = positive confidence
    display_score = score if label == "positive" else round(1 - score, 2)

    return {
        "symbol": symbol,
        "sentiment": label,
        "score": display_score,
        "articles": [{
            "title": f"Wikipedia: {company_name}",
            "url": f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}",
        }],
        "counts": counts,
    }
