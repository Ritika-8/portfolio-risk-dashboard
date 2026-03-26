import yfinance as yf
import pandas as pd

NSE_STOCKS = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "WIPRO": "Wipro",
    "AXISBANK": "Axis Bank",
    "SBIN": "State Bank of India",
    "TATAMOTORS": "Tata Motors",
    "ITC": "ITC Limited",
    "BAJFINANCE": "Bajaj Finance",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen and Toubro",
    "HINDUNILVR": "Hindustan Unilever",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharmaceutical",
    "ASIANPAINT": "Asian Paints",
    "TITAN": "Titan Company",
    "NESTLEIND": "Nestle India",
    "ULTRACEMCO": "UltraTech Cement",
}

PERIOD_MAP = {1: "1y", 2: "2y", 3: "5y"}


def fetch_stock_data(symbols: list, period_years: int = 2) -> pd.DataFrame:
    """Download historical closing prices for NSE stocks using ticker.history()."""
    period_str = PERIOD_MAP.get(period_years, "2y")
    all_data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=period_str)
            if not df.empty:
                series = df["Close"].copy()
                series.index = series.index.tz_localize(None)
                all_data[symbol] = series
            else:
                print(f"Warning: No data for {symbol}")
        except Exception as e:
            print(f"Warning: Could not fetch {symbol}: {e}")

    if not all_data:
        return pd.DataFrame()

    prices = pd.DataFrame(all_data)
    prices.dropna(how="all", inplace=True)
    return prices


def fetch_benchmark(period_years: int = 2) -> pd.Series:
    """Download Nifty 50 index as market benchmark."""
    period_str = PERIOD_MAP.get(period_years, "2y")
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period=period_str)
        if df.empty:
            return pd.Series(dtype=float)
        series = df["Close"].copy()
        series.index = series.index.tz_localize(None)
        return series
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 50: {e}")
        return pd.Series(dtype=float)
