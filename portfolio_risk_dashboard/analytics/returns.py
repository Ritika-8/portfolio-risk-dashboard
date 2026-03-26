import pandas as pd
import numpy as np

TRADING_DAYS = 252


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns).cumprod() - 1


def compute_annualised_return(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def compute_annualised_volatility(returns: pd.DataFrame) -> pd.Series:
    return returns.std() * np.sqrt(TRADING_DAYS)


def compute_portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)


def compute_rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()
