import numpy as np
import pandas as pd
from analytics.returns import TRADING_DAYS


def compute_sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    if portfolio_returns.empty or portfolio_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = portfolio_returns - daily_rf
    return float((excess.mean() / portfolio_returns.std()) * np.sqrt(TRADING_DAYS))


def compute_var(portfolio_returns: pd.Series, confidence: float = 0.95) -> dict:
    if portfolio_returns.empty:
        return {"var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0}
    sorted_r = portfolio_returns.sort_values()
    var_95 = sorted_r.quantile(1 - confidence)
    var_99 = sorted_r.quantile(0.01)
    worst = sorted_r[sorted_r <= var_95]
    cvar = worst.mean() if not worst.empty else var_95
    return {
        "var_95": abs(float(var_95)),
        "var_99": abs(float(var_99)),
        "cvar_95": abs(float(cvar)),
    }


def compute_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if portfolio_returns.empty or benchmark_returns.empty:
        return 1.0
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 10:
        return 1.0
    aligned.columns = ["portfolio", "benchmark"]
    try:
        cov = np.cov(aligned["portfolio"], aligned["benchmark"])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0
    except Exception:
        return 1.0


def compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    if portfolio_returns.empty:
        return 0.0
    cum = (1 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def monte_carlo_simulation(
    portfolio_returns: pd.Series,
    num_simulations: int = 500,
    num_days: int = 252,
    initial_value: float = 100000,
) -> np.ndarray:
    if portfolio_returns.empty:
        return np.full((num_simulations, num_days), initial_value)
    mean_r = portfolio_returns.mean()
    std_r = portfolio_returns.std()
    if std_r == 0 or np.isnan(std_r):
        return np.full((num_simulations, num_days), initial_value)
    sims = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        dr = np.random.normal(mean_r, std_r, num_days)
        sims[i] = initial_value * np.cumprod(1 + dr)
    return sims


def compute_sortino_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    if portfolio_returns.empty:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = portfolio_returns - daily_rf
    down = portfolio_returns[portfolio_returns < 0]
    if down.empty or down.std() == 0:
        return 0.0
    return float((excess.mean() * TRADING_DAYS) / (down.std() * np.sqrt(TRADING_DAYS)))
