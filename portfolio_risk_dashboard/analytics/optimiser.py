import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analytics.returns import TRADING_DAYS


def portfolio_performance(weights, returns):
    ret = np.dot(weights, returns.mean()) * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return ret, vol


def negative_sharpe(weights, returns, rf=0.065):
    ret, vol = portfolio_performance(weights, returns)
    return -(ret - rf) / vol if vol > 0 else 0.0


def optimise_portfolio(returns: pd.DataFrame, rf: float = 0.065) -> dict:
    n = len(returns.columns)
    w0 = np.array([1 / n] * n)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(n))
    try:
        res = minimize(negative_sharpe, w0, args=(returns, rf),
                       method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 1000})
        opt_w = res.x
    except Exception:
        opt_w = w0
    ret, vol = portfolio_performance(opt_w, returns)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return {
        "weights": dict(zip(returns.columns, np.round(opt_w, 4))),
        "expected_annual_return": round(ret * 100, 2),
        "expected_annual_volatility": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
    }


def generate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 300) -> pd.DataFrame:
    n = len(returns.columns)
    results = []
    for _ in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret, vol = portfolio_performance(w, returns)
        sharpe = (ret - 0.065) / vol if vol > 0 else 0
        results.append({"return": ret * 100, "volatility": vol * 100, "sharpe": sharpe})
    return pd.DataFrame(results)
