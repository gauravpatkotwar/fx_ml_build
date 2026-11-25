"""Backtest performance metrics"""
import numpy as np
import pandas as pd

def calculate_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
    """Calculate comprehensive backtest metrics"""
    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1 if len(returns) > 0 else 0
    ann_vol = returns.std() * np.sqrt(252 * 24)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    if len(trades) > 0:
        win_rate = (trades["pnl"] > 0).mean()
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean()
        avg_loss = trades[trades["pnl"] < 0]["pnl"].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        expectancy = trades["pnl"].mean()
    else:
        win_rate = avg_win = avg_loss = profit_factor = expectancy = 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": ann_return / (returns[returns < 0].std() * np.sqrt(252 * 24)) if len(returns[returns < 0]) > 0 else 0,
        "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "num_trades": len(trades)
    }