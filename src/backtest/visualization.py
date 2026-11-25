"""Backtest visualization"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(equity_curve: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(equity_curve.index, equity_curve["equity"])
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(True)
    
    returns = equity_curve["equity"].pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    axes[1].set_title("Drawdown (%)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(True)
    
    axes[2].plot(equity_curve.index, equity_curve["equity"].pct_change().dropna().rolling(100).mean() * 100)
    axes[2].set_title("Rolling 100-bar Return (%)")
    axes[2].set_ylabel("Return (%)")
    axes[2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_trade_analysis(trades: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(trades["pnl"], bins=30, edgecolor='black')
    axes[0, 0].set_title("PnL Distribution")
    axes[0, 0].set_xlabel("PnL ($)")
    
    axes[0, 1].hist(trades["duration"], bins=30, edgecolor='black')
    axes[0, 1].set_title("Trade Duration Distribution")
    axes[0, 1].set_xlabel("Duration (hours)")
    
    cumulative_pnl = trades["pnl"].cumsum()
    axes[1, 0].plot(cumulative_pnl)
    axes[1, 0].set_title("Cumulative PnL")
    axes[1, 0].set_ylabel("Cumulative PnL ($)")
    
    trades["pnl_pct"].plot(kind='box', ax=axes[1, 1])
    axes[1, 1].set_title("PnL % Box Plot")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()