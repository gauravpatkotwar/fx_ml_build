"""Risk management for paper trading"""
from typing import Dict

class RiskManager:
    def __init__(self, config: Dict):
        self.max_position_pct = config.get('max_position_size_pct', 0.1)
        self.max_portfolio_heat = config.get('max_portfolio_heat', 0.2)
        self.daily_loss_limit = config.get('daily_loss_limit_pct', 0.03)
    def calculate_position_size(self, portfolio, prices, signal_strength, atr, entry_price):
        equity = portfolio.get_equity(prices)
        risk_per_trade = equity * 0.01
        stop_distance = atr * 2
        units = risk_per_trade / (stop_distance * entry_price * 100000)
        units *= signal_strength
        max_units = (equity * self.max_position_pct) / (entry_price * 100000)
        return max(0, min(units, max_units))
