"""Main backtesting engine"""
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

from src.backtest.portfolio import Portfolio
from src.backtest.execution import ExecutionModel
from src.backtest.orders import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: Dict):
        self.max_position_pct = config.get("max_position_size_pct", 0.1)
        self.max_portfolio_heat = config.get("max_portfolio_heat", 0.2)
        self.max_correlated = config.get("max_correlated_positions", 3)
        self.corr_threshold = config.get("correlation_threshold", 0.7)
        self.daily_loss_limit = config.get("daily_loss_limit_pct", 0.03)
        self.weekly_loss_limit = config.get("weekly_loss_limit_pct", 0.06)
        self.max_dd_limit = config.get("max_drawdown_pct", 0.15)

    def check_limits(self, portfolio: Portfolio, prices: Dict) -> bool:
        equity = portfolio.get_equity(prices)
        if portfolio.equity_curve:
            peak = max(e["equity"] for e in portfolio.equity_curve)
            current_dd = (peak - equity) / peak
            if current_dd > self.max_dd_limit:
                logger.error(f"Max drawdown exceeded: {current_dd:.2%}")
                return False
        for inst, pos in portfolio.positions.items():
            if inst in prices:
                mid = (prices[inst]["bid"] + prices[inst]["ask"]) / 2
                notional = abs(pos.net_units) * mid * 100000
                if notional / equity > self.max_position_pct:
                    logger.warning(f"Position size limit for {inst}")
                    return False
        return True

    def calculate_position_size(self, portfolio: Portfolio, prices: Dict,
                                signal_strength: float, atr: float, entry_price: float) -> float:
        equity = portfolio.get_equity(prices)
        risk_per_trade = equity * 0.01
        stop_distance = atr * 2
        units = risk_per_trade / (stop_distance * entry_price * 100000)
        units *= signal_strength
        max_units = (equity * self.max_position_pct) / (entry_price * 100000)
        units = min(units, max_units)
        return max(0, units)