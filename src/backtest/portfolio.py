"""Portfolio management"""
import logging
from typing import Dict, List
from datetime import datetime
from collections import defaultdict
from src.backtest.orders import Position, Trade, Order, OrderSide

logger = logging.getLogger(__name__)


class Portfolio:
    """Multi-currency portfolio management"""

    def __init__(self, initial_capital: float, base_currency: str = "USD", max_leverage: float = 30):
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.max_leverage = max_leverage
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self._trade_id = 0

    def get_equity(self, prices: Dict[str, Dict[str, float]]) -> float:
        """Calculate total equity"""
        equity = self.cash
        for instrument, position in self.positions.items():
            if instrument in prices:
                bid = prices[instrument]["bid"]
                ask = prices[instrument]["ask"]

                if position.long_units > 0:
                    equity += position.long_units * (bid - position.long_avg_price) * 100000
                if position.short_units > 0:
                    equity += position.short_units * (position.short_avg_price - ask) * 100000
        return equity

    def get_margin_used(self, prices: Dict[str, Dict[str, float]]) -> float:
        """Calculate margin used"""
        margin = 0
        for instrument, position in self.positions.items():
            if instrument in prices:
                mid = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
                notional = abs(position.net_units) * mid * 100000
                margin += notional / self.max_leverage
        return margin

    def get_free_margin(self, prices: Dict[str, Dict[str, float]]) -> float:
        return self.get_equity(prices) - self.get_margin_used(prices)

    def update_position(self, order: Order, execution: Order, commission: float):
        """Update portfolio after order execution"""
        inst = order.instrument
        if inst not in self.positions:
            self.positions[inst] = Position(instrument=inst)

        pos = self.positions[inst]
        units = execution.filled_units
        price = execution.avg_fill_price

        if order.side == OrderSide.BUY:
            if pos.short_units > 0:
                close_units = min(units, pos.short_units)
                pnl = close_units * (pos.short_avg_price - price) * 100000
                pos.short_units -= close_units
                pos.realized_pl += pnl - commission
                units -= close_units
                if pos.short_units == 0:
                    pos.short_avg_price = 0

            if units > 0:
                total_cost = pos.long_units * pos.long_avg_price + units * price
                pos.long_units += units
                pos.long_avg_price = total_cost / pos.long_units if pos.long_units > 0 else 0

        else:
            if pos.long_units > 0:
                close_units = min(units, pos.long_units)
                pnl = close_units * (price - pos.long_avg_price) * 100000
                pos.long_units -= close_units
                pos.realized_pl += pnl - commission
                units -= close_units
                if pos.long_units == 0:
                    pos.long_avg_price = 0

            if units > 0:
                total_cost = pos.short_units * pos.short_avg_price + units * price
                pos.short_units += units
                pos.short_avg_price = total_cost / pos.short_units if pos.short_units > 0 else 0

        self.cash -= commission

    def record_equity(self, timestamp: datetime, prices: Dict[str, Dict[str, float]]):
        equity = self.get_equity(prices)
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self.cash,
            "margin_used": self.get_margin_used(prices),
            "free_margin": self.get_free_margin(prices),
            "unrealized_pl": equity - self.cash - self.initial_capital
        })