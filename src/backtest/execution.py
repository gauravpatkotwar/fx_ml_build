"""Realistic execution simulation"""
import logging
import numpy as np
from typing import Dict
from src.backtest.orders import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class ExecutionModel:
    """Realistic execution simulation with slippage, spread, commission"""

    def __init__(self, config: Dict):
        self.base_spread_pips = config.get("base_spread_pips", 0.8)
        self.commission_per_million = config.get("commission_per_million", 5.0)
        self.latency_ms = config.get("latency_ms", 100)
        self.partial_fill_prob = config.get("partial_fill_prob", 0.02)
        self.reject_prob = config.get("reject_prob", 0.001)
        self.slippage_model = config.get("slippage_model", "square_root")

        self.spreads = {
            "EUR_USD": 0.8, "GBP_USD": 1.0, "USD_JPY": 0.9,
            "USD_CHF": 1.2, "AUD_USD": 1.0, "USD_CAD": 1.2,
            "NZD_USD": 1.5, "EUR_GBP": 1.2, "EUR_JPY": 1.5,
            "GBP_JPY": 2.0
        }

    def get_spread(self, instrument: str, volatility: float, hour: int) -> float:
        """Dynamic spread based on volatility and session"""
        base = self.spreads.get(instrument, 1.5)

        if 8 <= hour < 16:  # London
            session_mult = 1.0
        elif 13 <= hour < 21:  # NY
            session_mult = 1.0
        elif 0 <= hour < 8:  # Asian
            session_mult = 1.5
        else:
            session_mult = 2.0

        vol_mult = 1 + min(volatility * 10, 2.0)
        return base * session_mult * vol_mult / 10000

    def calculate_slippage(self, instrument: str, units: float, volatility: float) -> float:
        """Square-root market impact model"""
        if self.slippage_model == "square_root":
            notional = abs(units) * 100000
            adv = 1e9
            participation = notional / adv
            base_spread = self.spreads.get(instrument, 1.5) / 10000
            return base_spread * np.sqrt(max(participation, 0.0001)) * (1 + volatility * 5)
        return 0

    def execute_order(
        self,
        order: Order,
        bid: float,
        ask: float,
        mid: float,
        volatility: float,
        timestamp
    ) -> Order:
        """Simulate order execution"""
        if np.random.random() < self.reject_prob:
            order.status = OrderStatus.REJECTED
            return order

        spread = ask - bid
        slippage = self.calculate_slippage(order.instrument, order.units, volatility)

        if order.type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                fill_price = ask + slippage
            else:
                fill_price = bid - slippage

        elif order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= ask:
                fill_price = min(order.price, ask + slippage)
            elif order.side == OrderSide.SELL and order.price <= bid:
                fill_price = max(order.price, bid - slippage)
            else:
                order.status = OrderStatus.PENDING
                return order

        elif order.type == OrderType.STOP:
            if order.side == OrderSide.BUY and order.stop_price <= ask:
                fill_price = ask + slippage
            elif order.side == OrderSide.SELL and order.stop_price >= bid:
                fill_price = bid - slippage
            else:
                order.status = OrderStatus.PENDING
                return order

        fill_units = order.units
        if np.random.random() < self.partial_fill_prob:
            fill_units = order.units * np.random.uniform(0.1, 0.9)
            order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.FILLED

        order.filled_units = fill_units
        order.avg_fill_price = fill_price
        order.timestamp = timestamp

        return order

    def calculate_commission(self, units: float, price: float) -> float:
        notional = abs(units) * price * 100000
        return (notional / 1_000_000) * self.commission_per_million