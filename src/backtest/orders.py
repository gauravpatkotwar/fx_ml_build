"""Order types and execution"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: str
    instrument: str
    side: OrderSide
    type: OrderType
    units: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    filled_units: float = 0
    avg_fill_price: float = 0
    timestamp: Optional[datetime] = None
    tag: str = ""


@dataclass
class Position:
    instrument: str
    long_units: float = 0
    short_units: float = 0
    long_avg_price: float = 0
    short_avg_price: float = 0
    unrealized_pl: float = 0
    realized_pl: float = 0

    @property
    def net_units(self) -> float:
        return self.long_units - self.short_units

    @property
    def is_long(self) -> bool:
        return self.net_units > 0

    @property
    def is_short(self) -> bool:
        return self.net_units < 0


@dataclass
class Trade:
    id: str
    instrument: str
    side: OrderSide
    entry_price: float
    exit_price: float
    units: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    tag: str = ""