"""Instrument management for OANDA"""
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.oanda.client import OANDAClient

logger = __import__('logging').getLogger(__name__)

@dataclass
class Instrument:
    name: str
    display_name: str
    pip_location: int
    display_precision: int
    trade_units_precision: int
    minimum_trade_size: str
    maximum_trailing_stop_distance: str
    minimum_trailing_stop_distance: str
    maximum_position_size: str
    maximum_order_units: str
    margin_rate: str
    commission: str

class InstrumentManager:
    """Manages instrument metadata from OANDA"""

    def __init__(self, client: OANDAClient):
        self.client = client
        self._instruments: Dict[str, Instrument] = {}

    def load_instruments(self, instrument_list: List[str]) -> Dict[str, Instrument]:
        for name in instrument_list:
            try:
                details = self.client.get_instrument_details(name)
                inst_data = details.get("instrument", {})
                self._instruments[name] = Instrument(
                    name=inst_data.get("name", name),
                    display_name=inst_data.get("displayName", name),
                    pip_location=inst_data.get("pipLocation", -4),
                    display_precision=inst_data.get("displayPrecision", 5),
                    trade_units_precision=inst_data.get("tradeUnitsPrecision", 0),
                    minimum_trade_size=inst_data.get("minimumTradeSize", "1"),
                    maximum_trailing_stop_distance=inst_data.get("maximumTrailingStopDistance", "1"),
                    minimum_trailing_stop_distance=inst_data.get("minimumTrailingStopDistance", "0.0001"),
                    maximum_position_size=inst_data.get("maximumPositionSize", "10000000"),
                    maximum_order_units=inst_data.get("maximumOrderUnits", "10000000"),
                    margin_rate=inst_data.get("marginRate", "0.05"),
                    commission=inst_data.get("commission", "0")
                )
            except Exception as e:
                logger.warning(f"Failed to load instrument {name}: {e}")
        return self._instruments

    def get(self, name: str) -> Optional[Instrument]:
        return self._instruments.get(name)

    def get_pip_value(self, name: str) -> float:
        inst = self.get(name)
        if inst:
            return 10 ** inst.pip_location
        return 0.0001

    def get_spread_pips(self, name: str, typical_spread: float = 0.8) -> float:
        spreads = {
            "EUR_USD": 0.8, "GBP_USD": 1.0, "USD_JPY": 0.9,
            "USD_CHF": 1.2, "AUD_USD": 1.0, "USD_CAD": 1.2,
            "NZD_USD": 1.5, "EUR_GBP": 1.2, "EUR_JPY": 1.5,
            "GBP_JPY": 2.0
        }
        return spreads.get(name, typical_spread)

    def all(self) -> Dict[str, Instrument]:
        return self._instruments.copy()