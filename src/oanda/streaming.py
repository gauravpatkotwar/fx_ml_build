"""OANDA Streaming Client"""
import logging
from typing import List, Callable
from src.oanda.client import OANDAClient

logger = logging.getLogger(__name__)

class OANDAStreamingClient:
    def __init__(self, client: OANDAClient, instruments: List[str]):
        self.client = client
        self.instruments = instruments
        self._running = False
        self._callbacks = []
    def add_callback(self, callback: Callable):
        self._callbacks.append(callback)
    def start(self):
        self._running = True
        logger.info(f"Starting price stream for {self.instruments}")
        for msg in self.client.stream_prices(self.instruments):
            if not self._running:
                break
            if msg.get("type") == "PRICE":
                for cb in self._callbacks:
                    try:
                        cb(msg)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            elif msg.get("type") == "HEARTBEAT":
                logger.debug("Heartbeat received")
    def stop(self):
        self._running = False
        logger.info("Stopping price stream")