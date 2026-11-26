"""OANDA REST and streaming API client.

Wraps the ``oandapyV20`` SDK with rate limiting, retry/backoff and a small
typed surface that the rest of the codebase depends on:

* :class:`OANDAConfig`  - connection settings (API key, account, hosts)
* :class:`OANDAClient`  - REST + streaming operations
* :func:`create_client_from_env` - factory that reads ``.env`` / environment
"""
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import oandapyV20
from dotenv import load_dotenv
from oandapyV20.endpoints.accounts import AccountDetails, AccountSummary
from oandapyV20.endpoints.instruments import Instruments, InstrumentsCandles
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.pricing import Pricing, PricingStream
from oandapyV20.endpoints.trades import TradeClose, TradesList

logger = logging.getLogger(__name__)

PRACTICE_REST_HOST = "https://api-fxpractice.oanda.com"
PRACTICE_STREAM_HOST = "https://stream-fxpractice.oanda.com"
LIVE_REST_HOST = "https://api-fxtrade.oanda.com"
LIVE_STREAM_HOST = "https://stream-fxtrade.oanda.com"

#: OANDA rejects candle requests asking for more than 5000 candles.
MAX_CANDLES_PER_REQUEST = 5000

#: Default number of candles returned when no explicit time range is given.
DEFAULT_CANDLE_COUNT = 500


class OANDAConfigurationError(RuntimeError):
    """Raised when credentials or connection settings are missing/invalid."""


class OANDAAPIError(RuntimeError):
    """Raised when an OANDA request fails after all retries."""


@dataclass
class OANDAConfig:
    """Connection settings for the OANDA API."""

    api_key: str
    account_id: str
    environment: str = "practice"
    rest_host: str = ""
    stream_host: str = ""
    requests_per_second: int = 100
    burst: int = 150
    max_retries: int = 3
    retry_backoff: float = 1.0
    timeout: int = 30

    def __post_init__(self) -> None:
        if not self.api_key:
            raise OANDAConfigurationError("OANDA api_key is required")
        if not self.account_id:
            raise OANDAConfigurationError("OANDA account_id is required")

        practice = self.environment.lower() != "live"
        if not self.rest_host:
            self.rest_host = PRACTICE_REST_HOST if practice else LIVE_REST_HOST
        if not self.stream_host:
            self.stream_host = PRACTICE_STREAM_HOST if practice else LIVE_STREAM_HOST

    @property
    def is_live(self) -> bool:
        return self.environment.lower() == "live"

    @classmethod
    def from_env(cls) -> "OANDAConfig":
        """Build a config from ``.env`` and the process environment."""
        load_dotenv()

        api_key = os.getenv("OANDA_API_KEY", "").strip()
        account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
        environment = os.getenv("OANDA_ENVIRONMENT", "practice").strip() or "practice"

        missing = [
            name
            for name, value in (
                ("OANDA_API_KEY", api_key),
                ("OANDA_ACCOUNT_ID", account_id),
            )
            if not value
        ]
        if missing:
            raise OANDAConfigurationError(
                f"Missing OANDA credentials: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your practice account values."
            )

        return cls(api_key=api_key, account_id=account_id, environment=environment)


class RateLimiter:
    """Sliding-window rate limiter honouring OANDA's requests-per-second cap."""

    def __init__(self, requests_per_second: int = 100, burst: int = 150):
        self.requests_per_second = max(1, requests_per_second)
        self.burst = max(1, burst)
        self._calls: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until another request is allowed."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 1.0:
                    self._calls.popleft()
                if len(self._calls) < self.burst:
                    self._calls.append(now)
                    return
                wait = 1.0 - (now - self._calls[0])
            time.sleep(max(0.001, wait))



class OANDAClient:
    """High level client for the OANDA v3 REST API and price stream."""

    def __init__(self, config: OANDAConfig):
        self.config = config
        self._rate_limiter = RateLimiter(config.requests_per_second, config.burst)
        self._instrument_cache: Dict[str, Dict] = {}

        # oandapyV20 accepts a custom hostname in place of "practice"/"live".
        default_host = LIVE_REST_HOST if config.is_live else PRACTICE_REST_HOST
        environment = (
            config.rest_host if config.rest_host != default_host else config.environment
        )
        self._api = oandapyV20.API(
            access_token=config.api_key,
            environment=environment,
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "OANDA client initialised (environment=%s, account=%s)",
            config.environment,
            config.account_id,
        )

    # ------------------------------------------------------------------
    # Low level request handling
    # ------------------------------------------------------------------
    def _request(self, endpoint: Any) -> Dict:
        """Execute an endpoint with rate limiting and retry/backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            self._rate_limiter.acquire()
            try:
                response = self._api.request(endpoint)
                data = getattr(response, "data", None)
                return data if data is not None else {}
            except Exception as exc:  # noqa: BLE001 - oandapyV20 raises V20Error
                last_error = exc
                status = getattr(getattr(exc, "httpResponse", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500

                if not retryable or attempt >= self.config.max_retries:
                    break

                backoff = self.config.retry_backoff * (2 ** attempt)
                logger.warning(
                    "OANDA request failed (status=%s, attempt=%s/%s), retrying in %.1fs: %s",
                    status,
                    attempt + 1,
                    self.config.max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

        raise OANDAAPIError(f"OANDA request failed: {last_error}") from last_error

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------
    def get_instrument_details(self, instrument: str) -> Dict:
        """Return metadata for a single instrument."""
        if instrument not in self._instrument_cache:
            response = self._request(Instruments(instrument=instrument))
            self._instrument_cache[instrument] = response

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        price: str = "MBA",
        count: Optional[int] = None,
    ) -> Dict:
        """Fetch historical candles.

        ``from_time``/``to_time`` are mutually exclusive with ``count`` in the
        OANDA API, so an explicit time range always wins.
        """
        params: Dict[str, Any] = {"granularity": granularity, "price": price}

        if from_time is not None or to_time is not None:
            if from_time is not None:
                params["from"] = self._format_time(from_time)
            if to_time is not None:
                params["to"] = self._format_time(to_time)
        else:
            params["count"] = min(
                count or DEFAULT_CANDLE_COUNT, MAX_CANDLES_PER_REQUEST
            )

        response = self._request(
            InstrumentsCandles(instrument=instrument, params=params)
        )
        return response

    def get_pricing(self, instruments: List[str]) -> Dict:
        """Return the current bid/ask snapshot for the given instruments."""
        response = self._request(Pricing(accountID=self.config.account_id))
        prices = response.get("prices", [])
        wanted = set(instruments)
        if wanted:
            prices = [p for p in prices if p.get("instrument") in wanted]
        return {"prices": prices}

    def stream_prices(self, instruments: List[str]) -> Iterator[Dict]:
        """Yield streaming price messages for ``instruments``.

        Each yielded message is a dict with a ``type`` key (``PRICE`` or
        ``HEARTBEAT``) plus the raw payload from OANDA.
        """
        params = {"instruments": ",".join(instruments)}
        stream = PricingStream(accountID=self.config.account_id, params=params)

        for response in stream:
            message_type = getattr(response, "type", None)
            data = getattr(response, "data", None) or {}
            if message_type:
                data = {"type": message_type, **data}
            yield data

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_account_summary(self) -> Dict:
        """Return the account balance/equity/margin summary."""
        return self._request(AccountSummary(accountID=self.config.account_id))

    def get_account_details(self) -> Dict:
        """Return full account details including open positions."""
        return self._request(AccountDetails(accountID=self.config.account_id))

    def get_positions(self) -> List[Dict]:
        """Return currently open positions."""
        details = self.get_account_details()
        return details.get("positions", [])

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------
    def get_open_trades(self) -> List[Dict]:
        """Return currently open trades."""
        response = self._request(TradesList(accountID=self.config.account_id))
        return response.get("trades", [])

    def create_order(
        self,
        instrument: str,
        units: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: str = "FOK",
    ) -> Dict:
        """Place an order on the account.

        Positive ``units`` buys, negative ``units`` sells.
        """
        order: Dict[str, Any] = {
            "type": order_type,
            "instrument": instrument,
            "units": str(units),
            "timeInForce": time_in_force,
        }
        if price is not None:
            order["price"] = str(price)
        if stop_loss is not None:
            order["stopLossOnFilled"] = {"price": str(stop_loss)}
        if take_profit is not None:
            order["takeProfitOnFilled"] = {"price": str(take_profit)}
        if trailing_stop is not None:
            order["trailingStopOnFilled"] = {"distance": str(trailing_stop)}

        return self._request(
            OrderCreate(accountID=self.config.account_id, data={"order": order})
        )

    def close_trade(self, trade_id: str, units: Optional[float] = None) -> Dict:
        """Close a trade, fully or partially."""
        data = None
        if units is not None:
            data = {"units": str(units)}
        return self._request(
            TradeClose(accountID=self.config.account_id, tradeID=trade_id, data=data)
        )

    def close_position(self, instrument: str) -> Dict:
        """Flatten the position on ``instrument``."""
        return self.create_order(instrument=instrument, units=0, order_type="MARKET")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_time(value: datetime) -> str:
        """Format a datetime as an RFC3339 string OANDA understands."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_client_from_env() -> OANDAClient:
    """Create an :class:`OANDAClient` using environment credentials."""
    return OANDAClient(OANDAConfig.from_env())


    def get_instruments(self) -> List[Dict]:
        """Return metadata for every tradable instrument."""
        response = self._request(Instruments(instrument="*"))
        return response.get("instruments", [])
