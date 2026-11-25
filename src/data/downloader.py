"""Download historical data from OANDA"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.oanda.client import OANDAClient
from src.data.storage import DataStorage
from src.data.validator import DataValidator

logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """Downloads and stores historical FX data from OANDA"""

    def __init__(
        self,
        client: OANDAClient,
        storage: DataStorage,
        validator: DataValidator,
        instruments: List[str],
        granularities: List[str],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 4
    ):
        self.client = client
        self.storage = storage
        self.validator = validator
        self.instruments = instruments
        self.granularities = granularities
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers

    def download_all(self, force_refresh: bool = False) -> Dict:
        """Download data for all instrument/granularity combinations"""
        results = {}
def _download_instrument_granularity(
        self,
        instrument: str,
        granularity: str,
        force_refresh: bool
    ) -> Dict:
        """Download data for one instrument/granularity"""
        if not force_refresh and self.storage.exists(instrument, granularity):
            logger.info(f"Skipping {instrument}_{granularity} - already exists")
            return {"status": "skipped", "reason": "exists"}

        logger.info(f"Downloading {instrument} {granularity} from {self.start_date} to {self.end_date}")

        all_candles = []
        current_start = self.start_date
        chunk_days = self._get_chunk_days(granularity)

        while current_start < self.end_date:
            current_end = min(current_start + timedelta(days=chunk_days), self.end_date)

            try:
                response = self.client.get_candles(
                    instrument=instrument,
                    granularity=granularity,
                    from_time=current_start,
                    to_time=current_end,
                    price="MBA"
                )

                candles = self._parse_candles(response, instrument, granularity)
                if candles:
                    all_candles.extend(candles)

                current_start = current_end

            except Exception as e:
                logger.error(f"Error downloading chunk: {e}")
                current_start = current_end
                continue

        if not all_candles:
            return {"status": "error", "reason": "no_data"}

        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        df = self.validator.validate_and_clean(df, instrument, granularity)

        # Store
        self.storage.write(df, instrument, granularity)

        return {
            "status": "success",
            "rows": len(df),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
            "gaps": self.validator.find_gaps(df, granularity)
        }

    def _get_chunk_days(self, granularity: str) -> int:
        """Determine chunk size based on granularity (OANDA max 5000 candles)"""
        granularity_minutes = {
            "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "H8": 480,
            "D": 1440, "W": 10080, "M": 43200
        }
        minutes = granularity_minutes.get(granularity, 60)
        return max(1, (5000 * minutes) // 1440)

    def _parse_candles(self, response: Dict, instrument: str, granularity: str) -> List[Dict]:
        """Parse OANDA candle response to standardized format"""
        candles = []
        for c in response.get("candles", []):
            if not c.get("complete", True):
                continue  # Skip incomplete candles

            mid = c.get("mid", {})
            bid = c.get("bid", {})
            ask = c.get("ask", {})

            candles.append({
                "timestamp": pd.Timestamp(c["time"]),
                "instrument": instrument,
                "granularity": granularity,
                # Mid prices
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                # Bid prices
                "bid_open": float(bid.get("o", 0)),
                "bid_high": float(bid.get("h", 0)),
                "bid_low": float(bid.get("l", 0)),
                "bid_close": float(bid.get("c", 0)),
                # Ask prices
                "ask_open": float(ask.get("o", 0)),
                "ask_high": float(ask.get("h", 0)),
                "ask_low": float(ask.get("l", 0)),
                "ask_close": float(ask.get("c", 0)),
                # Volume & spread
                "volume": int(c.get("volume", 0)),
                "spread": float(ask.get("c", 0)) - float(bid.get("c", 0)) if ask and bid else 0
            })
        return candles

        tasks = []
        for instrument in self.instruments:
            for granularity in self.granularities:
                tasks.append((instrument, granularity))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_instrument_granularity, inst, gran, force_refresh): (inst, gran)
                for inst, gran in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                instrument, granularity = futures[future]
                try:
                    result = future.result()
                    results[f"{instrument}_{granularity}"] = result
                except Exception as e:
                    logger.error(f"Failed {instrument}_{granularity}: {e}")
                    results[f"{instrument}_{granularity}"] = {"status": "error", "error": str(e)}

        return results