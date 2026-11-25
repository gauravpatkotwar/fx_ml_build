"""Data validation and cleaning"""
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and cleans OHLCV data"""

    def __init__(self, max_gap_hours: int = 48, max_spread_pips: float = 50):
        self.max_gap_hours = max_gap_hours
        self.max_spread_pips = max_spread_pips

    def validate_and_clean(self, df: pd.DataFrame, instrument: str, granularity: str) -> pd.DataFrame:
        """Validate and clean the dataframe"""
        df = df.copy()
        original_len = len(df)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Check for required columns
        required = ["timestamp", "open", "high", "low", "close", "volume", "spread"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Validate OHLC relationships
        df = self._validate_ohlc(df)

        # Remove outliers
        df = self._remove_outliers(df)

        # Forward fill small gaps
        df = self._fill_gaps(df, granularity)

        logger.info(f"Validated {instrument} {granularity}: {original_len} -> {len(df)} rows")
        return df

    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships"""
        # High >= Low
        mask = df["high"] >= df["low"]
        df = df[mask].copy()

        # High >= Open, Close
        mask = (df["high"] >= df["open"]) & (df["high"] >= df["close"])
        df = df[mask].copy()

        # Low <= Open, Close
        mask = (df["low"] <= df["open"]) & (df["low"] <= df["close"])
        df = df[mask].copy()

        return df.reset_index(drop=True)

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers using IQR method"""
        for col in ["open", "high", "low", "close"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)].copy()

        # Remove extreme spread outliers
        Q1 = df["spread"].quantile(0.25)
        Q3 = df["spread"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 5 * IQR
        df = df[df["spread"] <= upper].copy()

        return df.reset_index(drop=True)

    def _fill_gaps(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Forward fill small gaps in time series"""
        if len(df) < 2:
            return df

        # Expected frequency
        freq_map = {
            "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1H", "H4": "4H", "D": "1D"
        }
        freq = freq_map.get(granularity, "1H")

        # Create complete time range
        full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=freq)
        full_df = pd.DataFrame({"timestamp": full_range})

        # Merge
        df = full_df.merge(df, on="timestamp", how="left")

        # Forward fill price columns (max 3 periods)
        price_cols = ["open", "high", "low", "close", "bid_open", "bid_high", "bid_low", "bid_close",
                      "ask_open", "ask_high", "ask_low", "ask_close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

        # Volume and spread fill with 0
        for col in ["volume", "spread"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df.reset_index(drop=True)

    def find_gaps(self, df: pd.DataFrame, granularity: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find gaps in the time series"""
        freq_map = {
            "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1H", "H4": "4H", "D": "1D"
        }
        freq = freq_map.get(granularity, "1H")

        expected = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=freq)
        missing = expected.difference(df["timestamp"])

        if len(missing) == 0:
            return []

        # Group consecutive missing timestamps
        gaps = []
        gap_start = missing[0]
        prev = missing[0]

        for ts in missing[1:]:
            if (ts - prev) > pd.Timedelta(freq):
                gaps.append((gap_start, prev))
                gap_start = ts
            prev = ts

        gaps.append((gap_start, prev))
        return gaps