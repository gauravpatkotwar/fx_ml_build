#!/usr/bin/env python3
"""
Memory-safe Dukascopy tick data loader.
Downloads and processes tick data year by year to stay under 4GB RAM.
Aggregates ticks into 1-hour microstructure bars.
"""

import asyncio
import gc
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import tick_vault
except ImportError:
    print("tick-vault not installed. Run: pip install tick-vault")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DukascopyTickLoader:
    """Downloads and processes Dukascopy tick data year by year."""

    def __init__(
        self,
        symbol: str = "EURUSD",
        cache_dir: str = "/tmp/dukascopy_cache",
        output_dir: str = "/tmp",
        bar_freq: str = "1H",
        years: Optional[List[int]] = None,
    ):
        self.symbol = symbol
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.bar_freq = bar_freq
        self.years = years or list(range(2011, 2027))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def download_year(self, year: int):
        logger.info(f"Downloading {self.symbol} ticks for {year}...")
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        try:
            await tick_vault.download_range(
                symbol=self.symbol, start=start, end=end, output_dir=str(self.cache_dir)
            )
            df = tick_vault.read_tick_data(
                symbol=self.symbol, start=start, end=end, data_dir=str(self.cache_dir)
            )
            if df is None or len(df) == 0:
                logger.warning(f"No tick data for {year}")
                return None
            for col in ["timestamp", "ask", "bid", "ask_volume", "bid_volume"]:
                if col not in df.columns:
                    logger.error(f"Missing column: {col}")
                    return None
            logger.info(f"Downloaded {len(df):,} ticks for {year}")
            return df
        except Exception as e:
            logger.error(f"Failed to download {year}: {e}")
            return None

    def aggregate_to_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()
        df = ticks.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df["mid"] = (df["ask"] + df["bid"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = df["spread"] / df["mid"]
        df["vol_imbalance"] = (df["bid_volume"] - df["ask_volume"]) / (
            df["bid_volume"] + df["ask_volume"] + 1e-10
        )
        bars = pd.DataFrame(index=df.resample(self.bar_freq).first().index)
        bars.index.name = "timestamp"
        bars["mid_open"] = df["mid"].resample(self.bar_freq).first()
        bars["mid_high"] = df["mid"].resample(self.bar_freq).max()
        bars["mid_low"] = df["mid"].resample(self.bar_freq).min()
        bars["mid_close"] = df["mid"].resample(self.bar_freq).last()
        bars["spread_mean"] = df["spread"].resample(self.bar_freq).mean()
        bars["spread_max"] = df["spread"].resample(self.bar_freq).max()
        bars["spread_pct_mean"] = df["spread_pct"].resample(self.bar_freq).mean()
        bars["tick_count"] = df["mid"].resample(self.bar_freq).count()
        bars["vol_imbalance_mean"] = df["vol_imbalance"].resample(self.bar_freq).mean()
        bars["vol_imbalance_std"] = df["vol_imbalance"].resample(self.bar_freq).std()
        df["log_ret"] = np.log(df["mid"]).diff()
        realized_vol = df["log_ret"].resample(self.bar_freq).apply(
            lambda x: np.sqrt((x**2).sum()) if len(x) > 1 else 0.0
        )
        bars["realized_vol"] = realized_vol
        bars["bid_volume_sum"] = df["bid_volume"].resample(self.bar_freq).sum()
        bars["ask_volume_sum"] = df["ask_volume"].resample(self.bar_freq).sum()
        bars["total_volume"] = bars["bid_volume_sum"] + bars["ask_volume_sum"]
        df["mid_x_vol"] = df["mid"] * (df["bid_volume"] + df["ask_volume"])
        vwap_num = df["mid_x_vol"].resample(self.bar_freq).sum()
        vwap_den = (df["bid_volume"] + df["ask_volume"]).resample(self.bar_freq).sum()
        bars["vwap"] = vwap_num / (vwap_den + 1e-10)
        bars = bars[bars["tick_count"] >= 10].copy()
        bars = bars.ffill().bfill()
        bars = bars.reset_index()
        logger.info(f"Aggregated into {len(bars):,} {self.bar_freq} bars")
        return bars

    def save_features(self, bars: pd.DataFrame, year: int) -> str:
        output_path = self.output_dir / f"features_{year}.parquet"
        bars.to_parquet(output_path, compression="snappy", index=False)
        logger.info(f"Saved features to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return str(output_path)

    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared cache: {self.cache_dir}")

    async def process_year(self, year: int):
        logger.info(f"{'='*60}")
        logger.info(f"Processing {self.symbol} for {year}")
        logger.info(f"{'='*60}")
        ticks = await self.download_year(year)
        if ticks is None or len(ticks) == 0:
            logger.warning(f"No data for {year}, skipping")
            return None
        bars = self.aggregate_to_bars(ticks)
        if len(bars) == 0:
            logger.warning(f"No bars generated for {year}")
            return None
        output_path = self.save_features(bars, year)
        del ticks
        del bars
        gc.collect()
        self.clear_cache()
        logger.info(f"Completed {year}")
        return output_path

    async def process_all_years(self):
        saved_files = []
        for year in self.years:
            try:
                result = await self.process_year(year)
                if result:
                    saved_files.append(result)
            except Exception as e:
                logger.error(f"Failed to process {year}: {e}")
                continue
        logger.info(f"Completed {len(saved_files)}/{len(self.years)} years")
        return saved_files


def load_all_features(feature_dir: str = "/tmp", years: Optional[List[int]] = None) -> pd.DataFrame:
    years = years or list(range(2011, 2027))
    feature_dir = Path(feature_dir)
    all_dfs = []
    for year in years:
        path = feature_dir / f"features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df):,} rows from {path}")
        else:
            logger.warning(f"Missing: {path}")
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Total combined: {len(combined):,} rows")
    return combined


async def main():
    logger.info("Starting Dukascopy tick data download...")
    loader = DukascopyTickLoader(
        symbol="EURUSD",
        cache_dir="/tmp/dukascopy_cache",
        output_dir="/tmp",
        bar_freq="1H",
        years=list(range(2011, 2027)),
    )
    saved_files = await loader.process_all_years()
    logger.info(f"Saved {len(saved_files)} feature files")
    for f in saved_files:
        logger.info(f"  {f}")
    return saved_files


if __name__ == "__main__":
    asyncio.run(main())
