"""Parquet partitioned storage for FX data"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles reading/writing partitioned Parquet data"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, instrument: str, granularity: str) -> Path:
        """Get partitioned path: instrument/granularity/year/month/"""
        return self.base_path / instrument / granularity

    def write(self, df: pd.DataFrame, instrument: str, granularity: str):
        """Write dataframe to partitioned Parquet"""
        path = self._get_path(instrument, granularity)
        path.mkdir(parents=True, exist_ok=True)

        # Partition by year/month
        df = df.copy()
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # Write partitioned
        df.to_parquet(
            path,
            partition_cols=["year", "month"],
            index=False,
            compression="snappy"
        )
        logger.info(f"Written {len(df)} rows to {path}")

    def read(self, instrument: str, granularity: str,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> pd.DataFrame:
        """Read dataframe from partitioned Parquet"""
        path = self._get_path(instrument, granularity)

        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)

        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        return df.sort_values("timestamp").reset_index(drop=True)

    def exists(self, instrument: str, granularity: str) -> bool:
        """Check if data exists"""
        path = self._get_path(instrument, granularity)
        return path.exists()

    def list_available(self) -> dict:
        """List all available data"""
        result = {}
        for inst_dir in self.base_path.iterdir():
            if not inst_dir.is_dir():
                continue
            instrument = inst_dir.name
            result[instrument] = {}
            for gran_dir in inst_dir.iterdir():
                if not gran_dir.is_dir():
                    continue
                granularity = gran_dir.name
                # Count files
                parquet_files = list(gran_dir.rglob("*.parquet"))
                result[instrument][granularity] = len(parquet_files)
        return result

    def delete(self, instrument: str, granularity: str):
        """Delete data for instrument/granularity"""
        import shutil
        path = self._get_path(instrument, granularity)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted {path}")