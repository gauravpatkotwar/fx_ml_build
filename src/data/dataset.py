"""ML Dataset builder from processed FX data"""
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.features import FeatureEngineer
from src.data.storage import DataStorage

logger = logging.getLogger(__name__)


class MLDatasetBuilder:
    """Build ML-ready datasets from processed FX data"""

    def __init__(
        self,
        storage: DataStorage,
        feature_engineer: FeatureEngineer,
        instruments: List[str],
        granularity: str = "H1",
        lookback_window: int = 500,
        prediction_horizon: int = 4,
        label_type: str = "classification",
        n_classes: int = 5
    ):
        self.storage = storage
        self.feature_engineer = feature_engineer
        self.instruments = instruments
        self.granularity = granularity
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.label_type = label_type
        self.n_classes = n_classes

    def build_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """Build complete train/val/test dataset"""
        all_features = []
        all_labels = []

        for instrument in self.instruments:
            logger.info(f"Processing {instrument}...")
            df = self.storage.read(instrument, self.granularity, start_date, end_date)

            if df.empty:
                logger.warning(f"No data for {instrument}")
                continue

            # Create features
            df = self.feature_engineer.create_features(df)

            # Create labels
            df = self.feature_engineer.create_labels(
                df,
                horizon=self.prediction_horizon,
                label_type=self.label_type,
                n_classes=self.n_classes
            )

            # Drop NaN
            feature_cols = [c for c in df.columns if c not in
                           ["timestamp", "instrument", "granularity", "label",
                            "future_return", "future_log_return",
                            "label_q10", "label_q50", "label_q90"]]

            df_clean = df.dropna(subset=feature_cols + ["label"]).copy()

            if len(df_clean) == 0:
                logger.warning(f"No clean data for {instrument}")
                continue

            X = df_clean[feature_cols].values
            y = df_clean["label"].values

            all_features.append(X)
            all_labels.append(y)

        if not all_features:
            raise ValueError("No valid data for any instrument")

        X_all = np.vstack(all_features)
        y_all = np.concatenate(all_labels)

        # Time-series split (no shuffle)
        n = len(X_all)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        X_train, X_val, X_test = X_all[:val_idx], X_all[val_idx:test_idx], X_all[test_idx:]
        y_train, y_val, y_test = y_all[:val_idx], y_all[val_idx:test_idx], y_all[test_idx:]

        logger.info(f"Dataset shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_names": feature_cols,
            "n_classes": self.n_classes if self.label_type == "classification" else 1
        }

    def get_latest_features(self, instrument: str, n_bars: int = None) -> pd.DataFrame:
        """Get latest features for prediction"""
        n_bars = n_bars or self.lookback_window + 100
        df = self.storage.read(instrument, self.granularity)

        if df.empty:
            return pd.DataFrame()

        df = df.tail(n_bars).copy()
        df = self.feature_engineer.create_features(df)

        feature_cols = [c for c in df.columns if c not in
                       ["timestamp", "instrument", "granularity", "label",
                        "future_return", "future_log_return",
                        "label_q10", "label_q50", "label_q90"]]

        latest = df[feature_cols].iloc[[-1]].fillna(method="ffill").fillna(0)
        return latest