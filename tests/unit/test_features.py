"""Unit tests for feature engineering"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.features import FeatureEngineer

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, 1000)
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price * (1 + np.random.normal(0, 0.0001, 1000)),
        "high": price * (1 + np.abs(np.random.normal(0, 0.0002, 1000))),
        "low": price * (1 - np.abs(np.random.normal(0, 0.0002, 1000))),
        "close": price,
        "volume": np.random.randint(100, 1000, 1000),
        "spread": np.random.uniform(0.00005, 0.0002, 1000)
    })
    return df

def test_create_features(sample_data):
    fe = FeatureEngineer()
    df = fe.create_features(sample_data)
    
    assert len(df) == len(sample_data)
    assert "rsi_14" in df.columns
    assert "macd" in df.columns
    assert "atr_14" in df.columns
    assert "bb_position" in df.columns

def test_create_labels(sample_data):
    fe = FeatureEngineer()
    df = fe.create_features(sample_data)
    df = fe.create_labels(df, horizon=4, label_type="classification", n_classes=5)
    
    assert "label" in df.columns
    assert df["label"].notna().sum() > 0
    assert set(df["label"].dropna().unique()).issubset({0, 1, 2, 3, 4})

def test_feature_engineer_initialization():
    fe = FeatureEngineer(lookback_windows=[10, 20])
    assert fe.lookback_windows == [10, 20]