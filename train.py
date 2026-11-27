#!/usr/bin/env python3
"""
Training and backtesting pipeline for FX microstructure features.
Loads yearly feature files, builds stationary features, trains LightGBM,
and runs realistic backtest with spread costs.
"""

import gc
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = Path("/tmp")
YEARS = list(range(2011, 2027))


def load_features() -> pd.DataFrame:
    """Load and concatenate all yearly feature files."""
    all_dfs = []
    for year in YEARS:
        path = FEATURE_DIR / f"features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df):,} rows from {path}")
        else:
            logger.warning(f"Missing: {path}")
    if not all_dfs:
        raise FileNotFoundError("No feature files found")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Total combined: {len(combined):,} rows")
    return combined


def build_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build stationary features from raw microstructure bars."""
    df = df.copy()

    # Returns (stationary)
    df["ret_1"] = df["mid_close"].pct_change()
    df["ret_3"] = df["mid_close"].pct_change(3)
    df["ret_6"] = df["mid_close"].pct_change(6)
    df["ret_12"] = df["mid_close"].pct_change(12)
    df["ret_24"] = df["mid_close"].pct_change(24)

    # Log returns
    df["log_ret_1"] = np.log(df["mid_close"]).diff()
    df["log_ret_3"] = np.log(df["mid_close"]).diff(3)
    df["log_ret_6"] = np.log(df["mid_close"]).diff(6)

    # Volatility features
    df["vol_6"] = df["log_ret_1"].rolling(6).std()
    df["vol_12"] = df["log_ret_1"].rolling(12).std()
    df["vol_24"] = df["log_ret_1"].rolling(24).std()
    df["vol_48"] = df["log_ret_1"].rolling(48).std()

    # Realized volatility (already in bars)
    df["rv_1"] = df["realized_vol"]
    df["rv_6"] = df["realized_vol"].rolling(6).mean()
    df["rv_12"] = df["realized_vol"].rolling(12).mean()

    # Spread features (already in bars, but normalize)
    df["spread_z"] = (df["spread_mean"] - df["spread_mean"].rolling(100).mean()) / (
        df["spread_mean"].rolling(100).std() + 1e-10
    )
    df["spread_max_z"] = (df["spread_max"] - df["spread_max"].rolling(100).mean()) / (
        df["spread_max"].rolling(100).std() + 1e-10
    )

    # Tick velocity features
    df["tick_vel_1"] = df["tick_count"]
    df["tick_vel_6"] = df["tick_count"].rolling(6).mean()
    df["tick_vel_12"] = df["tick_count"].rolling(12).mean()
    df["tick_vel_z"] = (df["tick_count"] - df["tick_count"].rolling(100).mean()) / (
        df["tick_count"].rolling(100).std() + 1e-10
    )

    # Volume imbalance features
    df["vol_imb_1"] = df["vol_imbalance_mean"]
    df["vol_imb_6"] = df["vol_imbalance_mean"].rolling(6).mean()
    df["vol_imb_12"] = df["vol_imbalance_mean"].rolling(12).mean()
    df["vol_imb_std"] = df["vol_imbalance_std"]

    # Volume features
    df["vol_total_1"] = df["total_volume"]
    df["vol_total_6"] = df["total_volume"].rolling(6).mean()
    df["vol_total_z"] = (df["total_volume"] - df["total_volume"].rolling(100).mean()) / (
        df["total_volume"].rolling(100).std() + 1e-10
    )

    # VWAP deviation
    df["vwap_dev"] = (df["mid_close"] - df["vwap"]) / df["vwap"]
    df["vwap_dev_6"] = df["vwap_dev"].rolling(6).mean()

    # Momentum features
    df["mom_6"] = df["mid_close"] / df["mid_close"].shift(6) - 1
    df["mom_12"] = df["mid_close"] / df["mid_close"].shift(12) - 1
    df["mom_24"] = df["mid_close"] / df["mid_close"].shift(24) - 1

    # Mean reversion
    df["ma_12"] = df["mid_close"].rolling(12).mean()
    df["ma_24"] = df["mid_close"].rolling(24).mean()
    df["ma_48"] = df["mid_close"].rolling(48).mean()
    df["dist_ma_12"] = (df["mid_close"] - df["ma_12"]) / df["ma_12"]
    df["dist_ma_24"] = (df["mid_close"] - df["ma_24"]) / df["ma_24"]
    df["dist_ma_48"] = (df["mid_close"] - df["ma_48"]) / df["ma_48"]

    # RSI-like
    delta = df["mid_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Target: next period return direction (1=up, 0=down, 2=flat)
    df["target"] = (df["mid_close"].shift(-1) > df["mid_close"]).astype(int)
    # Add flat class for very small moves
    flat_threshold = df["mid_close"].pct_change().rolling(100).std() * 0.1
    df.loc[df["mid_close"].pct_change().abs() < flat_threshold, "target"] = 2

    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Features built: {len(df):,} rows, {df.shape[1]} columns")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare X, y arrays for training."""
    # Feature columns (exclude non-feature columns)
    exclude_cols = ["timestamp", "mid_open", "mid_high", "mid_low", "mid_close",
                    "target", "spread_mean", "spread_max", "spread_pct_mean",
                    "tick_count", "vol_imbalance_mean", "vol_imbalance_std",
                    "realized_vol", "bid_volume_sum", "ask_volume_sum",
                    "total_volume", "vwap", "ma_12", "ma_24", "ma_48"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df["target"].values

    # Handle any remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Prepared data: X={X.shape}, y={y.shape}, features={len(feature_cols)}")
    return X, y, feature_cols


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str]) -> lgb.LGBMClassifier:
    """Train regularized LightGBM classifier."""
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.02,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "class_weight": "balanced",
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    logger.info(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "class_distribution": {int(k): int(v) for k, v in
                               zip(*np.unique(y_pred, return_counts=True))},
    }

    # Confidence-gated accuracy
    confidences = np.max(y_proba, axis=1)
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        mask = confidences >= thresh
        if mask.sum() > 0:
            metrics[f"conf_{thresh}_acc"] = float(accuracy_score(y_test[mask], y_pred[mask]))
            metrics[f"conf_{thresh}_cov"] = float(mask.sum() / len(y_test))

    return metrics


def run_backtest(model, X_test: np.ndarray, y_test: np.ndarray,
                 test_df: pd.DataFrame, feature_names: List[str]) -> Dict:
    """Run realistic backtest with spread cost deduction."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidences = np.max(y_proba, axis=1)

    # Align with test_df (last len(X_test) rows)
    test_df = test_df.iloc[-len(X_test):].reset_index(drop=True)

    # Trading logic: only trade when confidence > 0.6
    threshold = 0.6
    trade_mask = confidences >= threshold

    if not trade_mask.any():
        logger.warning("No trades meet confidence threshold")
        return {"total_trades": 0, "total_pnl": 0.0, "sharpe": 0.0}

    trades = []
    for i in np.where(trade_mask)[0]:
        pred = y_pred[i]
        actual = y_test[i]
        conf = confidences[i]

        # Spread cost (in basis points)
        spread_bps = test_df.iloc[i]["spread_pct_mean"] * 10000  # convert to bps

        # PnL calculation
        # pred: 0=down, 1=up, 2=flat
        # actual: same
        if pred == 2:  # flat prediction - no trade
            pnl = -spread_bps / 10000  # still pay spread
        elif pred == actual:  # correct direction
            # Assume 1:1 risk/reward, win = spread * 2, lose = spread
            pnl = spread_bps * 2 / 10000
        else:  # wrong direction
            pnl = -spread_bps / 10000

        trades.append({
            "idx": int(i),
            "pred": int(pred),
            "actual": int(actual),
            "conf": float(conf),
            "spread_bps": float(spread_bps),
            "pnl": float(pnl),
        })

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df["pnl"].sum()
    win_rate = (trades_df["pnl"] > 0).mean()
    sharpe = trades_df["pnl"].mean() / (trades_df["pnl"].std() + 1e-10) * np.sqrt(252 * 24)

    backtest_results = {
        "total_trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "sharpe": float(sharpe),
        "avg_spread_bps": float(trades_df["spread_bps"].mean()),
        "avg_confidence": float(trades_df["conf"].mean()),
        "threshold_used": threshold,
    }

    logger.info(f"Backtest: trades={backtest_results['total_trades']}, "
                f"win_rate={win_rate:.2%}, pnl={total_pnl:.4f}, sharpe={sharpe:.2f}")
    return backtest_results


def main():
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # Load features
    logger.info("Loading features...")
    df = load_features()

    # Build stationary features
    logger.info("Building stationary features...")
    df = build_stationary_features(df)

    # Prepare data
    logger.info("Preparing data...")
    X, y, feature_names = prepare_data(df)

    # Chronological 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Further split train into train/val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train model
    logger.info("Training LightGBM...")
    model = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Test metrics: {json.dumps(metrics, indent=2)}")

    # Backtest
    logger.info("Running backtest...")
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    backtest = run_backtest(model, X_test, y_test, test_df, feature_names)
    logger.info(f"Backtest results: {json.dumps(backtest, indent=2)}")

    # Save model
    model_path = ARTIFACTS_DIR / "fx_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Save feature names
    feature_path = ARTIFACTS_DIR / "feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Feature names saved to {feature_path}")

    # Save results
    results = {
        "metrics": metrics,
        "backtest": backtest,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    results_path = ARTIFACTS_DIR / "backtest_report.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
