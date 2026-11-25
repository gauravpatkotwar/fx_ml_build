"""Technical and fundamental feature engineering for FX"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats
from scipy.signal import argrelextrema
import ta

class FeatureEngineer:
    """Generate 100+ features for FX prediction"""

    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [10, 20, 50, 100, 200, 500]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features from OHLCV data"""
        df = df.copy().sort_values("timestamp")

        required = ["open", "high", "low", "close", "volume", "spread"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = self._add_price_features(df)
        df = self._add_return_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_statistical_features(df)
        df = self._add_pattern_features(df)
        df = self._add_microstructure_features(df)
        df = self._add_time_features(df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["log_return_abs"] = df["log_return"].abs()
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["body_ratio"] = df["body_size"] / (df["high"] - df["low"] + 1e-10)
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)
        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.lookback_windows:
            df[f"return_{window}"] = df["close"].pct_change(window)
            df[f"log_return_{window}"] = np.log(df["close"] / df["close"].shift(window))
            df[f"vol_{window}"] = df["log_return"].rolling(window).std() * np.sqrt(252 * 24)
            df[f"vol_parksinson_{window}"] = self._parkinson_vol(df, window)
            df[f"vol_garman_klass_{window}"] = self._garman_klass_vol(df, window)
            df[f"skew_{window}"] = df["log_return"].rolling(window).skew()
            df[f"kurt_{window}"] = df["log_return"].rolling(window).kurt()
            df[f"sharpe_{window}"] = df[f"return_{window}"] / (df[f"vol_{window}"] + 1e-10)
        return df

    def _parkinson_vol(self, df: pd.DataFrame, window: int) -> pd.Series:
        rs = np.log(df["high"] / df["low"]) ** 2
def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]; high = df["high"]; low = df["low"]; volume = df["volume"]
        df["sma_20"] = ta.trend.sma_indicator(close, window=20)
        df["sma_50"] = ta.trend.sma_indicator(close, window=50)
        df["sma_200"] = ta.trend.sma_indicator(close, window=200)
        df["ema_20"] = ta.trend.ema_indicator(close, window=20)
        df["ema_50"] = ta.trend.ema_indicator(close, window=50)
        df["price_sma20_ratio"] = close / df["sma_20"]
        df["price_sma50_ratio"] = close / df["sma_50"]
        df["sma20_sma50_ratio"] = df["sma_20"] / df["sma_50"]
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd(); df["macd_signal"] = macd.macd_signal(); df["macd_hist"] = macd.macd_diff()
        df["adx"] = ta.trend.adx(high, low, close, window=14)
        df["adx_pos"] = ta.trend.adx_pos(high, low, close, window=14)
        df["adx_neg"] = ta.trend.adx_neg(high, low, close, window=14)
        aroon = ta.trend.AroonIndicator(close, window=25)
        df["aroon_up"] = aroon.aroon_up(); df["aroon_down"] = aroon.aroon_down()
        df["aroon_indicator"] = aroon.aroon_indicator()
        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["rsi_21"] = ta.momentum.rsi(close, window=21)
        df["stoch_k"] = ta.momentum.stoch(high, low, close, window=14)
        df["stoch_d"] = ta.momentum.stoch_signal(high, low, close, window=14)
        df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
        df["roc_10"] = ta.momentum.roc(close, window=10); df["roc_20"] = ta.momentum.roc(close, window=20)
        df["bb_upper"] = ta.volatility.bollinger_hband(close, window=20)
        df["bb_lower"] = ta.volatility.bollinger_lband(close, window=20)
        df["bb_mid"] = ta.volatility.bollinger_mavg(close, window=20)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["atr_21"] = ta.volatility.average_true_range(high, low, close, window=21)
        df["natr"] = ta.volatility.normalized_average_true_range(high, low, close, window=14)
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
        df["kc_upper"] = kc.keltner_channel_hband(); df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_mid"] = kc.keltner_channel_mband(); df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / df["kc_mid"]
        df["donchian_upper"] = high.rolling(20).max(); df["donchian_lower"] = low.rolling(20).min()
        df["donchian_mid"] = (df["donchian_upper"] + df["donchian_lower"]) / 2
        df["donchian_position"] = (close - df["donchian_lower"]) / (df["donchian_upper"] - df["donchian_lower"])
        df["volume_sma_20"] = volume.rolling(20).mean(); df["volume_ratio"] = volume / df["volume_sma_20"]
        df["obv"] = ta.volume.on_balance_volume(close, volume)
def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]; returns = df["log_return"]
        for window in self.lookback_windows:
            df[f"trend_slope_{window}"] = self._rolling_slope(close, window)
            df[f"trend_r2_{window}"] = self._rolling_r2(close, window)
            df[f"hurst_{window}"] = returns.rolling(window).apply(self._hurst_exponent, raw=True)
            df[f"entropy_{window}"] = returns.rolling(window).apply(self._sample_entropy, raw=True)
            df[f"fractal_dim_{window}"] = returns.rolling(window).apply(self._higuchi_fd, raw=True)
            df[f"zscore_{window}"] = (close - close.rolling(window).mean()) / (close.rolling(window).std() + 1e-10)
            df[f"quantile_25_{window}"] = close.rolling(window).quantile(0.25)
            df[f"quantile_75_{window}"] = close.rolling(window).quantile(0.75)
            df[f"iqr_{window}"] = df[f"quantile_75_{window}"] - df[f"quantile_25_{window}"]
            df[f"price_iqr_position_{window}"] = (close - df[f"quantile_25_{window}"]) / (df[f"iqr_{window}"] + 1e-10)
        return df

    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        def slope(x):
            if len(x) < 2: return np.nan
            x_vals = np.arange(len(x))
            return np.polyfit(x_vals, x, 1)[0]
        return series.rolling(window).apply(slope, raw=True)

    def _rolling_r2(self, series: pd.Series, window: int) -> pd.Series:
        def r2(x):
            if len(x) < 3: return np.nan
            x_vals = np.arange(len(x))
            coeffs = np.polyfit(x_vals, x, 1)
            y_pred = np.polyval(coeffs, x_vals)
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-10)
        return series.rolling(window).apply(r2, raw=True)

    def _hurst_exponent(self, x: np.ndarray) -> float:
        if len(x) < 20: return np.nan
        lags = range(2, min(20, len(x)))
        tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
        if len(tau) < 2 or any(t == 0 for t in tau): return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2

    def _sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        if len(x) < 10: return np.nan
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        N = len(x)
        def _phi(m):
            patterns = np.array([x[i:i+m] for i in range(N-m+1)])
            C = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= r, axis=1) - 1
            return np.mean(C / (N - m))
        try: return -np.log(_phi(m+1) / _phi(m))
        except: return np.nan

    def _higuchi_fd(self, x: np.ndarray, k_max: int = 10) -> float:
        if len(x) < k_max * 2: return np.nan
        N = len(x); L = []
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2: continue
                Lmk = np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / (len(idx) * k)
                Lk += Lmk
            L.append(Lk / k)
        if len(L) < 2: return np.nan
        try: return -np.polyfit(np.log(range(1, k_max+1)), np.log(L), 1)[0]
        except: return np.nan
        df["cmf"] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
        ichimoku = ta.trend.IchimokuIndicator(high, low)
        df["ichimoku_a"] = ichimoku.ichimoku_a(); df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line(); df["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        open_ = df["open"]; high = df["high"]; low = df["low"]; close = df["close"]
        df["doji"] = (abs(close - open_) / (high - low + 1e-10) < 0.1).astype(int)
        df["hammer"] = ((high - low) > 3 * abs(close - open_)) & \
                       ((close - low) / (high - low + 1e-10) > 0.6) & \
                       ((open_ - low) / (high - low + 1e-10) > 0.6)
        df["hammer"] = df["hammer"].astype(int)
        df["shooting_star"] = ((high - low) > 3 * abs(close - open_)) & \
                              ((high - close) / (high - low + 1e-10) > 0.6) & \
                              ((high - open_) / (high - low + 1e-10) > 0.6)
        df["shooting_star"] = df["shooting_star"].astype(int)
        df["bullish_engulfing"] = (close > open_) & (close.shift(1) < open_.shift(1)) & \
                                   (close > open_.shift(1)) & (open_ < close.shift(1))
        df["bearish_engulfing"] = (close < open_) & (close.shift(1) > open_.shift(1)) & \
                                   (close < open_.shift(1)) & (open_ > close.shift(1))
        for window in [20, 50, 100]:
            df[f"resistance_{window}"] = high.rolling(window).max()
            df[f"support_{window}"] = low.rolling(window).min()
            df[f"dist_resistance_{window}"] = (df[f"resistance_{window}"] - close) / close
            df[f"dist_support_{window}"] = (close - df[f"support_{window}"]) / close
        df["pivot"] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - low.shift(1)
        df["s1"] = 2 * df["pivot"] - high.shift(1)
        df["r2"] = df["pivot"] + (high.shift(1) - low.shift(1))
        df["s2"] = df["pivot"] - (high.shift(1) - low.shift(1))
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["spread_pct"] = df["spread"] / df["close"]
        df["spread_ma_20"] = df["spread_pct"].rolling(20).mean()
        df["spread_zscore"] = (df["spread_pct"] - df["spread_ma_20"]) / (df["spread_pct"].rolling(20).std() + 1e-10)
        df["bid_ask_imbalance"] = (df["bid_close"] - df["ask_close"]) / (df["bid_close"] + df["ask_close"])
        df["effective_spread"] = 2 * abs(df["close"] - (df["bid_close"] + df["ask_close"]) / 2)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]
        df["hour"] = ts.dt.hour; df["day_of_week"] = ts.dt.dayofweek; df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month; df["quarter"] = ts.dt.quarter
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24); df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7); df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12); df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["session_asian"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["session_london"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["session_ny"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
        df["session_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        return df

    def create_labels(self, df: pd.DataFrame, horizon: int = 4, label_type: str = "classification", n_classes: int = 5) -> pd.DataFrame:
        df = df.copy()
        df["future_return"] = df["close"].pct_change(horizon).shift(-horizon)
        df["future_log_return"] = np.log(df["close"].shift(-horizon) / df["close"])
        if label_type == "classification":
            if n_classes == 3:
                bins = [-np.inf, -0.002, 0.002, np.inf]; labels = [0, 1, 2]
            elif n_classes == 5:
                bins = [-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf]; labels = [0, 1, 2, 3, 4]
            else:
                quantiles = np.linspace(0, 1, n_classes + 1)
                bins = [-np.inf] + df["future_return"].quantile(quantiles[1:-1]).tolist() + [np.inf]
                labels = list(range(n_classes))
            df["label"] = pd.cut(df["future_return"], bins=bins, labels=labels).astype(float)
        elif label_type == "regression":
            df["label"] = df["future_log_return"]
        elif label_type == "quantile":
            for q in [0.1, 0.5, 0.9]:
                df[f"label_q{int(q*100)}"] = df["future_log_return"]
        return df
        return df
        return np.sqrt(rs.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252 * 24)

    def _garman_klass_vol(self, df: pd.DataFrame, window: int) -> pd.Series:
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(rs.rolling(window).mean()) * np.sqrt(252 * 24)