"""XGBoost model implementation"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import xgboost as xgb

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier for FX prediction"""

    def __init__(self, params: Dict = None):
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "mlogloss"
        }
        if params:
            default_params.update(params)
        super().__init__("xgboost", default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train XGBoost model"""
        self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 500),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        self.is_fitted = True

        # Return training metrics
        train_pred = self.predict(X)
        train_acc = (train_pred == y).mean()

        metrics = {"train_accuracy": float(train_acc)}

        if X_val is not None:
            val_pred = self.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            metrics["val_accuracy"] = float(val_acc)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance"""
        if not self.is_fitted:
            return []

        importance = self.model.get_score(importance_type="gain")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Map to feature names
        result = []
        for feat, score in sorted_features[:top_n]:
            idx = int(feat[1:])  # f0, f1, etc.
            name = self.feature_names[idx] if idx < len(self.feature_names) else feat
            result.append((name, float(score)))

        return result