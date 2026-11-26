"""LightGBM model implementation"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import lightgbm as lgb

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier for FX prediction"""

    def __init__(self, params: Dict = None):
        default_params = {
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "objective": "multiclass",
            "metric": "multi_logloss"
        }
        if params:
            default_params.update(params)
        super().__init__("lightgbm", default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train LightGBM model"""
        self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        n_classes = len(np.unique(y))
        self.params["num_class"] = n_classes
        self.n_classes_ = n_classes  # Store for predict_proba

        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get("n_estimators", 500),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        self.is_fitted = True

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
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        # Ensure 2D output for multi-class
        if preds.ndim == 1:
            # Binary case or single class - convert to 2D
            # This can happen if model was trained on single class
            n_classes = getattr(self, 'n_classes_', 2)
            if n_classes == 2:
                preds = np.column_stack([1 - preds, preds])
            else:
                # Multi-class with single class seen - create uniform distribution
                preds_1d = preds
                preds = np.full((len(X), n_classes), 1.0 / n_classes)
                # Put all probability on the predicted class
                pred_classes = np.round(preds_1d * (n_classes - 1)).astype(int)
                pred_classes = np.clip(pred_classes, 0, n_classes - 1)
                for i, c in enumerate(pred_classes):
                    preds[i, c] = 1.0
        return preds

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance"""
        if not self.is_fitted:
            return []

        importance = self.model.feature_importance(importance_type="gain")
        indices = np.argsort(importance)[::-1][:top_n]

        result = []
        for idx in indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            result.append((name, float(importance[idx])))

        return result