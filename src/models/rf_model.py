"""Random Forest model implementation"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for FX prediction"""

    def __init__(self, params: Dict = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced_subsample"
        }
        if params:
            default_params.update(params)
        super().__init__("random_forest", default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train Random Forest model"""
        self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

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
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance"""
        if not self.is_fitted:
            return []

        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]

        result = []
        for idx in indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            result.append((name, float(importance[idx])))

        return result