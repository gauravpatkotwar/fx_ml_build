"""Base model interface"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class BaseModel(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all models"""

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def __sklearn_tags__(self):
        from sklearn.utils import Tags
        return Tags(
            estimator_type="classifier",
            requires_y=True,
            requires_fit=True,
        )

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)"""
        raise NotImplementedError

    def predict_with_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        preds = self.predict(X)
        try:
            probs = self.predict_proba(X)
        except:
            probs = np.ones((len(X), 1))
        return preds, probs

    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk"""
        path = Path(path)
        data = joblib.load(path)
        # Handle both old format (name, params) and new format (params only)
        if "name" in data and "params" in data:
            instance = cls(data["params"])
        else:
            instance = cls(data.get("params", {}))
        instance.model = data["model"]
        instance.feature_names = data.get("feature_names")
        instance.is_fitted = data.get("is_fitted", True)
        return instance

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance"""
        if not self.is_fitted or self.feature_names is None:
            return []
        return []