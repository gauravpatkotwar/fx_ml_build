"""Ensemble model combining multiple base models"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.models.base import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.rf_model import RandomForestModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Stacking ensemble of XGBoost, LightGBM, and Random Forest"""

    def __init__(self, params: Dict = None):
        default_params = {
            "method": "stacking",
            "meta_learner": "logistic",
            "cv_folds": 5,
            "weights": [0.4, 0.35, 0.25]
        }
        if params:
            default_params.update(params)
        super().__init__("ensemble", default_params)
        self.base_models = []
        self.meta_model = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        self.base_models = [
            XGBoostModel(self.params.get("xgboost", {})),
            LightGBMModel(self.params.get("lightgbm", {})),
            RandomForestModel(self.params.get("random_forest", {}))
        ]

        n_samples = X.shape[0]
        n_models = len(self.base_models)
        n_classes = len(np.unique(y))

        meta_features = np.zeros((n_samples, n_models * n_classes))

        # Manual cross-validation for stacking
        cv_folds = self.params.get("cv_folds", 5)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{n_models}: {model.name}")
            
            # Create out-of-fold predictions
            oof_preds = np.zeros((n_samples, n_classes))
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Clone and fit model on fold
                fold_model = type(model)(model.params)
                fold_model.fit(X_train_fold, y_train_fold)
                fold_probs = fold_model.predict_proba(X_val_fold)
                
                # Ensure 2D output
                if fold_probs.ndim == 1:
                    # Handle single class case
                    fold_probs = np.full((len(X_val_fold), n_classes), 1.0 / n_classes)
                    pred_classes = fold_model.predict(X_val_fold)
                    for j, c in enumerate(pred_classes):
                        if c < n_classes:
                            fold_probs[j, c] = 1.0
                
                oof_preds[val_idx] = fold_probs
            
            meta_features[:, i*n_classes:(i+1)*n_classes] = oof_preds
            
            # Fit final model on full data
            model.fit(X, y)

        if self.params.get("meta_learner") == "logistic":
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        self.meta_model.fit(meta_features, y)
        
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

    def _get_probs_2d(self, model, X: np.ndarray, n_classes: int) -> np.ndarray:
        """Get 2D probabilities from model, handling edge cases."""
        probs = model.predict_proba(X)
        if probs.ndim == 1:
            # Handle single class case
            probs = np.full((len(X), n_classes), 1.0 / n_classes)
            pred_classes = model.predict(X)
            for j, c in enumerate(pred_classes):
                if c < n_classes:
                    probs[j, c] = 1.0
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        n_classes = len(self.meta_model.classes_)
        meta_features = np.zeros((n_samples, n_models * n_classes))
        for i, model in enumerate(self.base_models):
            probs = self._get_probs_2d(model, X, n_classes)
            meta_features[:, i*n_classes:(i+1)*n_classes] = probs
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        n_classes = len(self.meta_model.classes_)
        meta_features = np.zeros((n_samples, n_models * n_classes))
        for i, model in enumerate(self.base_models):
            probs = self._get_probs_2d(model, X, n_classes)
            meta_features[:, i*n_classes:(i+1)*n_classes] = probs
        return self.meta_model.predict_proba(meta_features)

    def predict_with_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.predict(X)
        probs = self.predict_proba(X)
        return preds, probs

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_types = []
        for i, model in enumerate(self.base_models):
            model.save(path / f"base_model_{i}.pkl")
            model_types.append(model.__class__.__name__)
        joblib.dump(self.meta_model, path / "meta_model.pkl")
        joblib.dump({
            "name": self.name, "params": self.params,
            "feature_names": self.base_models[0].feature_names if self.base_models else None,
            "is_fitted": self.is_fitted, "n_base_models": len(self.base_models),
            "model_types": model_types
        }, path / "ensemble_config.pkl")
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        config = joblib.load(path / "ensemble_config.pkl")
        instance = cls(config["params"])
        instance.is_fitted = config["is_fitted"]
        instance.feature_names = config["feature_names"]
        instance.base_models = []
        
        # Map model type names to classes
        model_class_map = {
            "XGBoostModel": XGBoostModel,
            "LightGBMModel": LightGBMModel,
            "RandomForestModel": RandomForestModel,
        }
        
        model_types = config.get("model_types", ["XGBoostModel", "LightGBMModel", "RandomForestModel"])
        for i, model_type in enumerate(model_types):
            model_class = model_class_map.get(model_type, XGBoostModel)
            model = model_class.load(path / f"base_model_{i}.pkl")
            instance.base_models.append(model)
        instance.meta_model = joblib.load(path / "meta_model.pkl")
        return instance

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        if not self.is_fitted or not self.base_models:
            return []
        all_importance = {}
        for model in self.base_models:
            for name, score in model.get_feature_importance(top_n * 2):
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(score)
        avg_importance = {name: np.mean(scores) for name, scores in all_importance.items()}
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
        for model in self.base_models:
            model.fit(X, y)

        self.is_fitted = True

        train_pred = self.predict(X)
        train_acc = (train_pred == y).mean()
        metrics = {"train_accuracy": float(train_acc)}

        if X_val is not None:
            val_pred = self.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            metrics["val_accuracy"] = float(val_acc)

        return metrics