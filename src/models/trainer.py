"""Model training pipeline with MLflow tracking"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.data.dataset import MLDatasetBuilder
from src.data.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.rf_model import RandomForestModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Training pipeline with hyperparameter optimization"""

    def __init__(self, training_config: Dict, model_configs: Dict):
        self.training_config = training_config
        self.model_configs = model_configs

    def train_ensemble(
        self,
        storage,
        feature_engineer: FeatureEngineer,
        instruments: List[str],
        granularity: str = "H1",
        optimize_hyperparams: bool = False
    ) -> Tuple[EnsembleModel, Dict]:
        """Train the ensemble model"""

        # Build dataset
        dataset_builder = MLDatasetBuilder(
            storage=storage,
            feature_engineer=feature_engineer,
            instruments=instruments,
            granularity=granularity,
            lookback_window=self.training_config.get("lookback_window", 500),
            prediction_horizon=self.training_config.get("prediction_horizon", 4),
            label_type=self.training_config.get("label_type", "classification"),
            n_classes=self.training_config.get("n_classes", 5)
        )

        dataset = dataset_builder.build_dataset(
            test_size=self.training_config["validation"].get("test_size", 0.2),
            val_size=0.1
        )

        X_train, y_train = dataset["X_train"], dataset["y_train"]
        X_val, y_val = dataset["X_val"], dataset["y_val"]

        # Optionally optimize hyperparameters
        if optimize_hyperparams:
            best_params = self._optimize_hyperparams(X_train, y_train, X_val, y_val)
            self.model_configs.update(best_params)

        # Train ensemble
        ensemble = EnsembleModel(self.model_configs)
        metrics = ensemble.fit(X_train, y_train, X_val, y_val)

        # Add test metrics
        test_pred = ensemble.predict(dataset["X_test"])
        test_acc = (test_pred == dataset["y_test"]).mean()
        metrics["test_accuracy"] = float(test_acc)

        logger.info(f"Training complete. Metrics: {metrics}")

        return ensemble, metrics

    def _optimize_hyperparams(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        optuna_config = self.training_config.get("optuna", {})
        n_trials = optuna_config.get("n_trials", 100)

        def objective(trial):
            # XGBoost params
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 0.001, 0.1, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.01, 1.0, log=True),
            }

            # LightGBM params
            lgb_params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 255),
                "learning_rate": trial.suggest_float("lgb_lr", 0.001, 0.1, log=True),
                "feature_fraction": trial.suggest_float("lgb_feat_frac", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("lgb_bag_frac", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("lgb_min_child", 10, 50),
            }

            # Random Forest params
            rf_params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
                "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("rf_min_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_leaf", 1, 10),
            }

            # Quick ensemble evaluation
            ensemble = EnsembleModel({
                "xgboost": xgb_params,
                "lightgbm": lgb_params,
                "random_forest": rf_params,
                "cv_folds": 3
            })
            metrics = ensemble.fit(X_train, y_train, X_val, y_val)
            return metrics.get("val_accuracy", 0)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.value}")

        # Return best params in correct format
        best = study.best_trial.params
        return {
            "xgboost": {k.replace("xgb_", ""): v for k, v in best.items() if k.startswith("xgb_")},
            "lightgbm": {k.replace("lgb_", ""): v for k, v in best.items() if k.startswith("lgb_")},
            "random_forest": {k.replace("rf_", ""): v for k, v in best.items() if k.startswith("rf_")}
        }