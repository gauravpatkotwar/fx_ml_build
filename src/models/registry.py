"""Model registry for versioning and promotion"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import joblib
import json
from datetime import datetime

from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model versions and promotions"""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.registry_path / "versions.json"
        self._load_versions()

    def _load_versions(self):
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                self.versions = json.load(f)
        else:
            self.versions = {"models": [], "current": None}

    def _save_versions(self):
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def register_model(
        self,
        model: EnsembleModel,
        metrics: Dict,
        run_id: str,
        tags: Dict = None
    ) -> str:
        """Register a new model version"""
        version = f"v{len(self.versions['models']) + 1:04d}"
        timestamp = datetime.utcnow().isoformat()

        version_dir = self.registry_path / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save(version_dir / "model")

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "run_id": run_id,
            "metrics": metrics,
            "tags": tags or {},
            "status": "staging"
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update versions
        self.versions["models"].append(metadata)
        self._save_versions()

        logger.info(f"Model registered as {version}")
        return version

    def promote_model(self, version: str, target: str = "production"):
        """Promote model to production/staging"""
        for m in self.versions["models"]:
            if m["version"] == version:
                m["status"] = target
                if target == "production":
                    self.versions["current"] = version
                break
        self._save_versions()
        logger.info(f"Model {version} promoted to {target}")

    def get_model(self, version: str = None) -> EnsembleModel:
        """Load a model version"""
        if version is None:
            version = self.versions.get("current")
            if version is None:
                raise ValueError("No current model set")

        version_dir = self.registry_path / version / "model"
        return EnsembleModel.load(version_dir)

    def get_latest_model(self) -> EnsembleModel:
        """Get the latest registered model"""
        if not self.versions["models"]:
            raise ValueError("No models registered")
        latest = self.versions["models"][-1]["version"]
        return self.get_model(latest)

    def get_production_model(self) -> EnsembleModel:
        """Get the current production model"""
        current = self.versions.get("current")
        if current is None:
            raise ValueError("No production model set")
        return self.get_model(current)

    def list_versions(self) -> List[Dict]:
        """List all model versions"""
        return self.versions["models"]

    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        v1 = next((m for m in self.versions["models"] if m["version"] == version1), None)
        v2 = next((m for m in self.versions["models"] if m["version"] == version2), None)

        if not v1 or not v2:
            raise ValueError("Version not found")

        comparison = {}
        for key in ["train_accuracy", "val_accuracy", "test_accuracy"]:
            if key in v1["metrics"] and key in v2["metrics"]:
                comparison[key] = {
                    version1: v1["metrics"][key],
                    version2: v2["metrics"][key],
                    "diff": v2["metrics"][key] - v1["metrics"][key]
                }

        return comparison