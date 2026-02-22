"""
Model Registry & Factory Pattern
Centralized model management for easy switching between models
"""

import os
import json
import joblib
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    model_type: str  # 'lightgbm', 'xgboost', 'rf_sklearn', etc.
    file_path: str
    features: list
    preprocessing_fn: Optional[Callable] = None
    description: str = ""
    created_date: str = ""
    accuracy: Optional[float] = None
    status: str = "active"  # active, deprecated, experimental


class ModelInterface(ABC):
    """Abstract base class for model wrappers"""
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass


class LightGBMWrapper(ModelInterface):
    """LightGBM model wrapper"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        proba = self.model.predict(X)
        # LightGBM returns probabilities directly for binary classification
        return proba if len(proba.shape) > 1 else proba


class XGBoostWrapper(ModelInterface):
    """XGBoost model wrapper"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RandomForestWrapper(ModelInterface):
    """Scikit-Learn RandomForest wrapper"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ModelRegistry:
    """Central registry for all models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model registry
        
        Args:
            models_dir: Directory containing all models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.registry: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, ModelInterface] = {}
        self.current_model: Optional[str] = None
        self.config_file = self.models_dir / "registry.json"
        
        # Load registry from file
        self._load_registry()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_type: str,
        file_path: str,
        features: list,
        preprocessing_fn: Optional[Callable] = None,
        description: str = "",
        accuracy: Optional[float] = None,
        status: str = "active"
    ) -> ModelMetadata:
        """
        Register a new model
        
        Args:
            name: Model name (e.g., 'insurance_lightgbm')
            version: Version (e.g., '1.0.0')
            model_type: Type ('lightgbm', 'xgboost', 'rf_sklearn')
            file_path: Path to model file
            features: List of required features
            preprocessing_fn: Custom preprocessing function
            description: Model description
            accuracy: Model accuracy score
            status: Model status
        
        Returns:
            ModelMetadata object
        """
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            file_path=file_path,
            features=features,
            preprocessing_fn=preprocessing_fn,
            description=description,
            accuracy=accuracy,
            status=status
        )
        
        key = f"{name}:{version}"
        self.registry[key] = metadata
        logger.info(f"Registered model: {key}")
        
        # Save registry
        self._save_registry()
        
        return metadata
    
    def load_model(self, name: str, version: str = "latest") -> ModelInterface:
        """
        Load a model by name and version
        
        Args:
            name: Model name
            version: Version (default: 'latest')
        
        Returns:
            Loaded model wrapper
        """
        # Get version
        if version == "latest":
            version = self._get_latest_version(name)
        
        key = f"{name}:{version}"
        
        # Check if already loaded
        if key in self.loaded_models:
            logger.info(f"Using cached model: {key}")
            return self.loaded_models[key]
        
        # Get metadata
        if key not in self.registry:
            raise ValueError(f"Model not found: {key}")
        
        metadata = self.registry[key]
        
        # Load model file
        model_path = metadata.file_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            raw_model = joblib.load(model_path)
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Wrap model based on type
        wrapper = self._get_wrapper(raw_model, metadata.model_type)
        
        # Cache it
        self.loaded_models[key] = wrapper
        
        logger.info(f"Successfully loaded: {key}")
        return wrapper
    
    def set_current_model(self, name: str, version: str = "latest"):
        """Set the current active model"""
        key = f"{name}:{version}"
        if key not in self.registry:
            raise ValueError(f"Model not found: {key}")
        
        self.current_model = key
        logger.info(f"Set current model to: {key}")
    
    def get_current_model(self) -> ModelInterface:
        """Get the current active model"""
        if not self.current_model:
            raise ValueError("No current model set")
        
        name, version = self.current_model.split(":")
        return self.load_model(name, version)
    
    def list_models(self, status: str = None) -> list:
        """
        List all registered models
        
        Args:
            status: Filter by status (optional)
        
        Returns:
            List of model metadata
        """
        models = []
        for key, metadata in self.registry.items():
            if status and metadata.status != status:
                continue
            models.append({
                "key": key,
                "name": metadata.name,
                "version": metadata.version,
                "type": metadata.model_type,
                "status": metadata.status,
                "accuracy": metadata.accuracy,
                "features_count": len(metadata.features),
                "description": metadata.description
            })
        return models
    
    def delete_model(self, name: str, version: str):
        """Delete a model from registry"""
        key = f"{name}:{version}"
        if key in self.registry:
            del self.registry[key]
            if key in self.loaded_models:
                del self.loaded_models[key]
            self._save_registry()
            logger.info(f"Deleted model: {key}")
    
    def get_model_features(self, name: str, version: str = "latest") -> list:
        """Get required features for a model"""
        if version == "latest":
            version = self._get_latest_version(name)
        
        key = f"{name}:{version}"
        if key not in self.registry:
            raise ValueError(f"Model not found: {key}")
        
        return self.registry[key].features
    
    def _get_wrapper(self, model, model_type: str) -> ModelInterface:
        """Get appropriate wrapper for model type"""
        if model_type == "lightgbm":
            return LightGBMWrapper(model)
        elif model_type == "xgboost":
            return XGBoostWrapper(model)
        elif model_type in ["rf_sklearn", "random_forest"]:
            return RandomForestWrapper(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_latest_version(self, name: str) -> str:
        """Get latest version of a model"""
        versions = [
            v.split(":")[1]
            for k, v in [(k, k.split(":")[1]) for k, m in self.registry.items()]
            if k.startswith(f"{name}:")
        ]
        if not versions:
            raise ValueError(f"No versions found for model: {name}")
        return sorted(versions)[-1]
    
    def _save_registry(self):
        """Save registry to JSON file"""
        data = {}
        for key, metadata in self.registry.items():
            data[key] = {
                "name": metadata.name,
                "version": metadata.version,
                "model_type": metadata.model_type,
                "file_path": metadata.file_path,
                "features": metadata.features,
                "description": metadata.description,
                "created_date": metadata.created_date,
                "accuracy": metadata.accuracy,
                "status": metadata.status
            }
        
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved registry to: {self.config_file}")
    
    def _load_registry(self):
        """Load registry from JSON file"""
        if not self.config_file.exists():
            logger.info("No existing registry found")
            return
        
        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
            
            for key, item in data.items():
                metadata = ModelMetadata(
                    name=item["name"],
                    version=item["version"],
                    model_type=item["model_type"],
                    file_path=item["file_path"],
                    features=item["features"],
                    description=item.get("description", ""),
                    created_date=item.get("created_date", ""),
                    accuracy=item.get("accuracy"),
                    status=item.get("status", "active")
                )
                self.registry[key] = metadata
            
            logger.info(f"Loaded {len(self.registry)} models from registry")
        except Exception as e:
            logger.error(f"Failed to load registry: {str(e)}")


# Global registry instance
_registry = None


def get_registry(models_dir: str = "models") -> ModelRegistry:
    """Get or create the global registry"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
    return _registry
