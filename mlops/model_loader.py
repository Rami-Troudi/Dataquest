"""
Model Loader for MLOps
Loads existing model.pkl and manages model switching
"""

import joblib
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage models from pkl files"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.registry_file = self.models_dir / "registry.json"
        self.current_model_key: Optional[str] = None
    
    def load_model(self, model_path: str, model_name: str = "default", version: str = "1.0.0"):
        """
        Load a model from pkl file
        
        Args:
            model_path: Path to .pkl or .joblib file
            model_name: Name for registry
            version: Version string
        """
        key = f"{model_name}:{version}"
        
        logger.info(f"Loading model from {model_path}...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            bundle = joblib.load(model_path)
            self.loaded_models[key] = {
                'bundle': bundle,
                'path': model_path,
                'name': model_name,
                'version': version,
                'loaded_at': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"✓ Model loaded: {key}")
            logger.info(f"  Model type: {bundle.get('model_type', 'unknown')}")
            logger.info(f"  File: {model_path}")
            
            return self.loaded_models[key]
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model(self, model_key: str):
        """Get loaded model by key"""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model not loaded: {model_key}")
        
        return self.loaded_models[model_key]
    
    def set_current_model(self, model_key: str):
        """Set current active model"""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model not loaded: {model_key}")
        
        self.current_model_key = model_key
        logger.info(f"Current model set to: {model_key}")
    
    def get_current_model(self):
        """Get currently active model"""
        if self.current_model_key is None:
            raise RuntimeError("No model set as current")
        
        return self.get_model(self.current_model_key)
    
    def list_loaded_models(self):
        """List all loaded models"""
        return list(self.loaded_models.keys())


class RFPreprocessor:
    """Preprocessor for RandomForest models from v4_rf_only.py"""
    
    def __init__(self, preprocessor_dict: Dict):
        """
        Initialize with preprocessor from model bundle
        
        Args:
            preprocessor_dict: Dict from bundle['preprocessor']
        """
        self.preprocessor = preprocessor_dict
        self.feature_columns = preprocessor_dict.get('feature_columns', [])
        self.categorical_columns = preprocessor_dict.get('categorical_columns', [])
        self.cat_mappings = preprocessor_dict.get('cat_mappings', {})
        self.numeric_medians = preprocessor_dict.get('numeric_medians', {})
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using stored preprocessor
        Mirrors v4_rf_only.transform_with_preprocessor
        """
        df = df.copy()
        
        # Drop ID and target columns if present
        if 'User_ID' in df.columns:
            df = df.drop(['User_ID'], axis=1)
        if 'Purchased_Coverage_Bundle' in df.columns:
            df = df.drop(['Purchased_Coverage_Bundle'], axis=1)
        
        # Handle categorical features
        for col in self.categorical_columns:
            if col in df.columns:
                values = df[col].astype(str).fillna('__MISSING__')
                mapping = self.cat_mappings.get(col, {})
                df[col] = values.map(mapping).fillna(-1).astype(int)
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Select only required features in correct order
        df = df[self.feature_columns]
        
        # Fill missing values with medians
        for col, median in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        
        return df


class UnifiedModelServer:
    """
    Unified model server for predictions
    Handles loading, preprocessing, and easy model switching
    """
    
    def __init__(self, models_dir: str = "models"):
        self.loader = ModelLoader(models_dir)
        self.current_model_key: Optional[str] = None
        self.current_bundle: Optional[Dict] = None
        self.current_preprocessor: Optional[RFPreprocessor] = None
    
    def load_model(self, model_path: str, model_name: str = "default", version: str = "1.0.0"):
        """Load a model and make it active"""
        
        model_info = self.loader.load_model(model_path, model_name, version)
        key = f"{model_name}:{version}"
        
        self.current_model_key = key
        self.current_bundle = model_info['bundle']
        
        # Initialize preprocessor from bundle
        if 'preprocessor' in self.current_bundle:
            self.current_preprocessor = RFPreprocessor(self.current_bundle['preprocessor'])
        
        return model_info
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        if self.current_preprocessor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        return self.current_preprocessor.transform(df)
    
    def predict(self, df: pd.DataFrame):
        """
        Make predictions on input data
        
        Args:
            df: Input dataframe in test.csv format
            
        Returns:
            Predictions array
        """
        if self.current_bundle is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Preprocess
        X = self.preprocess(df)
        
        # Get model
        model = self.current_bundle.get('rf_model')
        if model is None:
            raise RuntimeError("RandomForest model not found in bundle")
        
        return model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame):
        """Get prediction probabilities"""
        if self.current_bundle is None:
            raise RuntimeError("No model loaded")
        
        X = self.preprocess(df)
        model = self.current_bundle.get('rf_model')
        
        return model.predict_proba(X)
    
    def predict_with_confidence(self, df: pd.DataFrame):
        """Get predictions with confidence scores"""
        if self.current_bundle is None:
            raise RuntimeError("No model loaded")
        
        X = self.preprocess(df)
        model = self.current_bundle.get('rf_model')
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get info about current model"""
        if self.current_bundle is None:
            return {}
        
        return {
            'model_key': self.current_model_key,
            'model_type': self.current_bundle.get('model_type', 'unknown'),
            'features_count': len(self.current_preprocessor.feature_columns) if self.current_preprocessor else 0,
            'class_order': self.current_bundle.get('class_order', []),
            'tuning': self.current_bundle.get('tuning', {})
        }
    
    def get_required_features(self):
        """Get list of required input features"""
        if self.current_preprocessor is None:
            raise RuntimeError("No model loaded")
        
        return self.current_preprocessor.feature_columns


# Singleton instance
_server_instance = None


def get_model_server(models_dir: str = "models") -> UnifiedModelServer:
    """Get or create model server singleton"""
    global _server_instance
    
    if _server_instance is None:
        _server_instance = UnifiedModelServer(models_dir)
    
    return _server_instance
