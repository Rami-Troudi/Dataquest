"""
Feature Manager
Centralized feature definition, preprocessing, and validation
"""

import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


@dataclass
class FeatureConfig:
    """Configuration for a single feature"""
    name: str
    feature_type: str  # FeatureType value
    required: bool = True
    default: Optional[Any] = None
    description: str = ""
    validation_rules: Optional[Dict] = None
    transformation: Optional[str] = None  # 'log', 'sqrt', 'normalize', etc.


@dataclass
class FeatureGroup:
    """Group of related features"""
    group_name: str
    features: List[FeatureConfig]
    description: str = ""


class FeatureManager:
    """Centralized feature management"""
    
    def __init__(self, config_file: str = "features_config.json"):
        """
        Initialize feature manager
        
        Args:
            config_file: Path to feature configuration file
        """
        self.config_file = Path(config_file)
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_map: Dict[str, FeatureConfig] = {}
        self.transformations: Dict[str, Callable] = {}
        
        # Register default transformations
        self._register_default_transformations()
        
        # Load config if exists
        self._load_config()
    
    def add_feature(
        self,
        name: str,
        feature_type: str,
        group: str = "default",
        required: bool = True,
        default: Optional[Any] = None,
        description: str = "",
        validation_rules: Optional[Dict] = None,
        transformation: Optional[str] = None
    ) -> FeatureConfig:
        """Add a feature configuration"""
        config = FeatureConfig(
            name=name,
            feature_type=feature_type,
            required=required,
            default=default,
            description=description,
            validation_rules=validation_rules,
            transformation=transformation
        )
        
        self.feature_map[name] = config
        
        # Add to group
        if group not in self.feature_groups:
            self.feature_groups[group] = FeatureGroup(
                group_name=group,
                features=[],
                description=f"Feature group: {group}"
            )
        
        self.feature_groups[group].features.append(config)
        logger.info(f"Added feature: {name} to group: {group}")
        
        return config
    
    def add_feature_group(
        self,
        group_name: str,
        features: List[FeatureConfig],
        description: str = ""
    ):
        """Add a feature group"""
        self.feature_groups[group_name] = FeatureGroup(
            group_name=group_name,
            features=features,
            description=description
        )
        
        for feature in features:
            self.feature_map[feature.name] = feature
        
        logger.info(f"Added feature group: {group_name} with {len(features)} features")
    
    def validate_data(self, data: pd.DataFrame) -> tuple:
        """
        Validate input data
        
        Args:
            data: Input DataFrame
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        for feature_name, config in self.feature_map.items():
            # Check if required feature exists
            if config.required and feature_name not in data.columns:
                errors.append(f"Missing required feature: {feature_name}")
                continue
            
            if feature_name not in data.columns:
                continue
            
            # Validate data type
            col_data = data[feature_name]
            
            if config.feature_type == FeatureType.NUMERIC.value:
                if not pd.api.types.is_numeric_dtype(col_data):
                    errors.append(f"Feature {feature_name} should be numeric")
            
            elif config.feature_type == FeatureType.BOOLEAN.value:
                if not all(col_data.isin([0, 1, True, False])):
                    errors.append(f"Feature {feature_name} should be boolean")
            
            # Validate custom rules
            if config.validation_rules:
                if "min" in config.validation_rules:
                    min_val = config.validation_rules["min"]
                    if (col_data < min_val).any():
                        errors.append(f"Feature {feature_name} has values < {min_val}")
                
                if "max" in config.validation_rules:
                    max_val = config.validation_rules["max"]
                    if (col_data > max_val).any():
                        errors.append(f"Feature {feature_name} has values > {max_val}")
                
                if "allowed_values" in config.validation_rules:
                    allowed = config.validation_rules["allowed_values"]
                    if not col_data.isin(allowed).all():
                        errors.append(f"Feature {feature_name} has invalid values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        transformations: Optional[Dict[str, Callable]] = None
    ) -> pd.DataFrame:
        """
        Preprocess data using feature configurations
        
        Args:
            data: Input DataFrame
            transformations: Custom transformations to apply
        
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Fill missing values
        for feature_name, config in self.feature_map.items():
            if feature_name not in df.columns:
                if config.default is not None:
                    df[feature_name] = config.default
                continue
            
            # Fill NaN with default or mean/mode
            if df[feature_name].isna().any():
                if config.default is not None:
                    df[feature_name].fillna(config.default, inplace=True)
                elif config.feature_type == FeatureType.NUMERIC.value:
                    df[feature_name].fillna(df[feature_name].mean(), inplace=True)
                else:
                    df[feature_name].fillna(df[feature_name].mode()[0], inplace=True)
        
        # Apply transformations
        for feature_name, config in self.feature_map.items():
            if feature_name not in df.columns:
                continue
            
            if config.transformation:
                if config.transformation in self.transformations:
                    tf = self.transformations[config.transformation]
                    df[feature_name] = tf(df[feature_name])
                    logger.info(f"Applied {config.transformation} to {feature_name}")
            
            # Apply custom transformations
            if transformations and feature_name in transformations:
                df[feature_name] = transformations[feature_name](df[feature_name])
        
        return df
    
    def get_feature_names(
        self,
        group: Optional[str] = None,
        feature_type: Optional[str] = None
    ) -> List[str]:
        """Get feature names with optional filters"""
        features = []
        
        for name, config in self.feature_map.items():
            if group and name not in self.feature_groups.get(group, FeatureGroup("", [])).features:
                continue
            if feature_type and config.feature_type != feature_type:
                continue
            features.append(name)
        
        return features
    
    def get_feature_config(self, name: str) -> Optional[FeatureConfig]:
        """Get configuration for a feature"""
        return self.feature_map.get(name)
    
    def register_transformation(self, name: str, func: Callable):
        """Register a custom transformation function"""
        self.transformations[name] = func
        logger.info(f"Registered transformation: {name}")
    
    def list_features(self) -> List[Dict]:
        """List all features with their configurations"""
        features = []
        for name, config in self.feature_map.items():
            features.append({
                "name": name,
                "type": config.feature_type,
                "required": config.required,
                "description": config.description,
                "transformation": config.transformation
            })
        return features
    
    def _register_default_transformations(self):
        """Register default transformation functions"""
        self.transformations["log"] = lambda x: np.log1p(x.clip(lower=0))
        self.transformations["sqrt"] = lambda x: np.sqrt(x.clip(lower=0))
        self.transformations["normalize"] = lambda x: (x - x.mean()) / (x.std() + 1e-8)
        self.transformations["standardize"] = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    def save_config(self):
        """Save feature configuration to file"""
        config_data = {}
        
        for group_name, group in self.feature_groups.items():
            config_data[group_name] = {
                "description": group.description,
                "features": [
                    {
                        "name": f.name,
                        "type": f.feature_type,
                        "required": f.required,
                        "default": f.default,
                        "description": f.description,
                        "validation_rules": f.validation_rules,
                        "transformation": f.transformation
                    }
                    for f in group.features
                ]
            }
        
        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved feature config to: {self.config_file}")
    
    def _load_config(self):
        """Load feature configuration from file"""
        if not self.config_file.exists():
            logger.info("No existing feature config found")
            return
        
        try:
            with open(self.config_file, "r") as f:
                config_data = json.load(f)
            
            for group_name, group_data in config_data.items():
                features = [
                    FeatureConfig(
                        name=f["name"],
                        feature_type=f["type"],
                        required=f.get("required", True),
                        default=f.get("default"),
                        description=f.get("description", ""),
                        validation_rules=f.get("validation_rules"),
                        transformation=f.get("transformation")
                    )
                    for f in group_data.get("features", [])
                ]
                
                self.add_feature_group(
                    group_name,
                    features,
                    group_data.get("description", "")
                )
            
            logger.info(f"Loaded {len(self.feature_map)} features from config")
        except Exception as e:
            logger.error(f"Failed to load feature config: {str(e)}")


# Global instance
_feature_manager = None


def get_feature_manager(config_file: str = "features_config.json") -> FeatureManager:
    """Get or create the global feature manager"""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureManager(config_file)
    return _feature_manager
