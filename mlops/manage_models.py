"""
CLI Tool for Model Management
Easy command-line interface for managing models and switching between them
"""

import json
import sys
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from model_registry import get_registry
from feature_manager import get_feature_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCLI:
    """Command-line interface for model management"""
    
    def __init__(self, models_dir: str = "models", config_file: str = "features_config.json"):
        self.registry = get_registry(models_dir)
        self.feature_manager = get_feature_manager(config_file)
    
    def register_model(
        self,
        name: str,
        version: str,
        model_type: str,
        file_path: str,
        features_file: Optional[str] = None,
        description: str = "",
        accuracy: Optional[float] = None
    ):
        """Register a new model"""
        print(f"\n📦 Registering model: {name}:{version}")
        print(f"   Type: {model_type}")
        print(f"   File: {file_path}")
        
        # Load features
        features = []
        if features_file:
            with open(features_file, 'r') as f:
                features = json.load(f)
            print(f"   Features: {len(features)} loaded from {features_file}")
        
        # Register
        self.registry.register_model(
            name=name,
            version=version,
            model_type=model_type,
            file_path=file_path,
            features=features,
            description=description,
            accuracy=accuracy,
            status="active"
        )
        
        print("✅ Model registered successfully!")
    
    def set_current_model(self, name: str, version: str = "latest"):
        """Set current active model"""
        print(f"\n🔄 Switching to model: {name}:{version}")
        
        self.registry.set_current_model(name, version)
        
        print(f"✅ Current model set to: {name}:{version}")
    
    def list_models(self, status: Optional[str] = None, verbose: bool = False):
        """List all models"""
        models = self.registry.list_models(status=status)
        
        print(f"\n📚 Available Models ({len(models)} total)")
        print("-" * 100)
        
        for model in models:
            marker = "→" if model["key"] == self.registry.current_model else " "
            print(f"{marker} {model['key']:<30} | Status: {model['status']:<10} | Features: {model.get('features_count', 0)}")
            
            if verbose:
                print(f"  Type: {model['type']}")
                print(f"  Accuracy: {model['accuracy']}")
                print(f"  Description: {model['description']}")
        
        print(f"\nCurrent model: {self.registry.current_model}")
    
    def get_model_info(self, name: str, version: str = "latest"):
        """Get detailed model information"""
        key = f"{name}:{version}"
        
        if key not in self.registry.registry:
            print(f"❌ Model not found: {key}")
            return
        
        metadata = self.registry.registry[key]
        
        print(f"\n📋 Model Information:")
        print("-" * 50)
        print(f"Name:         {metadata.name}")
        print(f"Version:      {metadata.version}")
        print(f"Type:         {metadata.model_type}")
        print(f"Status:       {metadata.status}")
        print(f"File:         {metadata.file_path}")
        print(f"Created:      {metadata.created_date}")
        print(f"Accuracy:     {metadata.accuracy}")
        print(f"Description:  {metadata.description}")
        print(f"Features:     {len(metadata.features)}")
        
        if len(metadata.features) > 0:
            print(f"\nRequired Features:")
            for i, feature in enumerate(metadata.features[:10], 1):
                print(f"  {i}. {feature}")
            if len(metadata.features) > 10:
                print(f"  ... and {len(metadata.features) - 10} more")
    
    def delete_model(self, name: str, version: str):
        """Delete a model"""
        key = f"{name}:{version}"
        
        response = input(f"⚠️  Delete model {key}? (yes/no): ").lower()
        if response != "yes":
            print("❌ Cancelled")
            return
        
        self.registry.delete_model(name, version)
        print(f"✅ Model deleted: {key}")
    
    def add_feature(
        self,
        name: str,
        feature_type: str,
        group: str = "default",
        description: str = ""
    ):
        """Add a feature to configuration"""
        print(f"\n✏️  Adding feature: {name}")
        
        self.feature_manager.add_feature(
            name=name,
            feature_type=feature_type,
            group=group,
            description=description
        )
        
        self.feature_manager.save_config()
        print(f"✅ Feature added to group '{group}'")
    
    def list_features(self, group: Optional[str] = None):
        """List all features"""
        features = self.feature_manager.list_features()
        
        if group:
            # Filter by group - need to check feature_map
            group_features = [
                f for f in self.feature_manager.feature_map.items()
                if f[0] in features
            ]
        
        print(f"\n📋 Features ({len(features)} total)")
        print("-" * 80)
        
        for feature in features:
            print(f"  • {feature['name']:<30} | Type: {feature['type']:<12} | {feature['description']}")
    
    def validate_data(self, data_file: str):
        """Validate data against feature schema"""
        print(f"\n🔍 Validating data: {data_file}")
        
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.json'):
                df = pd.read_json(data_file)
            else:
                print("❌ Unsupported file format (use .csv or .json)")
                return
            
            is_valid, errors = self.feature_manager.validate_data(df)
            
            print(f"Records: {len(df)}")
            
            if is_valid:
                print("✅ Data validation passed!")
            else:
                print("❌ Validation errors found:")
                for error in errors:
                    print(f"  • {error}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def compare_models(self, model1: str, model2: str, version: str = "latest"):
        """Compare two models"""
        key1 = f"{model1}:{version}"
        key2 = f"{model2}:{version}"
        
        if key1 not in self.registry.registry or key2 not in self.registry.registry:
            print("❌ One or both models not found")
            return
        
        m1 = self.registry.registry[key1]
        m2 = self.registry.registry[key2]
        
        print(f"\n📊 Model Comparison")
        print("-" * 60)
        print(f"{'Attribute':<25} | {'Model 1':<15} | {'Model 2':<15}")
        print("-" * 60)
        print(f"{'Type':<25} | {m1.model_type:<15} | {m2.model_type:<15}")
        print(f"{'Status':<25} | {m1.status:<15} | {m2.status:<15}")
        print(f"{'Accuracy':<25} | {str(m1.accuracy):<15} | {str(m2.accuracy):<15}")
        print(f"{'Features Count':<25} | {str(len(m1.features)):<15} | {str(len(m2.features)):<15}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MLOps Model Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a model
  python manage_models.py register --name insurance_lgb --version 1.0.0 \\
    --type lightgbm --file models/insurance_lgb.joblib

  # Switch to a model
  python manage_models.py switch --name insurance_lgb --version 1.0.0

  # List all models
  python manage_models.py list --verbose

  # Get model information
  python manage_models.py info --name insurance_lgb --version 1.0.0

  # Add a feature
  python manage_models.py add-feature --name Adult_Dependents --type numeric \\
    --group household --description "Number of adult dependents"

  # List features
  python manage_models.py list-features

  # Validate data
  python manage_models.py validate --file data.csv

  # Compare models
  python manage_models.py compare --model1 insurance_lgb --model2 insurance_xgb
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register model
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--version", required=True, help="Model version")
    register_parser.add_argument("--type", required=True, help="Model type (lightgbm, xgboost, rf_sklearn)")
    register_parser.add_argument("--file", required=True, help="Path to model file")
    register_parser.add_argument("--features-file", help="Path to features JSON file")
    register_parser.add_argument("--description", default="", help="Model description")
    register_parser.add_argument("--accuracy", type=float, help="Model accuracy")
    
    # Switch model
    switch_parser = subparsers.add_parser("switch", help="Switch to a model")
    switch_parser.add_argument("--name", required=True, help="Model name")
    switch_parser.add_argument("--version", default="latest", help="Model version")
    
    # List models
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.add_argument("--status", help="Filter by status (active, deprecated, experimental)")
    list_parser.add_argument("--verbose", action="store_true", help="Show detailed info")
    
    # Model info
    info_parser = subparsers.add_parser("info", help="Get model information")
    info_parser.add_argument("--name", required=True, help="Model name")
    info_parser.add_argument("--version", default="latest", help="Model version")
    
    # Delete model
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("--name", required=True, help="Model name")
    delete_parser.add_argument("--version", required=True, help="Model version")
    
    # Add feature
    add_feature_parser = subparsers.add_parser("add-feature", help="Add a feature")
    add_feature_parser.add_argument("--name", required=True, help="Feature name")
    add_feature_parser.add_argument("--type", required=True, help="Feature type (numeric, categorical, boolean, datetime)")
    add_feature_parser.add_argument("--group", default="default", help="Feature group")
    add_feature_parser.add_argument("--description", default="", help="Feature description")
    
    # List features
    list_features_parser = subparsers.add_parser("list-features", help="List all features")
    list_features_parser.add_argument("--group", help="Filter by group")
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("--file", required=True, help="Path to data file (csv or json)")
    
    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model1", required=True, help="First model name")
    compare_parser.add_argument("--model2", required=True, help="Second model name")
    compare_parser.add_argument("--version", default="latest", help="Model version")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ModelCLI()
    
    try:
        if args.command == "register":
            cli.register_model(
                name=args.name,
                version=args.version,
                model_type=args.type,
                file_path=args.file,
                features_file=args.features_file,
                description=args.description,
                accuracy=args.accuracy
            )
        
        elif args.command == "switch":
            cli.set_current_model(args.name, args.version)
        
        elif args.command == "list":
            cli.list_models(status=args.status, verbose=args.verbose)
        
        elif args.command == "info":
            cli.get_model_info(args.name, args.version)
        
        elif args.command == "delete":
            cli.delete_model(args.name, args.version)
        
        elif args.command == "add-feature":
            cli.add_feature(args.name, args.type, args.group, args.description)
        
        elif args.command == "list-features":
            cli.list_features(args.group)
        
        elif args.command == "validate":
            cli.validate_data(args.file)
        
        elif args.command == "compare":
            cli.compare_models(args.model1, args.model2, args.version)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
