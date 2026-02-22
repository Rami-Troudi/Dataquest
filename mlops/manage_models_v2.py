"""
Model Management CLI
Easy command-line tool for managing and switching models
"""

import sys
import argparse
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.model_loader import get_model_server
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cmd_load(args):
    """Load a model"""
    server = get_model_server("mlops/models")
    
    print(f"\n📦 Loading model from: {args.path}")
    print(f"   Name: {args.name}")
    print(f"   Version: {args.version}")
    
    try:
        model_info = server.load_model(
            model_path=args.path,
            model_name=args.name,
            version=args.version
        )
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Key: {args.name}:{args.version}")
        print(f"   File: {args.path}")
        
        # Show model info
        info = server.get_current_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Type: {info.get('model_type')}")
        print(f"   Features: {info.get('features_count')}")
        print(f"   Classes: {info.get('class_order')}")
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {str(e)}")
        sys.exit(1)


def cmd_info(args):
    """Show current model information"""
    server = get_model_server("mlops/models")
    
    if server.current_bundle is None:
        print("❌ No model currently loaded")
        sys.exit(1)
    
    print(f"\n📋 Current Model: {server.current_model_key}")
    print("-" * 60)
    
    info = server.get_current_model_info()
    
    print(f"Type:          {info.get('model_type')}")
    print(f"Features:      {info.get('features_count')}")
    print(f"Classes:       {info.get('class_order')}")
    
    tuning = info.get('tuning', {})
    if tuning:
        print(f"\n📈 Tuning Configuration:")
        for key, value in tuning.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def cmd_features(args):
    """List required features for current model"""
    server = get_model_server("mlops/models")
    
    if server.current_bundle is None:
        print("❌ No model currently loaded")
        sys.exit(1)
    
    features = server.get_required_features()
    
    print(f"\n📊 Required Features ({len(features)} total):")
    print("-" * 60)
    
    for i, feature in enumerate(features, 1):
        print(f"{i:3}. {feature}")


def cmd_predict(args):
    """Make prediction on CSV file"""
    import pandas as pd
    
    server = get_model_server("mlops/models")
    
    if server.current_bundle is None:
        print("❌ No model currently loaded")
        sys.exit(1)
    
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    
    print(f"\n📖 Loading data from: {args.input}")
    
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} records")
        
        print(f"\n🔮 Making predictions...")
        result = server.predict_with_confidence(df)
        
        # Get user IDs if available
        user_ids = df['User_ID'].tolist() if 'User_ID' in df.columns else list(range(len(df)))
        
        # Save results
        output = pd.DataFrame({
            'User_ID': user_ids,
            'Prediction': result['predictions'],
            'Confidence': result['confidence']
        })
        
        if args.output:
            output.to_csv(args.output, index=False)
            print(f"\n✅ Predictions saved to: {args.output}")
        else:
            print(f"\n✅ Predictions:")
            print(output.head(10))
            if len(output) > 10:
                print(f"... and {len(output) - 10} more")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI"""
    parser = argparse.ArgumentParser(
        description="MLOps Model Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Load a model:
    python manage_models_v2.py load --path model.pkl --name insurance_rf

  Show current model info:
    python manage_models_v2.py info

  List required features:
    python manage_models_v2.py features

  Make predictions:
    python manage_models_v2.py predict --input test.csv --output predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('--path', required=True, help='Path to model file (.pkl or .joblib)')
    load_parser.add_argument('--name', default='default', help='Model name')
    load_parser.add_argument('--version', default='1.0.0', help='Model version')
    load_parser.set_defaults(func=cmd_load)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show current model information')
    info_parser.set_defaults(func=cmd_info)
    
    # Features command
    features_parser = subparsers.add_parser('features', help='List required features')
    features_parser.set_defaults(func=cmd_features)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on CSV file')
    predict_parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    predict_parser.add_argument('--output', '-o', help='Output CSV file for predictions')
    predict_parser.set_defaults(func=cmd_predict)
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
