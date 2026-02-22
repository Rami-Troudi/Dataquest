"""
Quick Start Guide for Model Registry and Feature Manager
Example setup to demonstrate easy model switching
"""

import json
import os
from model_registry import get_registry
from feature_manager import get_feature_manager, FeatureConfig, FeatureType


def setup_insurance_features():
    """Setup feature configuration for insurance model"""
    print("Setting up insurance features...")
    
    fm = get_feature_manager(config_file="features_config.json")
    
    # Household features group
    household_features = [
        FeatureConfig(
            name="Adult_Dependents",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0, "max": 10},
            description="Number of adult dependents"
        ),
        FeatureConfig(
            name="Child_Dependents",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0, "max": 20},
            description="Number of child dependents"
        ),
        FeatureConfig(
            name="Infant_Dependents",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0, "max": 10},
            description="Number of infant dependents"
        ),
    ]
    
    fm.add_feature_group("household", household_features, "Household composition features")
    
    # Claims and history
    claims_features = [
        FeatureConfig(
            name="Previous_Claims_Filed",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0},
            description="Number of previous claims filed"
        ),
        FeatureConfig(
            name="Years_Without_Claims",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0},
            description="Years without filing a claim"
        ),
    ]
    
    fm.add_feature_group("claims", claims_features, "Claims history features")
    
    # Policy features
    policy_features = [
        FeatureConfig(
            name="Previous_Policy_Duration_Months",
            feature_type="numeric",
            required=True,
            validation_rules={"min": 0},
            description="Previous policy duration in months"
        ),
        FeatureConfig(
            name="Policy_Start_Month",
            feature_type="categorical",
            required=True,
            validation_rules={"allowed_values": [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]},
            description="Policy start month"
        ),
        FeatureConfig(
            name="Deductible_Tier",
            feature_type="categorical",
            required=True,
            validation_rules={"allowed_values": ["Low", "Medium", "High"]},
            description="Deductible tier"
        ),
    ]
    
    fm.add_feature_group("policy", policy_features, "Policy features")
    
    # Save configuration
    fm.save_config()
    print("✅ Features configured!")
    print()


def register_example_models():
    """Register example models"""
    print("Registering example models...")
    
    registry = get_registry(models_dir="models")
    
    # Create models directory if needed
    os.makedirs("models", exist_ok=True)
    
    # Get features list from existing model
    import joblib
    
    try:
        # Try to load existing model and extract features
        if os.path.exists("model.joblib"):
            raw_model = joblib.load("model.joblib")
            # LightGBM models have feature_name attribute
            if hasattr(raw_model, 'feature_name'):
                features = list(raw_model.feature_name())
            else:
                features = []
        else:
            features = []
    except:
        features = []
    
    # Register LightGBM model
    registry.register_model(
        name="insurance_lgb",
        version="1.0.0",
        model_type="lightgbm",
        file_path="model.joblib",
        features=features,
        description="LightGBM insurance policy prediction model",
        accuracy=0.85,
        status="active"
    )
    
    print("✅ Models registered!")
    print()


def show_models_info():
    """Display all registered models"""
    print("=" * 80)
    print("REGISTERED MODELS")
    print("=" * 80)
    
    registry = get_registry(models_dir="models")
    models = registry.list_models()
    
    for model in models:
        print(f"\n📦 {model['key']}")
        print(f"   Type: {model['type']}")
        print(f"   Status: {model['status']}")
        print(f"   Accuracy: {model.get('accuracy', 'N/A')}")
        print(f"   Features: {model['features_count']}")
        print(f"   Description: {model['description']}")
    
    print()


def show_features_info():
    """Display feature configuration"""
    print("=" * 80)
    print("CONFIGURED FEATURES")
    print("=" * 80)
    
    fm = get_feature_manager(config_file="features_config.json")
    features = fm.list_features()
    
    for feature in features:
        print(f"\n  • {feature['name']}")
        print(f"    Type: {feature['type']}")
        print(f"    Required: {feature['required']}")
        print(f"    Transform: {feature.get('transformation', 'None')}")
        print(f"    Description: {feature['description']}")
    
    print()


def example_prediction():
    """Example prediction with new system"""
    print("=" * 80)
    print("EXAMPLE PREDICTION")
    print("=" * 80)
    
    from app_v2 import preprocess_insurance_data
    import pandas as pd
    
    registry = get_registry(models_dir="models")
    
    # Set current model
    registry.set_current_model("insurance_lgb", "1.0.0")
    
    # Load model
    model = registry.get_current_model()
    
    # Prepare sample data
    sample_data = {
        "Adult_Dependents": 1,
        "Child_Dependents": 2,
        "Infant_Dependents": 0,
        "Previous_Claims_Filed": 1,
        "Previous_Policy_Duration_Months": 24,
        "Years_Without_Claims": 5,
        "Existing_Policyholder": 1,
        "Policy_Cancelled_Post_Purchase": 0,
        "Policy_Amendments_Count": 2,
        "Custom_Riders_Requested": 1,
        "Vehicles_on_Policy": 2,
        "Days_Since_Quote": 5,
        "Underwriting_Processing_Days": 2,
        "Grace_Period_Extensions": 0,
        "Policy_Start_Month": "January",
        "Deductible_Tier": "Medium",
        "Broker_ID": "B001",
        "Employer_ID": "E001"
    }
    
    df = pd.DataFrame([sample_data])
    df_processed = preprocess_insurance_data(df)
    
    # Get required features
    features = registry.get_model_features("insurance_lgb", "1.0.0")
    X = df_processed[features]
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    print(f"\nModel: {registry.current_model}")
    print(f"Prediction: {prediction:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MLOps MODEL REGISTRY & FEATURE MANAGER - QUICK START")
    print("=" * 80 + "\n")
    
    # Setup
    setup_insurance_features()
    register_example_models()
    
    # Display info
    show_models_info()
    show_features_info()
    
    # Example
    try:
        example_prediction()
    except Exception as e:
        print(f"⚠️  Prediction example skipped: {str(e)}")
    
    print("=" * 80)
    print("✅ Setup complete! You can now:")
    print("   1. Use manage_models.py CLI for model management")
    print("   2. Start app_v2.py for the enhanced API")
    print("   3. Switch models with: python manage_models.py switch --name insurance_lgb")
    print("=" * 80 + "\n")
