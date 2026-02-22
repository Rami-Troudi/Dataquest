"""
Quick Integration Test
Tests the MLOps layer with existing model.pkl
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from mlops.model_loader import get_model_server

def test_model_loading():
    """Test loading model.pkl"""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)
    
    server = get_model_server("mlops/models")
    
    try:
        # Load model
        print("\n📦 Loading model from model.pkl...")
        server.load_model(
            model_path="model.pkl",
            model_name="insurance_rf",
            version="1.0.0"
        )
        
        # Get model info
        info = server.get_current_model_info()
        print(f"✓ Model loaded successfully!")
        print(f"  Type: {info['model_type']}")
        print(f"  Features: {info['features_count']}")
        print(f"  Classes: {info['class_order']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return False


def test_single_prediction():
    """Test single prediction"""
    print("\n" + "=" * 80)
    print("TEST 2: Single Prediction")
    print("=" * 80)
    
    try:
        server = get_model_server("mlops/models")
        
        # Load test data
        print("\n📖 Loading test data...")
        test_df = pd.read_csv("test.csv")
        print(f"✓ Loaded {len(test_df)} rows")
        
        # Get first record
        first_record = test_df.iloc[[0]]
        user_id = first_record['User_ID'].values[0]
        
        print(f"\n🔮 Making prediction for {user_id}...")
        
        # Predict
        result = server.predict_with_confidence(first_record)
        
        prediction = result['predictions'][0]
        confidence = result['confidence'][0]
        
        print(f"✓ Prediction: {prediction}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Probabilities: {result['probabilities'][0]}")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Prediction (first 10 records)")
    print("=" * 80)
    
    try:
        server = get_model_server("mlops/models")
        
        # Load test data
        print("\n📖 Loading test data...")
        test_df = pd.read_csv("test.csv").head(10)
        print(f"✓ Loaded {len(test_df)} rows")
        
        print(f"\n🔮 Making batch predictions...")
        
        # Predict
        result = server.predict_with_confidence(test_df)
        
        predictions = result['predictions']
        confidence = result['confidence']
        
        print(f"✓ Predictions completed!")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Average confidence: {confidence.mean():.4f}")
        
        # Show sample
        print(f"\n  Sample results:")
        for i in range(min(5, len(test_df))):
            user_id = test_df['User_ID'].iloc[i]
            print(f"    {user_id}: {predictions[i]} (confidence: {confidence[i]:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Batch prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_features():
    """Test feature extraction"""
    print("\n" + "=" * 80)
    print("TEST 4: Required Features")
    print("=" * 80)
    
    try:
        server = get_model_server("mlops/models")
        
        features = server.get_required_features()
        
        print(f"\n✓ Model requires {len(features)} features:")
        for i, feature in enumerate(features[:10], 1):
            print(f"  {i}. {feature}")
        
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")
        
        return True
    except Exception as e:
        print(f"✗ Failed to get features: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " MLOps Integration Tests - Using Existing model.pkl ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    results = []
    
    # Run tests
    results.append(("Model Loading", test_model_loading()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("Features", test_features()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! MLOps layer is working correctly.")
        print("\n📚 Next steps:")
        print("  1. Run the API: python -m uvicorn mlops.app_v3:app --reload")
        print("  2. Try predictions at http://localhost:8000/docs")
        print("  3. Load different models with POST /model/load")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
