"""
Test the solution.py implementation
"""

import pandas as pd
import sys
sys.path.append('.')

from solution import preprocess, load_model, predict

print("="*80)
print("TESTING SOLUTION.PY")
print("="*80)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv('test.csv')
print(f"Test data shape: {test_df.shape}")

# Test preprocessing
print("\n2. Testing preprocess()...")
test_processed = preprocess(test_df)
print(f"After preprocessing shape: {test_processed.shape}")
print(f"New columns added: {test_processed.shape[1] - test_df.shape[1]}")
print(f"Sample columns: {list(test_processed.columns[:10])}")

# Test model loading
print("\n3. Testing load_model()...")
try:
    model = load_model()
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Note: Run analysis_and_model.py first to generate model.pkl")
    sys.exit(1)

# Test prediction
print("\n4. Testing predict()...")
predictions = predict(test_processed, model)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions columns: {list(predictions.columns)}")
print(f"\nFirst 10 predictions:")
print(predictions.head(10))

# Validate output format
print("\n5. Validating output format...")
assert predictions.shape[1] == 2, "Output must have exactly 2 columns"
assert 'User_ID' in predictions.columns, "Output must have 'User_ID' column"
assert 'Purchased_Coverage_Bundle' in predictions.columns, "Output must have 'Purchased_Coverage_Bundle' column"
assert predictions.shape[0] == test_df.shape[0], "Number of predictions must match test set size"
assert predictions['Purchased_Coverage_Bundle'].dtype in ['int64', 'int32'], "Predictions must be integers"
assert predictions['Purchased_Coverage_Bundle'].min() >= 0, "Predictions must be >= 0"
assert predictions['Purchased_Coverage_Bundle'].max() <= 9, "Predictions must be <= 9"

print("\n✓ All validations passed!")

print("\n6. Prediction distribution:")
print(predictions['Purchased_Coverage_Bundle'].value_counts().sort_index())

print("\n" + "="*80)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*80)
