"""
MLOps API Test Suite
Comprehensive testing for the Insurance Model API
"""

import requests
import json
import time
from typing import Dict
import sys


class TestSuite:
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.tests_passed = 0
        self.tests_failed = 0
    
    def log(self, message: str, level: str = "INFO"):
        """Log test output"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def test_health_check(self) -> bool:
        """Test health endpoint"""
        self.log("Testing /health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            assert data.get("status") == "healthy", "API not healthy"
            assert data.get("model_loaded") == True, "Model not loaded"
            
            self.log("✓ Health check passed", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ Health check failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        self.log("Testing /predict endpoint...")
        try:
            payload = {
                "User_ID": "TEST_001",
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
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            data = response.json()
            
            assert "User_ID" in data, "Missing User_ID in response"
            assert "prediction" in data, "Missing prediction in response"
            assert isinstance(data["prediction"], (int, float)), "Invalid prediction type"
            
            self.log(f"✓ Single prediction passed ({elapsed_time:.2f}s): {data['prediction']}", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ Single prediction failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        self.log("Testing /predict_batch endpoint...")
        try:
            payload = {
                "predictions": [
                    {
                        "User_ID": "TEST_001",
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
                        "Deductible_Tier": "Medium"
                    },
                    {
                        "User_ID": "TEST_002",
                        "Adult_Dependents": 0,
                        "Child_Dependents": 1,
                        "Infant_Dependents": 1,
                        "Previous_Claims_Filed": 0,
                        "Previous_Policy_Duration_Months": 12,
                        "Years_Without_Claims": 3,
                        "Existing_Policyholder": 0,
                        "Policy_Cancelled_Post_Purchase": 0,
                        "Policy_Amendments_Count": 1,
                        "Custom_Riders_Requested": 0,
                        "Vehicles_on_Policy": 1,
                        "Days_Since_Quote": 10,
                        "Underwriting_Processing_Days": 3,
                        "Grace_Period_Extensions": 1,
                        "Policy_Start_Month": "February",
                        "Deductible_Tier": "High"
                    }
                ]
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict_batch",
                json=payload,
                timeout=30
            )
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            data = response.json()
            
            assert isinstance(data, list), "Response not a list"
            assert len(data) == 2, "Expected 2 predictions"
            assert all("prediction" in item for item in data), "Missing predictions"
            
            self.log(f"✓ Batch prediction passed ({elapsed_time:.2f}s): {len(data)} records", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ Batch prediction failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def test_invalid_request(self) -> bool:
        """Test error handling with invalid request"""
        self.log("Testing error handling...")
        try:
            payload = {
                "User_ID": "TEST_ERR",
                # Missing required fields
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )
            
            # Should return 422 (Validation Error)
            assert response.status_code == 422, f"Expected 422, got {response.status_code}"
            
            self.log("✓ Error handling test passed", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ Error handling test failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def test_api_docs(self) -> bool:
        """Test API documentation endpoint"""
        self.log("Testing API docs endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            response.raise_for_status()
            
            assert response.status_code == 200, "Docs not accessible"
            
            self.log("✓ API docs endpoint accessible", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ API docs test failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def test_response_schema(self) -> bool:
        """Test response schema validation"""
        self.log("Testing response schema...")
        try:
            payload = {
                "User_ID": "TEST_SCHEMA",
                "Adult_Dependents": 1,
                "Child_Dependents": 0,
                "Infant_Dependents": 0,
                "Previous_Claims_Filed": 0,
                "Previous_Policy_Duration_Months": 10,
                "Years_Without_Claims": 2,
                "Existing_Policyholder": 0,
                "Policy_Cancelled_Post_Purchase": 0,
                "Policy_Amendments_Count": 0,
                "Custom_Riders_Requested": 0,
                "Vehicles_on_Policy": 1,
                "Days_Since_Quote": 3,
                "Underwriting_Processing_Days": 1,
                "Grace_Period_Extensions": 0,
                "Policy_Start_Month": "March",
                "Deductible_Tier": "Low"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Check schema
            required_fields = ["User_ID", "prediction"]
            assert all(field in data for field in required_fields), "Missing required fields"
            assert 0 <= data["prediction"] <= 1, "Prediction out of valid range"
            
            self.log("✓ Response schema validation passed", "SUCCESS")
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"✗ Schema validation failed: {str(e)}", "ERROR")
            self.tests_failed += 1
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        self.log("=" * 60)
        self.log("Starting MLOps API Test Suite")
        self.log("=" * 60)
        
        print()
        
        # Run tests
        self.test_health_check()
        self.test_api_docs()
        self.test_single_prediction()
        self.test_batch_prediction()
        self.test_invalid_request()
        self.test_response_schema()
        
        # Summary
        print()
        self.log("=" * 60)
        self.log("Test Summary")
        self.log("=" * 60)
        self.log(f"Passed: {self.tests_passed}")
        self.log(f"Failed: {self.tests_failed}")
        
        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        self.log(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.tests_failed == 0:
            self.log("All tests passed! ✓", "SUCCESS")
            return True
        else:
            self.log(f"{self.tests_failed} test(s) failed", "ERROR")
            return False


if __name__ == "__main__":
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost"
    
    # Run tests
    suite = TestSuite(base_url)
    success = suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
