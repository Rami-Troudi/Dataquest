"""
MLOps API Client
Simple Python client for testing the Insurance Model API
"""

import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PredictionData:
    """Data class for prediction input"""
    user_id: str
    adult_dependents: int
    child_dependents: int
    infant_dependents: int
    previous_claims_filed: int
    previous_policy_duration_months: int
    years_without_claims: int
    existing_policyholder: int
    policy_cancelled_post_purchase: int
    policy_amendments_count: int
    custom_riders_requested: int
    vehicles_on_policy: int
    days_since_quote: int
    underwriting_processing_days: int
    grace_period_extensions: int
    policy_start_month: str
    deductible_tier: str
    broker_id: Optional[str] = None
    employer_id: Optional[str] = None

    def to_dict(self):
        return {
            "User_ID": self.user_id,
            "Adult_Dependents": self.adult_dependents,
            "Child_Dependents": self.child_dependents,
            "Infant_Dependents": self.infant_dependents,
            "Previous_Claims_Filed": self.previous_claims_filed,
            "Previous_Policy_Duration_Months": self.previous_policy_duration_months,
            "Years_Without_Claims": self.years_without_claims,
            "Existing_Policyholder": self.existing_policyholder,
            "Policy_Cancelled_Post_Purchase": self.policy_cancelled_post_purchase,
            "Policy_Amendments_Count": self.policy_amendments_count,
            "Custom_Riders_Requested": self.custom_riders_requested,
            "Vehicles_on_Policy": self.vehicles_on_policy,
            "Days_Since_Quote": self.days_since_quote,
            "Underwriting_Processing_Days": self.underwriting_processing_days,
            "Grace_Period_Extensions": self.grace_period_extensions,
            "Policy_Start_Month": self.policy_start_month,
            "Deductible_Tier": self.deductible_tier,
            "Broker_ID": self.broker_id,
            "Employer_ID": self.employer_id,
        }


class MLOpsAPIClient:
    """Client for MLOps API"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, data: PredictionData) -> Dict:
        """Make a single prediction"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=data.to_dict()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def predict_batch(self, data_list: List[PredictionData]) -> List[Dict]:
        """Make batch predictions"""
        try:
            payload = {
                "predictions": [data.to_dict() for data in data_list]
            }
            response = self.session.post(
                f"{self.base_url}/predict_batch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Batch prediction failed: {e}")
            return [{"error": str(e)}]
    
    def get_docs_url(self) -> str:
        """Get the API documentation URL"""
        return f"{self.base_url}/docs"


# Example usage
if __name__ == "__main__":
    print("MLOps API Client - Example Usage")
    print("=" * 50)
    
    # Initialize client
    client = MLOpsAPIClient("http://localhost")
    
    # Check health
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Example prediction data
    print("\n2. Single Prediction:")
    test_data = PredictionData(
        user_id="USER_001",
        adult_dependents=1,
        child_dependents=2,
        infant_dependents=0,
        previous_claims_filed=1,
        previous_policy_duration_months=24,
        years_without_claims=5,
        existing_policyholder=1,
        policy_cancelled_post_purchase=0,
        policy_amendments_count=2,
        custom_riders_requested=1,
        vehicles_on_policy=2,
        days_since_quote=5,
        underwriting_processing_days=2,
        grace_period_extensions=0,
        policy_start_month="January",
        deductible_tier="Medium",
        broker_id="B001",
        employer_id="E001"
    )
    
    prediction = client.predict(test_data)
    print(json.dumps(prediction, indent=2))
    
    # Batch predictions
    print("\n3. Batch Predictions:")
    test_data_2 = PredictionData(
        user_id="USER_002",
        adult_dependents=0,
        child_dependents=1,
        infant_dependents=1,
        previous_claims_filed=0,
        previous_policy_duration_months=12,
        years_without_claims=3,
        existing_policyholder=0,
        policy_cancelled_post_purchase=0,
        policy_amendments_count=1,
        custom_riders_requested=0,
        vehicles_on_policy=1,
        days_since_quote=10,
        underwriting_processing_days=3,
        grace_period_extensions=1,
        policy_start_month="February",
        deductible_tier="High"
    )
    
    batch_predictions = client.predict_batch([test_data, test_data_2])
    print(json.dumps(batch_predictions, indent=2))
    
    # API docs
    print(f"\n4. API Documentation: {client.get_docs_url()}")
