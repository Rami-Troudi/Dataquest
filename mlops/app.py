"""
MLOps REST API for Model Serving
Serves predictions from the trained insurance model via HTTP
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Model API",
    description="MLOps REST API for insurance policy prediction",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_loaded = False


# ==================== DATA MODELS ====================

class PredictionInput(BaseModel):
    """Input data for prediction"""
    User_ID: str
    Adult_Dependents: int
    Child_Dependents: int
    Infant_Dependents: int
    Previous_Claims_Filed: int
    Previous_Policy_Duration_Months: int
    Years_Without_Claims: int
    Existing_Policyholder: int
    Policy_Cancelled_Post_Purchase: int
    Policy_Amendments_Count: int
    Custom_Riders_Requested: int
    Vehicles_on_Policy: int
    Days_Since_Quote: int
    Underwriting_Processing_Days: int
    Grace_Period_Extensions: int
    Policy_Start_Month: str
    Deductible_Tier: str
    Broker_ID: Optional[str] = None
    Employer_ID: Optional[str] = None


class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    predictions: List[PredictionInput]


class PredictionResponse(BaseModel):
    """Prediction response"""
    User_ID: str
    prediction: float
    probability: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


# ==================== INFERENCE LOGIC ====================

def preprocess(df):
    """
    Preprocess input data - mirrors the training preprocessing
    """
    _MONTH_MAP = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    d = df.copy()

    # Household features
    d["Total_Dependents"] = (
        d["Adult_Dependents"] + d["Child_Dependents"] + d["Infant_Dependents"]
    )
    d["Has_Dependents"] = (d["Total_Dependents"] > 0).astype(np.int8)
    d["Has_Child"] = (d["Child_Dependents"] > 0).astype(np.int8)
    d["Has_Infant"] = (d["Infant_Dependents"] > 0).astype(np.int8)

    # Risk and loyalty features
    d["Claims_Rate"] = d["Previous_Claims_Filed"] / (d["Previous_Policy_Duration_Months"] + 1)
    d["Risk_Score"] = d["Previous_Claims_Filed"] / (d["Years_Without_Claims"] + 1)
    d["Loyalty_Score"] = (
        d["Years_Without_Claims"] * 0.3
        + d["Previous_Policy_Duration_Months"] * 0.02
        - d["Previous_Claims_Filed"] * 0.5
    )
    d["Claimfree_vs_Claims"] = d["Years_Without_Claims"] / (1 + d["Previous_Claims_Filed"])
    d["Existing_x_Claims"] = d["Existing_Policyholder"] * d["Previous_Claims_Filed"]

    # Policy features
    d["Post_Purchase_Activity"] = (
        d["Policy_Cancelled_Post_Purchase"] * d["Policy_Amendments_Count"]
    )
    d["Policy_Complexity"] = d["Custom_Riders_Requested"] + d["Vehicles_on_Policy"]
    d["Quote_To_UW_Ratio"] = d["Days_Since_Quote"] / (d["Underwriting_Processing_Days"] + 1)
    d["PostPurchase_v2"] = d["Grace_Period_Extensions"] + d["Policy_Amendments_Count"]
    d["Grace_per_Month"] = d["Grace_Period_Extensions"] / (1 + d["Previous_Policy_Duration_Months"])

    # Temporal features
    month_num = (
        d["Policy_Start_Month"].astype(str).str.strip().map(_MONTH_MAP).fillna(6).astype(int)
    )
    d["Month_Sin"] = np.sin(2 * np.pi * month_num / 12)
    d["Month_Cos"] = np.cos(2 * np.pi * month_num / 12)

    # Presence flags
    d["Has_Broker"] = d["Broker_ID"].notna().astype(np.int8)
    d["Has_Employer"] = d["Employer_ID"].notna().astype(np.int8)

    # Tenure bucket
    tenure = d["Previous_Policy_Duration_Months"].fillna(0).values
    d["Tenure_Bucket"] = np.digitize(tenure, bins=[1, 2, 3, 4]).astype(np.int8)

    # Underwriting friction
    d["Underwriting_Friction"] = np.log1p(
        d["Underwriting_Processing_Days"].fillna(0).clip(lower=0)
    )

    # Deductible x risk interaction
    ded_map = {"Low": 0, "Medium": 1, "High": 2}
    ded_ord = d["Deductible_Tier"].astype(str).map(ded_map).fillna(1)
    d["Deductible_x_Risk"] = d["Claims_Rate"] * ded_ord

    return d


def load_model():
    """Load the trained model from joblib"""
    global model, model_loaded
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        model = joblib.load(model_path)
        model_loaded = True
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False
        raise


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting up MLOps API...")
    try:
        load_model()
        logger.info("MLOps API ready for predictions")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    """
    Single prediction endpoint
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Preprocess
        df_processed = preprocess(df)
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        
        return PredictionResponse(
            User_ID=data.User_ID,
            prediction=float(prediction)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(data: BatchPredictionInput):
    """
    Batch prediction endpoint
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        
        # Convert inputs to DataFrame
        df = pd.DataFrame([item.dict() for item in data.predictions])
        
        # Preprocess
        df_processed = preprocess(df)
        
        # Make predictions
        predictions = model.predict(df_processed)
        
        # Build responses
        for i, pred_input in enumerate(data.predictions):
            results.append(PredictionResponse(
                User_ID=pred_input.User_ID,
                prediction=float(predictions[i])
            ))
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Insurance Model MLOps API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
