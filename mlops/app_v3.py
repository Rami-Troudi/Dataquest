"""
MLOps REST API for Model Serving with Easy Model Switching
Uses existing model.pkl and v4_rf_only preprocessing
Accepts input in test.csv format
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import os
import io
from datetime import datetime

from mlops.model_loader import get_model_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Model API - MLOps",
    description="REST API with model registry and easy model switching",
    version="3.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model server
model_server = None


# ==================== DATA MODELS ====================

class PredictionInput(BaseModel):
    """Single prediction input (test.csv format)"""
    User_ID: str
    Policy_Cancelled_Post_Purchase: int
    Policy_Start_Year: int
    Policy_Start_Week: int
    Policy_Start_Day: int
    Grace_Period_Extensions: int
    Previous_Policy_Duration_Months: int
    Adult_Dependents: int
    Child_Dependents: float
    Infant_Dependents: int
    Region_Code: str
    Existing_Policyholder: int
    Previous_Claims_Filed: int
    Years_Without_Claims: int
    Policy_Amendments_Count: int
    Broker_ID: Optional[float] = None
    Employer_ID: Optional[str] = None
    Underwriting_Processing_Days: int
    Vehicles_on_Policy: int
    Custom_Riders_Requested: int
    Broker_Agency_Type: str
    Deductible_Tier: str
    Acquisition_Channel: str
    Payment_Schedule: str
    Employment_Status: str
    Estimated_Annual_Income: float
    Days_Since_Quote: int
    Policy_Start_Month: str


class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    predictions: List[PredictionInput]


class PredictionResponse(BaseModel):
    """Prediction response"""
    User_ID: str
    prediction: int
    confidence: float
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str
    features_count: int
    class_order: List[int]
    tuning_config: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    current_model: Optional[str]
    version: str


# ==================== INITIALIZATION ====================

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global model_server
    
    logger.info("Starting MLOps API v3.0...")
    
    try:
        model_server = get_model_server("mlops/models")
        
        # Load default model
        model_path = "model.pkl"
        if os.path.exists(model_path):
            model_server.load_model(
                model_path=model_path,
                model_name="insurance_rf",
                version="1.0.0"
            )
            logger.info(f"✓ Default model loaded from {model_path}")
        else:
            logger.warning(f"Default model not found at {model_path}")
        
        logger.info("MLOps API ready!")
    
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


# ==================== CORE ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        info = model_server.get_current_model_info() if model_server else {}
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_server is not None and model_server.current_bundle is not None,
            current_model=model_server.current_model_key if model_server else None,
            version="3.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Insurance Model API - MLOps",
        "version": "3.0.0",
        "description": "Easy model switching with existing model.pkl and v4_rf_only preprocessing",
        "features": [
            "Single predictions",
            "Batch predictions",
            "CSV upload prediction",
            "Easy model switching",
            "Confidence scores"
        ],
        "quick_start": {
            "single_predict": "POST /predict",
            "batch_predict": "POST /predict_batch",
            "csv_predict": "POST /predict_csv",
            "health": "GET /health",
            "model_info": "GET /model/info"
        }
    }


# ==================== MODEL MANAGEMENT ====================

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information"""
    if model_server is None or model_server.current_bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        info = model_server.get_current_model_info()
        
        return ModelInfo(
            model_type=info.get('model_type', 'unknown'),
            features_count=info.get('features_count', 0),
            class_order=info.get('class_order', []),
            tuning_config=info.get('tuning', {})
        )
    except Exception as e:
        logger.error(f"Get model info error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/model/load")
async def load_model(model_path: str, model_name: str = "default", model_version: str = "1.0.0"):
    """Load a new model"""
    global model_server
    
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    try:
        model_info = model_server.load_model(
            model_path=model_path,
            model_name=model_name,
            version=model_version
        )
        
        return {
            "status": "success",
            "message": f"Loaded model from {model_path}",
            "model_key": f"{model_name}:{model_version}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Load model error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== PREDICTION ENDPOINTS ====================

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    """Make single prediction"""
    if model_server is None or model_server.current_bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Convert input to dataframe
        df = pd.DataFrame([data.dict()])
        
        # Get predictions with confidence
        result = model_server.predict_with_confidence(df)
        
        return PredictionResponse(
            User_ID=data.User_ID,
            prediction=int(result['predictions'][0]),
            confidence=float(result['confidence'][0]),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(data: BatchPredictionInput):
    """Make batch predictions"""
    if model_server is None or model_server.current_bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Convert inputs to dataframe
        df = pd.DataFrame([p.dict() for p in data.predictions])
        
        # Get predictions with confidence
        result = model_server.predict_with_confidence(df)
        
        responses = []
        for i, pred_input in enumerate(data.predictions):
            responses.append({
                "User_ID": pred_input.User_ID,
                "prediction": int(result['predictions'][i]),
                "confidence": float(result['confidence'][i]),
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "count": len(responses),
            "predictions": responses
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Make predictions from CSV file (test.csv format)
    
    Args:
        file: CSV file with test data
        
    Returns:
        Predictions with User_ID
    """
    if model_server is None or model_server.current_bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Extract User_ID for response
        user_ids = df['User_ID'].tolist() if 'User_ID' in df.columns else list(range(len(df)))
        
        # Get predictions
        result = model_server.predict_with_confidence(df)
        
        responses = []
        for i, user_id in enumerate(user_ids):
            responses.append({
                "User_ID": user_id,
                "prediction": int(result['predictions'][i]),
                "confidence": float(result['confidence'][i]),
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "file": file.filename,
            "count": len(responses),
            "predictions": responses
        }
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
async def get_required_features():
    """Get list of required input features"""
    if model_server is None or model_server.current_preprocessor is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        features = model_server.get_required_features()
        
        return {
            "count": len(features),
            "features": features
        }
    except Exception as e:
        logger.error(f"Get features error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
