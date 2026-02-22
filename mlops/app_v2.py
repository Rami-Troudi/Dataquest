"""
MLOps REST API with Model Registry & Feature Management
Enhanced version with easy model switching and flexible feature handling
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from model_registry import get_registry, ModelRegistry
from feature_manager import get_feature_manager, FeatureManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Advanced REST API with model registry and feature management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
registry: Optional[ModelRegistry] = None
feature_manager: Optional[FeatureManager] = None


# ==================== DATA MODELS ====================

class PredictionInput(BaseModel):
    """Flexible prediction input - accepts any features"""
    User_ID: str
    features: Dict[str, Any]
    model_name: Optional[str] = None
    model_version: Optional[str] = "latest"


class BatchPredictionInput(BaseModel):
    """Batch prediction"""
    predictions: List[PredictionInput]


class PredictionResponse(BaseModel):
    """Prediction response"""
    User_ID: str
    model_used: str
    prediction: float
    confidence: Optional[float] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information"""
    name: str
    version: str
    type: str
    status: str
    accuracy: Optional[float]
    features_count: int
    description: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    current_model: Optional[str]
    models_loaded: int
    version: str


# ==================== PREPROCESSING ====================

def preprocess_insurance_data(df):
    """
    Preprocess insurance data - original preprocessing logic
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


# ==================== INITIALIZATION ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global registry, feature_manager
    
    logger.info("Starting MLOps API v2.0...")
    
    try:
        # Initialize registry
        registry = get_registry(models_dir="models")
        logger.info(f"Registry initialized with {len(registry.registry)} models")
        
        # Initialize feature manager
        feature_manager = get_feature_manager(config_file="features_config.json")
        logger.info(f"Feature manager initialized with {len(feature_manager.feature_map)} features")
        
        logger.info("MLOps API ready for predictions")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


# ==================== CORE API ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    current = None
    if registry and registry.current_model:
        current = registry.current_model
    
    return HealthResponse(
        status="healthy" if registry and len(registry.loaded_models) > 0 else "unhealthy",
        current_model=current,
        models_loaded=len(registry.loaded_models) if registry else 0,
        version="2.0.0"
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "MLOps Model Serving API",
        "version": "2.0.0",
        "features": {
            "model_registry": "Easy model switching",
            "feature_management": "Centralized feature config",
            "flexible_input": "Accept any features",
            "batch_processing": "Efficient bulk predictions"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "models": "/models",
            "models/switch": "/models/switch",
            "predict": "/predict",
            "predict_batch": "/predict_batch"
        }
    }


# ==================== MODEL MANAGEMENT ====================

@app.get("/models")
async def list_models(status: Optional[str] = None):
    """List all available models"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    
    try:
        models = registry.list_models(status=status)
        return {
            "total": len(models),
            "current": registry.current_model,
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/{name}")
async def get_model_info(name: str, version: str = "latest"):
    """Get information about a specific model"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    
    try:
        key = f"{name}:{version}"
        if key not in registry.registry:
            raise ValueError(f"Model not found: {key}")
        
        metadata = registry.registry[key]
        features = metadata.features
        
        return ModelInfoResponse(
            name=metadata.name,
            version=metadata.version,
            type=metadata.model_type,
            status=metadata.status,
            accuracy=metadata.accuracy,
            features_count=len(features),
            description=metadata.description
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/switch")
async def switch_model(model_name: str, model_version: str = "latest"):
    """Switch to a different model"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    
    try:
        registry.set_current_model(model_name, model_version)
        return {
            "status": "success",
            "current_model": registry.current_model,
            "message": f"Switched to {model_name}:{model_version}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== PREDICTION ENDPOINTS ====================

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    """Make a single prediction"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    
    try:
        # Get model
        if data.model_name:
            model_wrapper = registry.load_model(data.model_name, data.model_version or "latest")
            model_key = f"{data.model_name}:{data.model_version}"
        else:
            if not registry.current_model:
                raise ValueError("No current model set. Set model with /models/switch")
            model_wrapper = registry.get_current_model()
            model_key = registry.current_model
        
        # Prepare data
        df = pd.DataFrame([data.features])
        
        # Preprocess (using default insurance preprocessing)
        df_processed = preprocess_insurance_data(df)
        
        # Get required features for model
        name, version = model_key.split(":")
        required_features = registry.get_model_features(name, version)
        
        # Select only required features
        X = df_processed[required_features]
        
        # Predict
        prediction = model_wrapper.predict(X)[0]
        
        return PredictionResponse(
            User_ID=data.User_ID,
            model_used=model_key,
            prediction=float(prediction),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(data: BatchPredictionInput):
    """Make batch predictions"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    
    try:
        results = []
        
        # Get model (use first prediction's model or current)
        if data.predictions[0].model_name:
            model_wrapper = registry.load_model(
                data.predictions[0].model_name,
                data.predictions[0].model_version or "latest"
            )
            model_key = f"{data.predictions[0].model_name}:{data.predictions[0].model_version}"
        else:
            if not registry.current_model:
                raise ValueError("No current model set")
            model_wrapper = registry.get_current_model()
            model_key = registry.current_model
        
        # Prepare batch data
        df = pd.DataFrame([p.features for p in data.predictions])
        
        # Preprocess
        df_processed = preprocess_insurance_data(df)
        
        # Get required features
        name, version = model_key.split(":")
        required_features = registry.get_model_features(name, version)
        X = df_processed[required_features]
        
        # Predict
        predictions = model_wrapper.predict(X)
        
        # Build responses
        for i, pred_input in enumerate(data.predictions):
            results.append(PredictionResponse(
                User_ID=pred_input.User_ID,
                model_used=model_key,
                prediction=float(predictions[i]),
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== FEATURE ENDPOINTS ====================

@app.get("/features")
async def list_features(group: Optional[str] = None, feature_type: Optional[str] = None):
    """List all features"""
    if not feature_manager:
        raise HTTPException(status_code=503, detail="Feature manager not initialized")
    
    try:
        features = feature_manager.list_features()
        
        # Filter if needed
        if group:
            features = [f for f in features if f.get("group") == group]
        if feature_type:
            features = [f for f in features if f["type"] == feature_type]
        
        return {
            "total": len(features),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features/{feature_name}")
async def get_feature_info(feature_name: str):
    """Get information about a specific feature"""
    if not feature_manager:
        raise HTTPException(status_code=503, detail="Feature manager not initialized")
    
    try:
        config = feature_manager.get_feature_config(feature_name)
        if not config:
            raise ValueError(f"Feature not found: {feature_name}")
        
        return {
            "name": config.name,
            "type": config.feature_type,
            "required": config.required,
            "description": config.description,
            "transformation": config.transformation,
            "validation_rules": config.validation_rules
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
