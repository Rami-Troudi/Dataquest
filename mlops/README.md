# MLOps Docker Deployment - Insurance Model API

Complete production-ready MLOps infrastructure for serving ML models via Docker and REST API.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Documentation](#api-documentation)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (includes Docker and Docker Compose)
- Git

### Start Services (Windows)
```batch
deploy.bat up
```

### Start Services (Linux/Mac)
```bash
bash deploy.sh up
```

### Access API
- **API Endpoint**: http://localhost
- **Interactive Docs**: http://localhost/docs
- **Health Check**: http://localhost/health

## 🏗️ Architecture

```
Internet
   │
   ├─ HTTP  (Port 80/443)
   │
   └─── Nginx (Reverse Proxy)
        ├─ Load Balancing
        ├─ SSL/TLS Termination
        ├─ Rate Limiting
        └─ Request Routing
             │
             └─── FastAPI (Port 8000)
                  ├─ Request Validation
                  ├─ Preprocessing
                  ├─ Model Serving
                  └─ Response Formatting
                       │
                       └─── LightGBM Model
                            └─ model.joblib

```

## 📦 Components

### `app.py`
FastAPI application with:
- Single prediction endpoint
- Batch prediction endpoint
- Health check endpoint
- Automatic API documentation (Swagger UI)

### `Dockerfile`
Multi-stage Docker build:
- ~400MB total image size
- Optimized for production
- Security hardened
- Health checks enabled

### `docker-compose.yml`
Orchestrates:
- MLOps API service
- Nginx reverse proxy
- Volume management
- Network configuration

### `requirements-docker.txt`
All Python dependencies with pinned versions

## 🖥️ Local Deployment

### 1. Build Image
```bash
docker build -t insurance-mlops:latest .
```

### 2. Run Container
```bash
docker run -d \
  --name insurance-mlops-api \
  -p 8000:8000 \
  -v $(pwd)/model.joblib:/app/model.joblib:ro \
  insurance-mlops:latest
```

### 3. Test Prediction
```python
import requests

response = requests.post('http://localhost/predict', json={
    "User_ID": "USER_001",
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
})

print(response.json())
```

## ☁️ Cloud Deployment

### Azure Container Instances
```bash
az group create --name insurance-mlops --location eastus

az container create \
  --resource-group insurance-mlops \
  --name insurance-api \
  --image insurance-mlops:latest \
  --ports 80 \
  --memory 2 \
  --cpu 0.5
```

### AWS Elastic Container Service (ECS)
```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag insurance-mlops:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest

# Create task definition and service in AWS Console
```

### Google Cloud Run
```bash
# Build in Cloud Build
gcloud builds submit --tag gcr.io/my-project/insurance-mlops

# Deploy to Cloud Run
gcloud run deploy insurance-api \
  --image gcr.io/my-project/insurance-mlops \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 60s \
  --max-instances 100
```

### Heroku
```bash
# Create app
heroku create insurance-mlops-api

# Push container
heroku container:push web
heroku container:release web

# Open app
heroku open
```

### DigitalOcean App Platform
```yaml
name: insurance-mlops
services:
- name: api
  github:
    branch: main
    repo: your-username/insurance-mlops
  build_command: docker build -t insurance-mlops .
  http_port: 8000
  run_command: uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

### Base URL
```
http://localhost  (Local)
https://api.yourdomain.com  (Production)
```

### Endpoints

#### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Single Prediction
```
POST /predict
Content-Type: application/json
```

Request:
```json
{
  "User_ID": "USER_001",
  "Adult_Dependents": 1,
  "Child_Dependents": 2,
  ...
}
```

Response:
```json
{
  "User_ID": "USER_001",
  "prediction": 0.85
}
```

#### Batch Prediction
```
POST /predict_batch
Content-Type: application/json
```

Request:
```json
{
  "predictions": [
    { "User_ID": "USER_001", ... },
    { "User_ID": "USER_002", ... }
  ]
}
```

Response:
```json
[
  { "User_ID": "USER_001", "prediction": 0.85 },
  { "User_ID": "USER_002", "prediction": 0.62 }
]
```

#### API Documentation UI
```
GET /docs
GET /redoc
```

## 📊 Monitoring

### View Logs
```bash
docker-compose logs -f mlops-api
```

### Container Statistics
```bash
docker stats insurance-mlops-api
```

### Verify Health
```bash
curl http://localhost/health
```

### Monitor in Production
- Use CloudWatch (AWS), Stackdriver (GCP), or Application Insights (Azure)
- Export metrics to Prometheus
- Setup alerting for model performance degradation

## 🔐 Security

### Current Settings
- Read-only model volume
- Non-root container execution
- Health checks enabled
- Resource limits configured

### Recommended Enhancements
1. **Add API Authentication**
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict(data: PredictionInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify JWT or API key
    ...
```

2. **Enable HTTPS**
- Modify nginx.conf with SSL certificates
- Use Let's Encrypt for free certificates

3. **Rate Limiting**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(data: PredictionInput):
    ...
```

4. **Request Logging**
- Store audit logs
- Monitor for suspicious patterns

## 🐛 Troubleshooting

### Issue: Container won't start
```bash
docker logs insurance-mlops-api
```
Check for model file, permissions, or missing dependencies.

### Issue: Slow predictions
- Check Docker stats: `docker stats`
- Monitor model file I/O
- Consider batch processing

### Issue: High memory usage
- Reduce replica count
- Optimize preprocessing
- Enable memory limits in docker-compose

### Issue: Connection refused
```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart
```

## 📝 Example Client Usage

### Python
```python
from client import MLOpsAPIClient, PredictionData

client = MLOpsAPIClient("http://localhost")

data = PredictionData(
    user_id="USER_001",
    adult_dependents=1,
    # ... other fields
)

result = client.predict(data)
print(f"Prediction: {result['prediction']}")
```

### cURL
```bash
curl -X POST "http://localhost/predict" \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ User_ID: 'USER_001', ... })
});

const result = await response.json();
console.log(result.prediction);
```

## 📈 Performance Benchmarks

- **Single Prediction**: ~50-100ms
- **Batch Prediction (100)**: ~500-1000ms
- **Throughput**: ~100-200 req/sec per container
- **Memory Usage**: ~300-500MB per container
- **Startup Time**: ~5-10 seconds

## 🔄 Model Update Process

1. Train new model locally
2. Save as `model.joblib`
3. Replace in mlops folder
4. Rebuild Docker image: `docker-compose build --no-cache`
5. Restart services: `docker-compose up -d`
6. Verify with health check

## 📞 Support

For issues or feature requests:
1. Check logs: `docker-compose logs`
2. Review DEPLOYMENT_GUIDE.md
3. Test with client.py
4. Check API docs at /docs

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**Maintained By**: MLOps Team
