# MLOps Docker Deployment Guide

## Overview
This MLOps deployment provides a production-ready REST API for serving insurance model predictions via Docker containers.

## Architecture

```
┌─────────────────┐
│   Client/User   │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Nginx   │  (Reverse Proxy, Load Balancer)
    │  :80/:443│
    └────┬─────┘
         │
    ┌────▼──────────────┐
    │  FastAPI App      │
    │  :8000            │
    │  - /predict       │
    │  - /predict_batch │
    │  - /health        │
    │  - /docs          │
    └────┬──────────────┘
         │
    ┌────▼────────┐
    │  Model File │
    │ model.joblib│
    └─────────────┘
```

## Requirements
- Docker >= 20.10
- Docker Compose >= 1.29
- Git (for version control)

## Quick Start

### 1. Build and Run with Docker Compose
```bash
cd mlops
docker-compose up -d
```

### 2. Verify Deployment
```bash
# Check health
curl http://localhost/health

# View API documentation
curl http://localhost http://localhost/docs
```

### 3. Make Predictions

**Single Prediction:**
```bash
curl -X POST "http://localhost/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "User_ID": "USER_001",
        "Adult_Dependents": 1,
        "Child_Dependents": 2,
        ...
      },
      {
        "User_ID": "USER_002",
        "Adult_Dependents": 0,
        "Child_Dependents": 1,
        ...
      }
    ]
  }'
```

## Docker Commands

### Build Image
```bash
docker build -t insurance-mlops:latest .
```

### Run Container
```bash
docker run -d \
  --name insurance-mlops-api \
  -p 8000:8000 \
  -v $(pwd)/model.joblib:/app/model.joblib:ro \
  insurance-mlops:latest
```

### View Logs
```bash
docker logs -f insurance-mlops-api
```

### Stop Container
```bash
docker stop insurance-mlops-api
docker rm insurance-mlops-api
```

## Docker Compose Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f mlops-api
```

### Rebuild Images
```bash
docker-compose build --no-cache
```

## Deployment Environments

### Local Development
```bash
docker-compose -f docker-compose.yml up -d
```

### Production (AWS ECS, Google Cloud Run, etc.)
The image can be pushed to any container registry:

```bash
# Tag for registry
docker tag insurance-mlops:latest myregistry.azurecr.io/insurance-mlops:latest

# Push to registry
docker push myregistry.azurecr.io/insurance-mlops:latest
```

## Environment Variables
Set in docker-compose.yml or during container run:

- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `PYTHONDONTWRITEBYTECODE=1` - Don't write .pyc files

## Monitoring & Health Checks

### Health Endpoint
```bash
GET /health
```

Returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### API Documentation
Interactive Swagger UI:
```
http://localhost/docs
```

## Performance Optimization

1. **Multi-stage Build**: Reduces image size
2. **Slim Base Image**: Uses python:3.11-slim
3. **Read-only Model**: model.joblib mounted as read-only
4. **Health Checks**: Automatic container restart on failure
5. **Batch Predictions**: Efficient bulk processing

## Security Considerations

1. Use environment-specific `.env` files
2. Enable SSL/TLS in production (Nginx configuration ready)
3. Use Docker secrets for sensitive data
4. Restrict image registry access
5. Regular image scanning for vulnerabilities

## Scaling

### Horizontal Scaling (Multiple Containers)
```yaml
services:
  mlops-api:
    replicas: 3
```

### With Orchestration (Kubernetes)
```bash
kubectl apply -f k8s-deployment.yaml
```

## Troubleshooting

### Container won't start
```bash
docker logs insurance-mlops-api
```

### Model not loading
- Check model.joblib exists in the container
- Verify file permissions
- Check container logs

### Slow predictions
- Monitor CPU/Memory: `docker stats`
- Check input data size
- Consider batch predictions
- Scale horizontally

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

## API Response Examples

### Successful Prediction
```json
{
  "User_ID": "USER_001",
  "prediction": 0.85
}
```

### Error Response
```json
{
  "detail": "Prediction failed: preprocessing error"
}
```

## Further Enhancements

1. Add authentication (API keys, OAuth2)
2. Implement rate limiting
3. Add request/response caching
4. Implement model versioning
5. Add Prometheus metrics
6. Setup ELK logging stack
7. Add request tracing (OpenTelemetry)
8. Implement A/B testing framework

## Contact & Support
For issues or questions, open a GitHub issue or contact the MLOps team.
